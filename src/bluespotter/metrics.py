"""Segmentation quality metrics for instance masks.

WHY NOT ROC/AUC
---------------
The obvious ask for "a quality number" is ROC/AUC, and it is the wrong tool
here. ROC needs a false-positive *rate*, which needs true negatives. In instance
segmentation the negative class is every region that is not a cell — unbounded,
and overwhelmingly background. FPR therefore sits near zero whatever the model
does, and AUC reads ~0.98 for a good model and a bad one alike. It would give
you a confident number that cannot distinguish the thing you care about.

What does work is instance matching: pair each predicted mask with a ground-truth
mask by intersection-over-union, then count.

  TP  a prediction matched to a GT mask at IoU >= tau
  FP  a prediction with no match          (a hallucinated cell)
  FN  a GT mask with no match             (a missed cell)

  precision = TP / (TP + FP)    "of the cells it found, how many were real"
  recall    = TP / (TP + FN)    "of the real cells, how many did it find"
  F1        = harmonic mean
  AP        = TP / (TP + FP + FN)   -- the Cellpose/StarDist convention

That last one is worth flagging: "average precision" in the segmentation
literature (and in `cellpose.metrics.average_precision`) is *not* the area under
a precision-recall curve. It is a single ratio that penalises both misses and
hallucinations at once. Reporting it under the same name as the detection-style
AP is a well-established confusion; we compute the Cellpose definition so numbers
are comparable to that paper, and also give precision/recall separately so
nothing is hidden inside a composite.

ON MATCHING BEING GREEDY
------------------------
At tau >= 0.5 a predicted mask can exceed IoU 0.5 with at most one ground-truth
mask (two would have to overlap each other by more than half, and instance masks
are disjoint). So the assignment is unique and greedy matching is exact — no
scipy, no Hungarian algorithm, and CI can run this with numpy alone. Below 0.5
that guarantee fails, which is one reason the field evaluates at 0.5 and above.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Thresholds the Cellpose paper reports, so our numbers sit next to theirs.
DEFAULT_IOU_THRESHOLDS = (0.5, 0.75, 0.9)


def _iou_matrix(true_masks: np.ndarray, pred_masks: np.ndarray) -> np.ndarray:
    """IoU between every GT instance and every predicted instance.

    Both inputs are label images: 0 is background, each positive integer is one
    instance. Returns an array of shape (n_true, n_pred).
    """
    true_ids = np.unique(true_masks)
    pred_ids = np.unique(pred_masks)
    true_ids = true_ids[true_ids > 0]
    pred_ids = pred_ids[pred_ids > 0]

    if len(true_ids) == 0 or len(pred_ids) == 0:
        return np.zeros((len(true_ids), len(pred_ids)), dtype=float)

    # One pass over the pixels via a joint histogram, rather than n_true*n_pred
    # boolean intersections — on a 4000x4000 slice with 300 cells the naive form
    # is minutes and this is milliseconds.
    t_index = {v: i for i, v in enumerate(true_ids)}
    p_index = {v: i for i, v in enumerate(pred_ids)}

    t_flat = true_masks.ravel()
    p_flat = pred_masks.ravel()
    both = (t_flat > 0) & (p_flat > 0)

    overlap = np.zeros((len(true_ids), len(pred_ids)), dtype=np.int64)
    if both.any():
        ti = np.fromiter((t_index[v] for v in t_flat[both]), dtype=np.int64,
                         count=int(both.sum()))
        pi = np.fromiter((p_index[v] for v in p_flat[both]), dtype=np.int64,
                         count=int(both.sum()))
        np.add.at(overlap, (ti, pi), 1)

    t_area = np.array([(true_masks == v).sum() for v in true_ids], dtype=np.int64)
    p_area = np.array([(pred_masks == v).sum() for v in pred_ids], dtype=np.int64)

    union = t_area[:, None] + p_area[None, :] - overlap
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, overlap / union, 0.0)
    return iou


def match_instances(true_masks: np.ndarray, pred_masks: np.ndarray,
                    iou_threshold: float = 0.5) -> tuple[int, int, int]:
    """Return (tp, fp, fn) at `iou_threshold`.

    Greedy, highest-IoU-first. Exact for thresholds >= 0.5 (see module docstring);
    for lower thresholds it is a lower bound on the optimal assignment, which is
    the safe direction to be wrong in when reporting quality.
    """
    iou = _iou_matrix(true_masks, pred_masks)
    n_true, n_pred = iou.shape

    tp = 0
    if iou.size:
        used_true: set[int] = set()
        used_pred: set[int] = set()
        # Sort candidate pairs by IoU descending, take each if both ends free.
        pairs = np.argwhere(iou >= iou_threshold)
        order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]]) if len(pairs) else []
        for k in order:
            t, p = pairs[k]
            if t in used_true or p in used_pred:
                continue
            used_true.add(int(t))
            used_pred.add(int(p))
            tp += 1

    return tp, n_pred - tp, n_true - tp


def score_one(true_masks: np.ndarray, pred_masks: np.ndarray,
              iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
              ) -> dict[str, Any]:
    """Score a single image. Returns per-threshold counts plus cell counts."""
    true_masks = np.asarray(true_masks)
    pred_masks = np.asarray(pred_masks)

    n_true = int(len(np.unique(true_masks)) - (1 if (true_masks == 0).any() else 0))
    n_pred = int(len(np.unique(pred_masks)) - (1 if (pred_masks == 0).any() else 0))

    per_threshold = {}
    for tau in iou_thresholds:
        tp, fp, fn = match_instances(true_masks, pred_masks, tau)
        per_threshold[f"{tau:g}"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            # Cellpose's "average precision": one number that punishes both a
            # missed cell and an invented one.
            "ap": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        }

    return {
        "n_true": n_true,
        "n_pred": n_pred,
        # Signed, so systematic over- or under-counting is visible. The lab's
        # actual readout is a cell count, so this is the number that matters most
        # even when IoU looks fine.
        "count_error": n_pred - n_true,
        "per_threshold": per_threshold,
    }


def aggregate(per_image: list[dict[str, Any]],
              iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
              ) -> dict[str, Any]:
    """Pool per-image scores into dataset-level metrics.

    Counts are pooled before dividing, not averaged per image. Averaging ratios
    would let a slice with three neurons weigh as much as one with three hundred,
    which for a structure whose size varies along the rostrocaudal axis would
    quietly bias the score toward the small end of the LC.
    """
    out: dict[str, Any] = {"n_images": len(per_image)}
    for tau in iou_thresholds:
        key = f"{tau:g}"
        tp = sum(s["per_threshold"][key]["tp"] for s in per_image)
        fp = sum(s["per_threshold"][key]["fp"] for s in per_image)
        fn = sum(s["per_threshold"][key]["fn"] for s in per_image)
        out[f"precision@{key}"] = tp / (tp + fp) if (tp + fp) else 0.0
        out[f"recall@{key}"] = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"f1@{key}"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out[f"ap@{key}"] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        out[f"tp@{key}"], out[f"fp@{key}"], out[f"fn@{key}"] = tp, fp, fn

    total_true = sum(s["n_true"] for s in per_image)
    total_pred = sum(s["n_pred"] for s in per_image)
    out["n_true_total"] = total_true
    out["n_pred_total"] = total_pred
    out["count_error_total"] = total_pred - total_true
    out["count_error_mean_abs"] = (
        float(np.mean([abs(s["count_error"]) for s in per_image])) if per_image else 0.0
    )
    # Relative count bias is what a downstream paper would actually quote.
    out["count_bias_pct"] = (
        100.0 * (total_pred - total_true) / total_true if total_true else 0.0
    )
    return out


def precision_recall_curve(per_image: list[dict[str, Any]],
                           thresholds: tuple[float, ...] | None = None,
                           ) -> list[dict[str, float]]:
    """Precision and recall as a function of IoU threshold.

    This is the curve worth plotting for instance segmentation: it shows how fast
    quality degrades as you demand tighter boundary agreement, which is exactly
    the question "are the outlines good or merely in the right place".
    """
    thresholds = thresholds or tuple(round(t, 2) for t in np.arange(0.5, 1.0, 0.05))
    rows = []
    for tau in thresholds:
        key = f"{tau:g}"
        available = [s for s in per_image if key in s["per_threshold"]]
        if not available:
            continue
        tp = sum(s["per_threshold"][key]["tp"] for s in available)
        fp = sum(s["per_threshold"][key]["fp"] for s in available)
        fn = sum(s["per_threshold"][key]["fn"] for s in available)
        rows.append({
            "iou_threshold": float(tau),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        })
    return rows
