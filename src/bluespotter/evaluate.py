"""Segment the held-out split and score it. DVC stage; needs a GPU, so Colab.

This is the stage that turns "the training loss went down" into "the model finds
92% of the neurons a human marked, and invents 4% that are not there". Until it
runs, nothing in the repo can tell you whether the model is any good — a loss
curve only says the optimiser is working.

It writes three things:

  reports/segmentation_metrics.json   scalars only, so `dvc metrics show` and
                                      `dvc metrics diff` work on a pull request
  reports/segmentation_detail.json    per-image and per-mouse breakdown
  reports/pr_curve.csv                precision/recall vs IoU threshold, plotted
                                      by `dvc plots show`

The per-mouse breakdown matters more than it looks. With 20 held-out animals, a
single mean hides the case that actually bites: the model works on 19 animals and
falls over on one staining batch. For a platform whose whole claim is
cross-laboratory comparability, "works on average" is not the property being
sold.

Run:

    python -m bluespotter.evaluate                    # the registered model
    python -m bluespotter.evaluate --model path.pth   # a specific checkpoint
    python -m bluespotter.evaluate --limit 20         # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, load_config
from .metrics import DEFAULT_IOU_THRESHOLDS, aggregate, precision_recall_curve, score_one


def _load_pairs(cfg: Config, split: str, limit: int | None) -> list[tuple[dict, Any, Any]]:
    """Read (row, image, true_mask) for a split, one at a time.

    Deliberately not the bulk loader used for training: the test split is 55 GiB
    and we only need one image in memory at a time to score it.
    """
    from .manifest import load_pair

    manifest = cfg.repo_manifest(split)
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found — run `dvc repro sync-manifests` first")

    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit:
        rows = rows[:limit]

    out = []
    for row in rows:
        try:
            image, mask = load_pair(cfg.nmslices_root, row)
        except Exception as exc:
            print(f"    [skip] {row.get('image_name', '?')}: {type(exc).__name__} {exc}")
            continue
        out.append((row, image, mask))
    return out


def evaluate(cfg: Config, model: Any, split: str = "test",
             limit: int | None = None,
             iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
             ) -> dict[str, Any]:
    """Segment every image in `split` and score against its ground truth."""
    pairs = _load_pairs(cfg, split, limit)
    print(f"  scoring {len(pairs)} image(s) from {split}")

    per_image: list[dict[str, Any]] = []
    for i, (row, image, true_mask) in enumerate(pairs, 1):
        pred_mask, _flows, _styles = model.eval(image, batch_size=1)
        score = score_one(true_mask, np.asarray(pred_mask).astype(np.int32),
                          iou_thresholds)
        score["image_name"] = row.get("image_name", "")
        score["mouse"] = row.get("mouse", "")
        score["source"] = row.get("source", "")
        score["channel"] = row.get("channel", "")
        per_image.append(score)
        if i % 10 == 0:
            print(f"    scored {i}/{len(pairs)}")

    overall = aggregate(per_image, iou_thresholds)

    # Per-mouse: the breakdown that shows whether the mean is honest.
    by_mouse_images: dict[str, list] = defaultdict(list)
    for s in per_image:
        by_mouse_images[s["mouse"]].append(s)
    by_mouse = {m: aggregate(v, iou_thresholds) for m, v in sorted(by_mouse_images.items())}

    # Worst animal at the headline threshold — the number a reviewer will ask for
    # and the one a mean would have hidden.
    key = f"f1@{iou_thresholds[0]:g}"
    if by_mouse:
        worst = min(by_mouse.items(), key=lambda kv: kv[1][key])
        overall["worst_mouse"] = worst[0]
        overall[f"worst_mouse_{key}"] = worst[1][key]

    return {
        "overall": overall,
        "by_mouse": by_mouse,
        "per_image": per_image,
        "pr_curve": precision_recall_curve(per_image),
    }


def _write_reports(cfg: Config, result: dict[str, Any], split: str,
                   model_desc: str) -> Path:
    repo = cfg.repo_root
    rdir = repo / cfg.dvc.get("report_dir", "reports")
    rdir.mkdir(parents=True, exist_ok=True)

    # Scalars only in the metrics file: `dvc metrics show` flattens nested JSON
    # and a per-mouse dict would produce a table hundreds of columns wide.
    metrics = {k: v for k, v in result["overall"].items()
               if isinstance(v, (int, float, str))}
    metrics["split"] = split
    metrics["model"] = model_desc

    # Stamp the dataset this was scored on, so CI can refuse metrics that
    # describe data the repo no longer pins.
    summary = rdir / f"{split}_index_summary.json"
    if summary.exists():
        metrics[f"dataset_hash_{split}"] = json.loads(
            summary.read_text()).get("dataset_hash", "")

    (rdir / "segmentation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (rdir / "segmentation_detail.json").write_text(json.dumps(
        {"by_mouse": result["by_mouse"], "per_image": result["per_image"]}, indent=2) + "\n")

    with open(rdir / "pr_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["iou_threshold", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(result["pr_curve"])

    return rdir / "segmentation_metrics.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=None)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--model", default=None,
                    help="path to weights (default: model.name in the Drive model dir)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N images (smoke test)")
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.params)

    from cellpose import models

    weights = Path(args.model) if args.model else cfg.model_dir / cfg.model["name"]
    if not weights.exists():
        print(f"[evaluate] FAILED: no weights at {weights}. Train first, or pass --model.")
        return 1

    print(f"[evaluate] model  : {weights}")
    print(f"[evaluate] split  : {args.split}")
    model = models.CellposeModel(gpu=True, pretrained_model=str(weights))

    result = evaluate(cfg, model, split=args.split, limit=args.limit)
    out = _write_reports(cfg, result, args.split, str(weights.name))

    o = result["overall"]
    print(f"\n  images        : {o['n_images']}")
    print(f"  cells         : {o['n_true_total']} true / {o['n_pred_total']} predicted "
          f"({o['count_bias_pct']:+.1f}% bias)")
    for tau in DEFAULT_IOU_THRESHOLDS:
        k = f"{tau:g}"
        print(f"  IoU>={k:<4}     precision {o[f'precision@{k}']:.3f}   "
              f"recall {o[f'recall@{k}']:.3f}   F1 {o[f'f1@{k}']:.3f}   "
              f"AP {o[f'ap@{k}']:.3f}")
    if "worst_mouse" in o:
        print(f"  worst animal  : {o['worst_mouse']} "
              f"(F1@0.5 = {o['worst_mouse_f1@0.5']:.3f})")
    print(f"  reports       -> {out.parent}")

    # Best-effort MLflow logging: a failure to reach the tracking server should
    # not lose the report we just computed on disk.
    try:
        import mlflow

        from .mlflow_utils import start_tracking

        start_tracking(cfg)
        with mlflow.start_run(run_name=f"eval_{args.split}_{weights.stem}"):
            mlflow.log_metrics({k: v for k, v in o.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(out))
            mlflow.log_artifact(str(out.parent / "pr_curve.csv"))
    except Exception as exc:
        print(f"  (MLflow logging skipped: {type(exc).__name__} {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
