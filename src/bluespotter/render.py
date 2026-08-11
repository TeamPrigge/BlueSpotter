"""Render the assertion set as pictures next to its numbers, for review on GitHub.

WHY BOTHER, WHEN THERE ARE ALREADY METRICS
------------------------------------------
F1 = 0.83 does not tell you *how* the model is wrong, and the two ways it can be
wrong want opposite fixes. Fusing touching somata and inventing cells in
background both cost you F1; only one of them is fixed by more training data.
A person looking at three panels sees the difference in a second, and no scalar
in `segmentation_metrics.json` will ever say it.

So this writes `reports/QUALITY.md`: one row per assertion case, three panels —

    image   |   ground truth   |   model prediction

with the per-case numbers beside them. GitHub renders it directly, so a pull
request that changes the model shows what changed, not just that something did.
Outlines rather than filled overlays, because a filled mask hides the pixels you
are trying to judge.

Colours are fixed and meaningful, not decorative:
  green   a predicted cell matched to a real one
  red     a predicted cell with no match — invented
  amber   a real cell the model missed

Run (Colab, needs the model):

    python -m bluespotter.render                 # with predictions
    python -m bluespotter.render --no-predict    # ground truth only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import match_instances, score_one

FIG_DIR = "figures"

MATCHED = (60, 200, 90)      # green
INVENTED = (230, 60, 60)     # red
MISSED = (245, 170, 40)      # amber
TRUTH = (70, 150, 240)       # blue, for the ground-truth panel


def _outline(mask: np.ndarray, label: int) -> np.ndarray:
    """Boolean edge of one instance, without pulling in scipy."""
    m = mask == label
    edge = np.zeros_like(m)
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[1:, :] |= m[:-1, :] != m[1:, :]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge[:, 1:] |= m[:, :-1] != m[:, 1:]
    return edge & m


def overlay(image8: np.ndarray, mask: np.ndarray,
            colours: dict[int, tuple[int, int, int]] | None = None,
            default: tuple[int, int, int] = TRUTH) -> np.ndarray:
    """RGB image with each instance outlined."""
    rgb = np.dstack([image8] * 3).astype(np.uint8)
    for label in np.unique(mask):
        if label == 0:
            continue
        edge = _outline(mask, int(label))
        rgb[edge] = (colours or {}).get(int(label), default)
    return rgb


def classify(true_mask: np.ndarray, pred_mask: np.ndarray,
             iou_threshold: float = 0.5) -> tuple[dict, dict]:
    """Colour every predicted and every true instance by whether it matched."""
    from .metrics import _iou_matrix

    iou = _iou_matrix(true_mask, pred_mask)
    true_ids = [int(v) for v in np.unique(true_mask) if v > 0]
    pred_ids = [int(v) for v in np.unique(pred_mask) if v > 0]

    pred_colours: dict[int, tuple[int, int, int]] = {p: INVENTED for p in pred_ids}
    true_colours: dict[int, tuple[int, int, int]] = {t: MISSED for t in true_ids}

    if iou.size:
        used_t: set[int] = set()
        used_p: set[int] = set()
        pairs = np.argwhere(iou >= iou_threshold)
        order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]]) if len(pairs) else []
        for k in order:
            t, p = pairs[k]
            if int(t) in used_t or int(p) in used_p:
                continue
            used_t.add(int(t))
            used_p.add(int(p))
            pred_colours[pred_ids[int(p)]] = MATCHED
            true_colours[true_ids[int(t)]] = MATCHED
    return true_colours, pred_colours


def _panel(*images: np.ndarray, gap: int = 6) -> np.ndarray:
    """Stack panels side by side on a light background."""
    h = max(im.shape[0] for im in images)
    parts = []
    for im in images:
        pad = np.full((h, im.shape[1], 3), 245, np.uint8)
        pad[: im.shape[0]] = im
        parts.append(pad)
        parts.append(np.full((h, gap, 3), 245, np.uint8))
    return np.hstack(parts[:-1])


def render_case(assertions: Path, case: dict, model: Any | None,
                out_dir: Path) -> dict[str, Any]:
    from skimage.io import imread, imsave

    image8 = imread(str(assertions / f"{case['name']}_image.png"))
    truth = imread(str(assertions / f"{case['name']}_masks.png")).astype(np.int32)

    panels = [np.dstack([image8] * 3).astype(np.uint8), overlay(image8, truth)]
    row: dict[str, Any] = {**case}

    if model is not None:
        pred, _flows, _styles = model.eval(image8, batch_size=1)
        pred = np.asarray(pred).astype(np.int32)
        true_colours, pred_colours = classify(truth, pred)
        panels[1] = overlay(image8, truth, true_colours)
        panels.append(overlay(image8, pred, pred_colours))

        s = score_one(truth, pred)
        tp, fp, fn = match_instances(truth, pred, 0.5)
        row.update(n_pred=s["n_pred"], tp=tp, fp=fp, fn=fn,
                   precision=s["per_threshold"]["0.5"]["precision"],
                   recall=s["per_threshold"]["0.5"]["recall"],
                   f1=s["per_threshold"]["0.5"]["f1"],
                   count_error=s["count_error"])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig = out_dir / f"{case['name']}.png"
    imsave(str(fig), _panel(*panels), check_contrast=False)
    row["figure"] = fig.name
    return row


def write_markdown(rows: list[dict[str, Any]], reports: Path,
                   contract: dict, model_desc: str, with_predictions: bool) -> Path:
    lines = [
        "# Assertion set — visual review",
        "",
        f"`{len(rows)}` case(s) from `{contract.get('n_animals', '?')}` animals, "
        f"`{contract.get('total_cells', '?')}` labelled cells. "
        f"Model: `{model_desc}`.",
        "",
        "Crops come from the held-out split, so nothing here was trained on. "
        "`whole_lc` cases are downscaled to show the whole nucleus; image and mask "
        "are scaled together and the model runs on the same downscaled image, so "
        "predictions and truth stay in the same space.",
        "",
    ]
    if with_predictions:
        lines += [
            "Panels: **image** | **ground truth** | **prediction**. "
            "Outlines are coloured "
            "🟩 matched · 🟥 invented (false positive) · 🟧 missed (false negative).",
            "",
            "| case | kind | cells | found | P | R | F1 | count err |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| [{r['name']}](#{r['name'].lower().replace('_', '-')}) | {r['kind']} | "
                f"{r['n_cells']} | {r.get('n_pred', '-')} | "
                f"{r.get('precision', float('nan')):.2f} | {r.get('recall', float('nan')):.2f} | "
                f"{r.get('f1', float('nan')):.2f} | {r.get('count_error', 0):+d} |")
    else:
        lines += ["Panels: **image** | **ground truth**. "
                  "No model was available, so there are no predictions to show.", ""]

    lines.append("")
    for r in rows:
        lines += [
            f"## {r['name']}",
            "",
            f"`{r['kind']}` · {r['mouse']} · {r['channel']} · {r['source']} · "
            f"{r['width']}x{r['height']}px"
            + (f" (scaled x{r['scale']:.2f})" if r.get("scale", 1.0) != 1.0 else "")
            + f" · {r['n_cells']} labelled cells",
            "",
            f"![{r['name']}]({FIG_DIR}/{r['figure']})",
            "",
        ]

    out = reports / "QUALITY.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=None)
    ap.add_argument("--assertions", default="assertions")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-predict", action="store_true",
                    help="render ground truth only, without loading a model")
    args = ap.parse_args(argv)

    from .config import load_config

    cfg = load_config(args.params)
    repo = cfg.repo_root
    assertions = repo / args.assertions
    contract_path = assertions / "expected.json"
    if not contract_path.exists():
        print(f"[render] no assertion set at {assertions} — "
              f"run `python -m bluespotter.make_assertions` first")
        return 1
    contract = json.loads(contract_path.read_text())

    model, model_desc = None, "none (ground truth only)"
    if not args.no_predict:
        weights = Path(args.model) if args.model else cfg.model_dir / cfg.model["name"]
        if weights.exists():
            from cellpose import models

            model = models.CellposeModel(gpu=True, pretrained_model=str(weights))
            model_desc = weights.name
        else:
            print(f"[render] no weights at {weights} — rendering ground truth only")

    reports = repo / cfg.dvc.get("report_dir", "reports")
    rows = [render_case(assertions, c, model, reports / FIG_DIR)
            for c in contract["cases"]]
    out = write_markdown(rows, reports, contract, model_desc, model is not None)

    print(f"[render] {len(rows)} panel(s) -> {reports / FIG_DIR}")
    print(f"[render] page -> {out.relative_to(repo)}")
    if model is not None:
        worst = min(rows, key=lambda r: r.get("f1", 1.0))
        print(f"[render] weakest case: {worst['name']} (F1 {worst.get('f1', 0):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
