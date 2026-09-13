"""Export the held-out split for independent human counting.

WHAT THIS IS FOR
----------------
Three students plus the original annotator count neurons in the same sections,
blind. That gives the **human ceiling**: if experienced scorers agree with each
other to within ~10%, a model that lands inside that band is performing at human
level, and chasing a tighter number is chasing label noise.

It also replaces guesswork in the quality gate. `params.yaml` currently asserts
`count_bias_pct <= 10` — a number picked with no empirical basis whatsoever.
Inter-rater spread measured on these 248 sections replaces it with one that can
be defended in a methods section.

COUNTS, NOT OUTLINES
--------------------
Scorers report a single integer per section. No ROIs. That means agreement is
computed on counts (ICC, Bland-Altman, coefficient of variation) and not IoU or
F1 — a rater could in principle count the right number in the wrong places and
look perfect. That gap is covered separately: the existing hand-drawn masks give
us outline-level ground truth already. What we do not have, and what this
supplies, is how much *counting* varies between people.

RESOLUTION IS THE WHOLE GAME
----------------------------
A whole-LC view downscaled to fit a fixed box puts somata at 3 px across, and
nobody can count that — the assertion set demonstrated exactly this failure at
x0.15. So the scale is not fixed. Each section's median soma diameter is
measured from its mask, and the scale is chosen to put that at
`TARGET_SOMA_PX`. Sections that cannot reach it without exceeding `MAX_PX` are
exported at their best achievable scale and flagged in the key, so a section
that turns out to be hard to count is a known quantity rather than a mystery.

BLINDING
--------
Sections get opaque IDs (LC-001 ...) in a shuffled order, so a scorer cannot
infer animal, cohort, channel or slide position from the filename, and cannot
tell which sections neighbour each other. The mapping and the ground-truth
counts go to `key.csv`, which is for you and **must not** be shared with the
scorers: a rater who can see the existing count is measuring their agreement
with it, not counting independently.

Run in Colab:

    python -m bluespotter.scoring_set                 # all 248
    python -m bluespotter.scoring_set --limit 10      # trial run first
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
from pathlib import Path

import numpy as np

TARGET_SOMA_PX = 15      # a soma this wide is comfortably countable on screen
MIN_SOMA_PX = 8          # below this, flag the section as hard to count
MAX_PX = 4000            # longest side of an exported image
SEED = 20260913          # fixed, so re-running reproduces the same IDs


def soma_diameter(mask: np.ndarray) -> float:
    """Median equivalent diameter of the labelled cells, in pixels.

    Equivalent diameter (from area) rather than a bounding box, because somata
    are roughly round and a bbox inflates any cell that sits diagonally.
    """
    ids, counts = np.unique(mask[mask > 0], return_counts=True)
    if len(ids) == 0:
        return 0.0
    return float(np.median(2.0 * np.sqrt(counts / np.pi)))


def choose_scale(mask: np.ndarray) -> tuple[float, bool]:
    """Scale that makes the median soma TARGET_SOMA_PX wide. (scale, ok)

    Never upscales: inventing pixels does not make a cell easier to see, it just
    makes a bigger blur. `ok` is False when the size cap forces somata below
    MIN_SOMA_PX, i.e. the section is genuinely hard to count.
    """
    d = soma_diameter(mask)
    if d <= 0:
        return 1.0, False
    scale = min(1.0, TARGET_SOMA_PX / d)

    longest = max(mask.shape[:2])
    if longest * scale > MAX_PX:
        scale = MAX_PX / longest
    return scale, (d * scale) >= MIN_SOMA_PX


def downscale(image: np.ndarray, scale: float) -> np.ndarray:
    """Nearest-neighbour subsample. Fine here: this is for looking at, and the
    mask is not being carried along, so there are no instance IDs to corrupt."""
    if scale >= 1.0:
        return image
    h, w = image.shape[:2]
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    ys = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
    xs = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
    return image[np.ix_(ys, xs)]


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Percentile stretch to 8 bit for on-screen viewing.

    Discards absolute intensity, which is irrelevant for counting and would
    otherwise make faint sections effectively invisible on a normal monitor.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1) if img.shape[-1] >= 3 else img[..., 0]
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(img.min()), float(max(img.max(), img.min() + 1))
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def anonymous_ids(n: int, seed: int = SEED) -> list[str]:
    """Shuffled opaque IDs, deterministic for a given n and seed."""
    ids = [f"LC-{i + 1:03d}" for i in range(n)]
    random.Random(seed).shuffle(ids)
    return ids


def build(csv_path: Path, nm_root: Path, out_dir: Path,
          limit: int | None = None, progress_every: int = 20) -> dict:
    """Export one manifest as a blind counting set. Returns a report dict."""
    from skimage.io import imsave

    from .manifest import load_pair

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit is not None and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(k * step)] for k in range(limit)]

    ids = anonymous_ids(len(rows))
    key: list[dict] = []
    failures: list[dict] = []

    for i, row in enumerate(rows):
        try:
            image, mask = load_pair(nm_root, row)
            n_cells = int(len(np.unique(mask[mask > 0])))
            if n_cells == 0:
                raise ValueError("no labelled cells in the reference mask")

            d = soma_diameter(mask)
            scale, ok = choose_scale(mask)
            out = to_uint8(downscale(np.asarray(image), scale))
            imsave(str(images_dir / f"{ids[i]}.png"), out, check_contrast=False)

            key.append({
                "scoring_id": ids[i],
                "reference_count": n_cells,
                "mouse": row.get("mouse", "?"),
                "channel": row.get("channel", "?"),
                "source": row.get("source", "?"),
                "image_name": row.get("image_name", "?"),
                "scale": round(scale, 4),
                "soma_px_native": round(d, 1),
                "soma_px_exported": round(d * scale, 1),
                "width": int(out.shape[1]),
                "height": int(out.shape[0]),
                "countable": ok,
            })
        except Exception as e:
            failures.append({"row": i, "image_name": row.get("image_name", "?"),
                             "mouse": row.get("mouse", "?"),
                             "cause": type(e).__name__, "detail": str(e)[:200]})
        finally:
            gc.collect()

        if (i + 1) % progress_every == 0:
            print(f"    {i + 1}/{len(rows)}  exported={len(key)} "
                  f"failed={len(failures)}")

    key.sort(key=lambda r: r["scoring_id"])

    # The sheet scorers receive: IDs only, in ID order, nothing else. Any extra
    # column is a cue -- even image size hints at which animal it came from.
    sheet = out_dir / "scoring_sheet.csv"
    with open(sheet, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scoring_id", "count", "confidence_1_to_5", "notes"])
        for r in key:
            w.writerow([r["scoring_id"], "", "", ""])

    key_path = out_dir / "key.csv"
    with open(key_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]) if key else ["scoring_id"])
        w.writeheader()
        w.writerows(key)

    counts = [r["reference_count"] for r in key]
    hard = [r["scoring_id"] for r in key if not r["countable"]]
    report = {
        "n_exported": len(key),
        "n_failed": len(failures),
        "n_animals": len({r["mouse"] for r in key}),
        "reference_total_cells": int(sum(counts)),
        "counts_min": int(min(counts)) if counts else 0,
        "counts_max": int(max(counts)) if counts else 0,
        "counts_median": int(np.median(counts)) if counts else 0,
        "n_flagged_hard_to_count": len(hard),
        "flagged": hard,
        "failures": failures,
        "sheet": str(sheet),
        "key": str(key_path),
    }
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")

    readme = out_dir / "README_FOR_SCORERS.txt"
    readme.write_text(
        "COUNTING THE LOCUS COERULEUS — instructions\n"
        "==========================================\n\n"
        "In `images/` there is one PNG per section, named LC-001 ... .\n"
        "Open `scoring_sheet.csv`, put YOUR NAME in the filename\n"
        "(e.g. scoring_sheet_anna.csv), and fill in one number per row:\n"
        "how many labelled neurons you can count in that image.\n\n"
        "  count                 whole number of neurons you see\n"
        "  confidence_1_to_5     5 = sure, 1 = mostly guessing\n"
        "  notes                 anything odd (damaged section, unclear edge)\n\n"
        "Please:\n"
        "  - work alone, and do not compare with the others until all are done\n"
        "  - count every section, even the difficult ones; a confidence of 1\n"
        "    is far more useful to us than a skipped row\n"
        "  - do not adjust brightness between images if you can avoid it\n"
        "  - Fiji's Plugins > Analyze > Cell Counter makes this much easier\n"
        "    once a section has more than ~30 cells\n\n"
        "The images are in a scrambled order and tell you nothing about which\n"
        "animal they came from. That is deliberate — it keeps your count\n"
        "independent, which is the entire point of the exercise.\n\n"
        "There is no right answer we are checking you against. We are measuring\n"
        "how much counts differ between people, so your honest count is the\n"
        "useful one.\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="default: <drive_root>/scoring_set/<split>")
    args = ap.parse_args(argv)

    from .config import load_config

    cfg = load_config(args.params)
    out_dir = Path(args.out) if args.out else \
        cfg.drive_root / "scoring_set" / args.split

    print(f"[scoring_set] {cfg.manifest_for(args.split)} -> {out_dir}")
    r = build(cfg.manifest_for(args.split), cfg.nmslices_root, out_dir,
              limit=args.limit)

    print(f"[scoring_set] {r['n_exported']} sections from {r['n_animals']} animals, "
          f"{r['reference_total_cells']} reference cells "
          f"(median {r['counts_median']}/section, range "
          f"{r['counts_min']}-{r['counts_max']})")
    if r["n_failed"]:
        print(f"[scoring_set] {r['n_failed']} rows could not be exported")
    if r["n_flagged_hard_to_count"]:
        print(f"[scoring_set] {r['n_flagged_hard_to_count']} section(s) could not "
              f"reach {MIN_SOMA_PX}px somata — flagged 'countable=False' in key.csv")
    print(f"[scoring_set] give scorers: {out_dir / 'images'}, "
          f"{Path(r['sheet']).name}, README_FOR_SCORERS.txt")
    print(f"[scoring_set] KEEP BACK: {Path(r['key']).name} (has the reference counts)")
    return 0


# --------------------------------------------------------------------------- #
# agreement, once the sheets come back
# --------------------------------------------------------------------------- #
def agreement(sheets: dict[str, dict[str, int]], reference: dict[str, int] | None = None
              ) -> dict:
    """Inter-rater agreement on counts.

    sheets    {rater_name: {scoring_id: count}}
    reference {scoring_id: count} — the existing masks, treated as one more
              rater rather than as truth, because that is what it is.

    Reports ICC(2,1) (absolute agreement, two-way random), mean pairwise
    absolute and percentage difference, and the per-section coefficient of
    variation. Correlation is deliberately not the headline: two raters who
    disagree by a constant factor correlate at r=1.0 while disagreeing about
    every single number.
    """
    raters = dict(sheets)
    if reference:
        raters["reference"] = reference

    ids = sorted(set.intersection(*(set(v) for v in raters.values())))
    if not ids:
        raise ValueError("no sections scored by every rater")
    names = sorted(raters)
    m = np.array([[float(raters[r][i]) for r in names] for i in ids])  # sections x raters

    n, k = m.shape
    if n < 2:
        # ICC needs between-section variance to partition. One section in common
        # means the sheets barely overlap, which is a data problem to report,
        # not a statistic to compute.
        return {"n_sections": n, "raters": names, "icc_2_1": None,
                "mean_cv_pct": round(float(100 * (m.std(axis=1)
                                                  / np.maximum(m.mean(axis=1), 1)).mean()), 1),
                "worst_sections": ids,
                "pairwise": {},
                "warning": "fewer than 2 sections scored by every rater"}

    grand = m.mean()
    ms_rows = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    icc = float((ms_rows - ms_err) / denom) if denom else float("nan")

    pairs = {}
    for a in range(k):
        for b in range(a + 1, k):
            d = m[:, a] - m[:, b]
            mean_ab = (m[:, a] + m[:, b]) / 2
            pct = 100 * np.abs(d) / np.maximum(mean_ab, 1)
            pairs[f"{names[a]} vs {names[b]}"] = {
                "bias": round(float(d.mean()), 2),
                "limits_of_agreement": [round(float(d.mean() - 1.96 * d.std()), 1),
                                        round(float(d.mean() + 1.96 * d.std()), 1)],
                "mean_abs_diff": round(float(np.abs(d).mean()), 2),
                "mean_pct_diff": round(float(pct.mean()), 1),
            }

    cv = m.std(axis=1) / np.maximum(m.mean(axis=1), 1)
    return {
        "n_sections": n,
        "raters": names,
        "icc_2_1": round(icc, 4),
        "mean_cv_pct": round(float(100 * cv.mean()), 1),
        "worst_sections": [ids[i] for i in np.argsort(-cv)[:10]],
        "pairwise": pairs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
