"""Cut the committed assertion set out of real held-out slices.

WHY REAL PIXELS, AND WHY THIS BREAKS A RULE ON PURPOSE
------------------------------------------------------
`CLAUDE.md` says image data never enters git, and for the dataset that is
right — 185 GiB of slides belong on Drive. But an assertion set of synthetic
discs tests the arithmetic and nothing else: no staining variability, no touching
somata, no neuromelanin granules, no scanner noise. It would pass a model that
falls over on a real slide, which is the only failure anyone cares about.

So this carves a *small, bounded* exception, kept small by storing PNGs rather
than raw arrays. `tests/test_assertions.py` fails the build above 20 MB; past
that it has stopped being an assertion set and become a dataset.

TWO KINDS OF CASE, TESTING DIFFERENT THINGS
-------------------------------------------
`detail`    A 384 px window at native resolution, centred on the densest cluster
            of neurons in the slice. Tests boundary accuracy on touching somata —
            the crowded field where Cellpose characteristically fuses two cells
            into one.

`whole_lc`  The bounding box of *every* labelled neuron in the slice, downscaled
            to fit `MAX_PX`. Tests the thing a detail crop cannot: does the model
            find the whole nucleus, keep its shape, and get the total count right
            at a magnification where each soma is only a few pixels across.

Downscaling is honest here because image and mask are scaled together and the
model is run on the same downscaled image, so predictions and ground truth live
in the same space. What it does change is which cells are *resolvable* — and a
model that loses small somata at low magnification is a real finding, not an
artefact.

WHAT IS DELIBERATELY LOST
-------------------------
Images are written as 8-bit PNGs after a percentile contrast stretch. That
discards absolute intensity, so these crops are useless for neuromelanin
quantification and must never be used for it. For asking "did it find the cell
and draw the right outline", which is all this set is for, it costs nothing and
saves an order of magnitude of space. Masks are 16-bit PNGs, which preserves
instance IDs exactly — nearest-neighbour on the way down, so a label is never
interpolated into a value that means a different cell.

Run in Colab (needs the Drive mount):

    python -m bluespotter.make_assertions --n 20
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DETAIL_PX = 384       # native-resolution window
MAX_PX = 640          # preferred longest side of a downscaled whole-LC view
HARD_MAX_PX = 1600    # but never shrink so far that somata stop being resolvable
MIN_SCALE = 0.25      # a 20 px soma must survive as >= 5 px, or the case is
                      # unsegmentable by anything and tests nothing
MIN_CELLS = 5         # below this a crop is too noisy to assert on
PAD = 0.12            # fraction of the LC bounding box added as context


def _load(row: dict, nm_root: Path):
    from .manifest import load_pair

    return load_pair(nm_root, row)


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def to_uint8(image: np.ndarray) -> np.ndarray:
    """Percentile-stretch to 8 bit. Loses absolute intensity, keeps structure."""
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1) if img.shape[-1] >= 3 else img[..., 0]
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(img.min()), float(max(img.max(), img.min() + 1))
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def downscale(image: np.ndarray, mask: np.ndarray,
              max_px: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Shrink so the longest side is <= max_px. Masks use nearest neighbour.

    Anything smoother than nearest on a label image invents instance IDs that
    were never there — averaging cell 3 and cell 7 gives you cell 5.
    """
    h, w = mask.shape[:2]
    longest = max(h, w)
    if longest <= max_px:
        return image, mask, 1.0

    scale = max_px / longest
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    ys = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
    xs = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
    return image[np.ix_(ys, xs)], mask[np.ix_(ys, xs)], scale


def _renumber(mask: np.ndarray, reference: np.ndarray | None = None,
              min_fraction: float = 0.5) -> np.ndarray:
    """Relabel to 1..n, dropping instances the crop only clipped a sliver of.

    A half-cell at the border is not a fair thing to score a model against: it
    can neither be found nor missed cleanly.
    """
    out = np.zeros_like(mask)
    keep = 0
    for cid in np.unique(mask):
        if cid == 0:
            continue
        piece = mask == cid
        if reference is not None and piece.sum() < min_fraction * (reference == cid).sum():
            continue
        keep += 1
        out[piece] = keep
    return out


def _write_case(out_dir: Path, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    from skimage.io import imsave

    imsave(str(out_dir / f"{name}_image.png"), to_uint8(image), check_contrast=False)
    imsave(str(out_dir / f"{name}_masks.png"), mask.astype(np.uint16), check_contrast=False)


# --------------------------------------------------------------------------- #
# crop selection
# --------------------------------------------------------------------------- #
def densest_window(mask: np.ndarray, size: int) -> tuple[int, int] | None:
    """Top-left corner of the `size` window containing the most instances.

    Centring on a randomly chosen cell, as the first version did, lands on the
    sparse edge of the LC about as often as the middle and yields three-cell
    crops. One miss then moves recall by a third, which is too noisy to gate on.
    """
    ids = np.unique(mask)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return None

    centres = []
    for cid in ids:
        ys, xs = np.nonzero(mask == cid)
        centres.append((ys.mean(), xs.mean()))
    centres_arr = np.array(centres)

    h, w = mask.shape[:2]
    best, best_n = None, -1
    for cy, cx in centres_arr:
        y0 = int(np.clip(cy - size / 2, 0, max(0, h - size)))
        x0 = int(np.clip(cx - size / 2, 0, max(0, w - size)))
        inside = ((centres_arr[:, 0] >= y0) & (centres_arr[:, 0] < y0 + size) &
                  (centres_arr[:, 1] >= x0) & (centres_arr[:, 1] < x0 + size)).sum()
        if inside > best_n:
            best, best_n = (y0, x0), inside
    return best


def detail_crop(image: np.ndarray, mask: np.ndarray):
    corner = densest_window(mask, DETAIL_PX)
    if corner is None:
        return None
    y0, x0 = corner
    img_c = image[y0:y0 + DETAIL_PX, x0:x0 + DETAIL_PX]
    msk_c = _renumber(mask[y0:y0 + DETAIL_PX, x0:x0 + DETAIL_PX], reference=mask)
    return img_c, msk_c, 1.0


def whole_lc_crop(image: np.ndarray, mask: np.ndarray):
    """Bounding box of every labelled neuron, padded, then downscaled."""
    ys, xs = np.nonzero(mask > 0)
    if len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    pad_y = int(PAD * (ys.max() - ys.min() + 1))
    pad_x = int(PAD * (xs.max() - xs.min() + 1))
    y0, y1 = max(0, ys.min() - pad_y), min(h, ys.max() + pad_y + 1)
    x0, x1 = max(0, xs.min() - pad_x), min(w, xs.max() + pad_x + 1)

    img_c = image[y0:y1, x0:x1]
    msk_c = mask[y0:y1, x0:x1]

    # Fitting a 10,000 px slice into 640 px is a x0.06 shrink, which turns a
    # 20 px soma into one pixel. The resulting case would fail for every model
    # that will ever exist and tell you nothing about any of them. So the target
    # size is whichever is larger: the preferred width, or the width that keeps
    # the scale at MIN_SCALE — capped so the file stays small enough for git.
    longest = max(msk_c.shape[:2])
    target = min(HARD_MAX_PX, max(MAX_PX, int(longest * MIN_SCALE)))
    img_s, msk_s, scale = downscale(img_c, msk_c, target)
    # Renumber after scaling: a cell that shrank below a pixel is genuinely gone
    # and must not be counted as ground truth the model is expected to find.
    return img_s, _renumber(msk_s), scale


def background_crop(image: np.ndarray, mask: np.ndarray, seed: int):
    h, w = mask.shape[:2]
    if h < DETAIL_PX or w < DETAIL_PX:
        return None
    rng = np.random.default_rng(seed)
    for _ in range(80):
        y0 = int(rng.integers(0, h - DETAIL_PX))
        x0 = int(rng.integers(0, w - DETAIL_PX))
        if mask[y0:y0 + DETAIL_PX, x0:x0 + DETAIL_PX].max() == 0:
            return (image[y0:y0 + DETAIL_PX, x0:x0 + DETAIL_PX],
                    np.zeros((DETAIL_PX, DETAIL_PX), np.int32), 1.0)
    return None


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(manifest: Path, nm_root: Path, out_dir: Path, n: int = 20,
          per_animal: int = 2, n_background: int = 3) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # data/ is gitignored, so a fresh clone or a restarted Colab session has no
    # manifests until sync-manifests has run. Say that, rather than leaving a
    # bare path and a puzzle.
    if not Path(manifest).exists():
        raise FileNotFoundError(
            f"{manifest} not found. The manifests live on Drive and are copied in "
            f"by the sync-manifests stage, which has not run in this session:\n"
            f"    dvc repro sync-manifests")

    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))

    by_mouse: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mouse[r.get("mouse", "")].append(r)

    def key(mouse: str) -> str:
        return hashlib.md5(f"assert:{mouse}".encode()).hexdigest()  # selection only

    cases: list[dict[str, Any]] = []
    backgrounds = 0

    for i, mouse in enumerate(sorted(by_mouse, key=key)):
        if len({c["mouse"] for c in cases}) >= n:
            break
        # Try several slices per animal: the first one may be sparse, and giving
        # up on the animal entirely was how the first pass lost half its cases.
        candidates = sorted(by_mouse[mouse], key=lambda r: r["image_name"])[:per_animal + 2]
        made_for_mouse = 0

        for row in candidates:
            if made_for_mouse >= per_animal:
                break
            try:
                image, mask = _load(row, nm_root)
            except Exception as exc:
                print(f"  [skip] {row.get('image_name')}: {type(exc).__name__} {exc}")
                continue

            base = f"{mouse}_{row.get('channel', 'NA')}".replace("/", "_")
            for kind, maker in (("detail", detail_crop), ("whole_lc", whole_lc_crop)):
                made = maker(image, mask)
                if made is None:
                    continue
                img_c, msk_c, scale = made
                if int(msk_c.max()) < MIN_CELLS:
                    continue
                name = f"{base}_{kind}"
                if any(c["name"] == name for c in cases):
                    continue
                _write_case(out_dir, name, img_c, msk_c)
                cases.append({
                    "name": name, "kind": kind,
                    "n_cells": int(msk_c.max()),
                    "scale": round(float(scale), 4),
                    "height": int(msk_c.shape[0]), "width": int(msk_c.shape[1]),
                    "mouse": mouse, "channel": row.get("channel", ""),
                    "source": row.get("source", ""),
                    "from_image": row.get("image_name", ""),
                })
                made_for_mouse += 1

            if backgrounds < n_background:
                made = background_crop(image, mask, seed=i)
                if made is not None:
                    name = f"{base}_background"
                    if not any(c["name"] == name for c in cases):
                        _write_case(out_dir, name, made[0], made[1])
                        cases.append({
                            "name": name, "kind": "background", "n_cells": 0,
                            "scale": 1.0,
                            "height": int(made[1].shape[0]), "width": int(made[1].shape[1]),
                            "mouse": mouse, "channel": row.get("channel", ""),
                            "source": row.get("source", ""),
                            "from_image": row.get("image_name", ""),
                        })
                        backgrounds += 1

    total = sum(p.stat().st_size for p in out_dir.glob("*.png"))
    manifest_out = {
        "format": "png",
        "detail_px": DETAIL_PX,
        "max_px": MAX_PX,
        "source_split": "test",
        "n_cases": len(cases),
        "n_animals": len({c["mouse"] for c in cases}),
        "total_cells": sum(c["n_cells"] for c in cases),
        "total_bytes": total,
        "by_kind": {k: sum(1 for c in cases if c["kind"] == k)
                    for k in ("detail", "whole_lc", "background")},
        "cases": cases,
    }
    (out_dir / "expected.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    return manifest_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=None)
    ap.add_argument("--n", type=int, default=20, help="max number of animals")
    ap.add_argument("--per-animal", type=int, default=2, help="max crops per animal")
    ap.add_argument("--out", default="assertions")
    args = ap.parse_args(argv)

    from .config import load_config

    cfg = load_config(args.params)
    out = build(cfg.repo_manifest("test"), cfg.nmslices_root,
                Path(cfg.repo_root) / args.out, n=args.n, per_animal=args.per_animal)

    print(f"\n  {out['n_cases']} case(s) from {out['n_animals']} animals, "
          f"{out['total_cells']} labelled cells, {out['total_bytes'] / 1e6:.1f} MB")
    print(f"  by kind: {out['by_kind']}")
    for c in out["cases"]:
        px = f"{c['width']}x{c['height']}"
        sc = "" if c["scale"] == 1.0 else f"  (x{c['scale']:.2f})"
        print(f"    {c['n_cells']:4} cells  {c['kind']:11} {px:>9}{sc:9}  "
              f"{c['name']:34} {c['source']}")
    if out["total_bytes"] > 20e6:
        print("\n  WARNING: over 20 MB. Lower --n or --per-animal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
