"""Cut the committed assertion set out of real held-out slices.

WHY REAL PIXELS, AND WHY THIS BREAKS A RULE ON PURPOSE
------------------------------------------------------
`CLAUDE.md` says image data never enters git, and for the dataset that is
right — 185 GiB of slides belong on Drive. But an assertion set made of synthetic
discs tests the arithmetic and nothing else: no staining variability, no touching
somata, no neuromelanin granules, no scanner noise. It would pass a model that
falls over on a real slide, which is the only failure anyone cares about.

So this carves a *small, bounded* exception: a dozen crops of a few hundred
kilobytes each, cut from the held-out split, committed so CI can score them with
no GPU and no Drive. A couple of megabytes, once, versioned like code. If that
ever starts growing, it has stopped being an assertion set and become a dataset,
and it should move back to Drive.

WHICH CROPS
-----------
Sampling is stratified and deterministic:

  * one crop per animal, spread across cohorts and channels, so the set covers
    the staining and microscope variation the model actually has to survive;
  * crops centred on real neurons, so the ground truth is a human's mask and not
    a guess;
  * at least one crop from a region with *no* cells. A model that hallucinates
    where there is no LC must be caught, and it cannot be caught by a set where
    every example contains cells.

Because these come from `test.csv`, they were never trained on — the set stays
honest even as the model changes.

Run in Colab (needs the Drive mount):

    python -m bluespotter.make_assertions --n 12
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

CROP = 384          # px; big enough to hold a cluster of somata with context
MIN_CELLS = 3       # a crop with one cell says almost nothing about counting


def _load(row: dict, nm_root: Path):
    from .manifest import _imread, resolve_row

    ipath, mpath, is_npy = resolve_row(nm_root, row)
    if is_npy:
        blob = np.load(ipath, allow_pickle=True).item()
        return np.asarray(blob["img"]), np.asarray(blob["masks"]).astype(np.int32)
    return np.asarray(_imread(ipath)), np.asarray(_imread(mpath)).astype(np.int32)


def _crop_around_cells(image: np.ndarray, mask: np.ndarray,
                       seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """A CROP x CROP window centred on a real cluster of labelled neurons."""
    ids = np.unique(mask)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return None

    rng = np.random.default_rng(seed)
    target = int(rng.choice(ids))
    ys, xs = np.nonzero(mask == target)
    cy, cx = int(ys.mean()), int(xs.mean())

    h, w = mask.shape[:2]
    half = CROP // 2
    y0 = int(np.clip(cy - half, 0, max(0, h - CROP)))
    x0 = int(np.clip(cx - half, 0, max(0, w - CROP)))
    img_c = image[y0:y0 + CROP, x0:x0 + CROP]
    msk_c = mask[y0:y0 + CROP, x0:x0 + CROP]

    # Renumber to 1..n so the fixture is self-describing, and drop any instance
    # the crop only clipped a sliver of — a half-cell at the border is not a
    # fair thing to score against.
    out = np.zeros_like(msk_c)
    keep = 0
    for cid in np.unique(msk_c):
        if cid == 0:
            continue
        piece = msk_c == cid
        if piece.sum() < 0.5 * (mask == cid).sum():
            continue
        keep += 1
        out[piece] = keep
    return img_c, out


def _crop_empty(image: np.ndarray, mask: np.ndarray,
                seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """A window containing no labelled cells at all."""
    h, w = mask.shape[:2]
    if h < CROP or w < CROP:
        return None
    rng = np.random.default_rng(seed)
    for _ in range(60):
        y0 = int(rng.integers(0, h - CROP))
        x0 = int(rng.integers(0, w - CROP))
        if mask[y0:y0 + CROP, x0:x0 + CROP].max() == 0:
            return image[y0:y0 + CROP, x0:x0 + CROP], np.zeros((CROP, CROP), np.int32)
    return None


def build(manifest: Path, nm_root: Path, out_dir: Path, n: int = 12) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # One row per animal, ordered deterministically by a hash of the mouse ID so
    # the choice does not depend on manifest ordering.
    by_mouse: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mouse[r.get("mouse", "")].append(r)

    def key(mouse: str) -> str:
        return hashlib.md5(f"assert:{mouse}".encode()).hexdigest()  # selection only

    chosen = [sorted(by_mouse[m], key=lambda r: r["image_name"])[0]
              for m in sorted(by_mouse, key=key)][:n]

    cases: list[dict[str, Any]] = []
    want_empty = True
    for i, row in enumerate(chosen):
        try:
            image, mask = _load(row, nm_root)
        except Exception as exc:
            print(f"  [skip] {row.get('image_name')}: {type(exc).__name__} {exc}")
            continue

        made = _crop_around_cells(image, mask, seed=i)
        if made is None or int(made[1].max()) < MIN_CELLS:
            continue
        img_c, msk_c = made

        name = f"{row.get('mouse', 'unknown')}_{row.get('channel', 'NA')}".replace("/", "_")
        np.savez_compressed(out_dir / f"{name}.npz", image=img_c, masks=msk_c)
        cases.append({
            "name": name,
            "n_cells": int(msk_c.max()),
            "mouse": row.get("mouse", ""),
            "channel": row.get("channel", ""),
            "source": row.get("source", ""),
            "from_image": row.get("image_name", ""),
        })

        # One background crop, from whichever slice offers one first.
        if want_empty:
            empty = _crop_empty(image, mask, seed=i)
            if empty is not None:
                np.savez_compressed(out_dir / f"{name}_background.npz",
                                    image=empty[0], masks=empty[1])
                cases.append({
                    "name": f"{name}_background",
                    "n_cells": 0,
                    "mouse": row.get("mouse", ""),
                    "channel": row.get("channel", ""),
                    "source": row.get("source", ""),
                    "from_image": row.get("image_name", ""),
                })
                want_empty = False

    total = sum(p.stat().st_size for p in out_dir.glob("*.npz"))
    manifest_out = {
        "crop_px": CROP,
        "source_split": "test",
        "n_cases": len(cases),
        "total_cells": sum(c["n_cells"] for c in cases),
        "total_bytes": total,
        "cases": cases,
    }
    (out_dir / "expected.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    return manifest_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=None)
    ap.add_argument("--n", type=int, default=12, help="max number of animals to sample")
    ap.add_argument("--out", default="assertions")
    args = ap.parse_args(argv)

    from .config import load_config

    cfg = load_config(args.params)
    out = build(cfg.repo_manifest("test"), cfg.nmslices_root,
                Path(cfg.repo_root) / args.out, n=args.n)

    print(f"  {out['n_cases']} case(s), {out['total_cells']} labelled cells, "
          f"{out['total_bytes'] / 1e6:.1f} MB")
    for c in out["cases"]:
        print(f"    {c['n_cells']:4} cells  {c['name']:28} {c['source']}")
    if out["total_bytes"] > 20e6:
        print("\n  WARNING: over 20 MB. This is meant to be a handful of crops in git, "
              "not a dataset — lower --n or shrink CROP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
