"""Build the on-disk training cache that lets Cellpose train without holding
the dataset in RAM.

WHY THIS EXISTS
---------------
`train.py` used to load every slice into a Python list and hand the list to
`train_seg`. At 1,386 slices of ~10,000 px that needs far more memory than any
Colab runtime has, so the full dataset simply could not be trained on.

Cellpose already solves this and we were not using it. `train_seg` accepts
`train_files` / `train_labels_files` with `load_files=False`, and then
`_get_batch` does `io.imread(files[i])` for the current batch only. Peak memory
becomes batch-size images instead of the whole corpus.

Two things it needs that a manifest row does not give it directly:

1. **An image file it can read.** Rows whose mask is a Cellpose `_seg.npy` may
   have the image embedded in the pickle rather than sitting beside it, so there
   is no path to hand over. Those get written out.
2. **A flows file, not a mask file.** This is the subtle one. `_get_batch` does
   `io.imread(labels_files[i])[1:]`, i.e. it expects the 4-channel output of
   `dynamics.labels_to_flows` — [labels, cellprob, flowY, flowX] — and drops the
   first channel. Handing it a raw mask silently trains on garbage. So flows are
   computed here, once, and cached.

Computing flows is also the expensive part of startup (it ran at ~1 image/sec in
the first smoke run), so caching them makes every subsequent run start fast.

NO TILING, AND WHY IT WOULD HAVE BEEN POINTLESS
-----------------------------------------------
An earlier plan was to pre-cut slices into tiles. That was unnecessary: Cellpose
takes a random rotated/rescaled `bsize` crop from each image on every batch
anyway (`random_rotate_and_resize`, bsize=256 for cpsam). Pre-cutting would fix
the crop pattern and lose that augmentation diversity while buying nothing that
lazy loading does not already buy. Full images stay full images.

RESUMABLE ON PURPOSE
--------------------
The Drive mount drops under sustained reads of large files, and this pass reads
every file in the dataset. So each row is written to a temp file and renamed
into place, and an existing complete pair is skipped. Re-running after a crash
continues where it stopped rather than starting the multi-hour job again.

Run in Colab:

    python -m bluespotter.prepare                 # both splits
    python -m bluespotter.prepare --split train --limit 50
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np

FLOWS_SUFFIX = "_flows.tif"


def cache_name(row: dict, i: int) -> str:
    """Stable, filesystem-safe stem for a manifest row.

    Includes the row index because image_name is not unique across cohorts —
    two labs both have a `slice1_seg.npy` — and a collision would silently make
    two rows share one cached pair.
    """
    stem = Path(str(row.get("image_name") or f"row{i}")).stem
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in stem)
    return f"{i:05d}_{safe}"


def is_cached(cache_dir: Path, name: str) -> bool:
    img = cache_dir / f"{name}.tif"
    flows = cache_dir / f"{name}{FLOWS_SUFFIX}"
    return img.exists() and flows.exists() and img.stat().st_size > 0 \
        and flows.stat().st_size > 0


def _atomic_imsave(path: Path, array: np.ndarray) -> None:
    """Write via a temp file so a crash never leaves a half-written cache entry
    that `is_cached` would then happily skip.

    Uses tifffile rather than skimage.io.imsave because skimage guesses at
    photometric interpretation and silently moves a leading axis of 4 to the
    end: a (4, H, W) flows array comes back as (H, W, 4). Cellpose then does
    `io.imread(labels_file)[1:]`, slices the wrong axis, and trains on nonsense
    without raising anything. A test pins the round-trip shape.
    """
    import tifffile

    tmp = path.with_suffix(path.suffix + ".part")
    tifffile.imwrite(str(tmp), np.ascontiguousarray(array))
    tmp.replace(path)


def _to_flows(mask: np.ndarray, device=None) -> np.ndarray:
    """Mask -> the 4-channel array Cellpose expects in a labels file."""
    from cellpose import dynamics

    flows = dynamics.labels_to_flows([mask.astype(np.int32)], device=device)[0]
    return np.asarray(flows, dtype=np.float32)


def prepare_split(csv_path: Path, nm_root: Path, cache_dir: Path,
                  limit: int | None = None, device=None,
                  progress_every: int = 25, flows_fn=None) -> dict:
    """Materialise one split into `cache_dir`. Returns a report dict.

    flows_fn is injectable so the cache logic — resumability, atomic writes,
    failure reporting — can be tested without cellpose installed. Tests must not
    need a GPU or a heavyweight import (CLAUDE.md 12), and CI installs neither.
    """
    from .manifest import load_pair

    flows_fn = flows_fn or _to_flows

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit is not None and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(k * step)] for k in range(limit)]

    image_files: list[str] = []
    label_files: list[str] = []
    failures: list[dict] = []
    n_skipped = 0
    total_bytes = 0

    for i, row in enumerate(rows):
        name = cache_name(row, i)
        img_path = cache_dir / f"{name}.tif"
        flow_path = cache_dir / f"{name}{FLOWS_SUFFIX}"

        if is_cached(cache_dir, name):
            n_skipped += 1
            image_files.append(str(img_path))
            label_files.append(str(flow_path))
            total_bytes += img_path.stat().st_size + flow_path.stat().st_size
            continue

        try:
            image, mask = load_pair(nm_root, row)
            if int(np.max(mask)) == 0:
                raise ValueError("mask contains no labelled cells")
            _atomic_imsave(img_path, np.asarray(image))
            _atomic_imsave(flow_path, flows_fn(mask, device=device))
            total_bytes += img_path.stat().st_size + flow_path.stat().st_size
            image_files.append(str(img_path))
            label_files.append(str(flow_path))
        except Exception as e:
            # Grouped by exception type in the report, so ~90 failures across
            # the corpus become a handful of actionable causes rather than a
            # wall of text.
            failures.append({
                "row": i,
                "image_name": row.get("image_name", "?"),
                "mouse": row.get("mouse", "?"),
                "source": row.get("source", "?"),
                "cause": type(e).__name__,
                "detail": str(e)[:300],
            })
        finally:
            gc.collect()

        if (i + 1) % progress_every == 0:
            print(f"    {i + 1}/{len(rows)}  cached={len(image_files)} "
                  f"skipped={n_skipped} failed={len(failures)} "
                  f"{total_bytes / 2**30:.1f} GiB")

    return {
        "manifest": str(csv_path),
        "n_rows": len(rows),
        "n_usable": len(image_files),
        "n_already_cached": n_skipped,
        "n_failed": len(failures),
        "cache_gib": round(total_bytes / 2**30, 3),
        "causes": dict(Counter(f["cause"] for f in failures)),
        "image_files": image_files,
        "label_files": label_files,
        "failures": failures,
    }


def cache_dir_for(cfg, split: str) -> Path:
    root = Path(cfg.train.get("cache_dir") or cfg.data["local_cache"])
    return root / "prepared" / split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=None)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from .config import load_config

    cfg = load_config(args.params)
    device = None
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
    except ImportError:
        pass

    splits = ["train", "test"] if args.split == "both" else [args.split]
    report = {}
    for split in splits:
        out = cache_dir_for(cfg, split)
        print(f"[prepare] {split}: {cfg.manifest_for(split)} -> {out}")
        r = prepare_split(cfg.manifest_for(split), cfg.nmslices_root, out,
                          limit=args.limit, device=device)
        print(f"[prepare] {split}: {r['n_usable']}/{r['n_rows']} usable "
              f"({r['n_already_cached']} already cached, {r['n_failed']} failed), "
              f"{r['cache_gib']} GiB")
        for cause, n in sorted(r["causes"].items(), key=lambda kv: -kv[1]):
            print(f"             {n:4d}  {cause}")
        # File lists are long and only useful to the trainer; keep them out of
        # the committed report.
        report[split] = {k: v for k, v in r.items()
                         if k not in ("image_files", "label_files")}

    reports = cfg.repo_root / cfg.dvc.get("report_dir", "reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "prepare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[prepare] report -> {reports / 'prepare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
