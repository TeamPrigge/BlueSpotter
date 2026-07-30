"""Load a BlueSpotter dataset from a manifest CSV (links, not copies).

The referenced image/mask files already live in the mounted Google Drive
(TeamPrigge/Data/NM_Slices/...), so we read them straight off the mount by path
- no download, no auth, no storage duplication. If a file can't be found on the
mount we fall back to downloading it by Drive file-ID via the API.

Two mask conventions are supported:
  * cp_masks_png : image is a .tif, mask is a Cellpose `_cp_masks.png`
  * seg_npy      : image AND mask live inside one Cellpose `_seg.npy`
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Source cohort -> folder layout under the NM_Slices root.
_ED_COHORT = "NM_hightiter_histology_brains_ED"
_LOW_COHORT = "NM_lowtiter_histology_brains"


def resolve_row(nm_root: Path, row: dict):
    """Return (image_path, mask_path, is_npy) on the mounted Drive for a row.

    This is the single source of truth for manifest-row -> Drive-path mapping.
    Both the training loader and the DVC indexing stage (`drive_index.py`) call
    it, so a layout change only ever has to be made here.
    """
    ch = row.get("channel", "TH")
    iname, mname = row["image_name"], row["mask_name"]
    if row["mask_type"].startswith("seg_npy"):
        p = nm_root / _LOW_COHORT / "masks" / "npy_masks" / iname
        return p, p, True
    # cp_masks_png (ED cohort): image in processed/cropped/<ch>, mask in masks/<ch>
    img = nm_root / _ED_COHORT / "processed" / "cropped" / ch / iname
    msk = nm_root / _ED_COHORT / "masks" / ch / mname
    return img, msk, False


# Backwards-compatible private alias (older call sites).
_resolve = resolve_row


def _imread(path: Path):
    from skimage.io import imread
    return imread(str(path))


def load_manifest(csv_path, nm_root, cache_dir=None, service=None):
    """Return (images, labels) lists ready for cellpose.train.train_seg.

    csv_path : mounted-Drive path to train.csv / test.csv
    nm_root  : mounted-Drive path to .../Data/NM_Slices
    """
    csv_path, nm_root = Path(csv_path), Path(nm_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  Manifest: {csv_path.name}  ({len(rows)} rows)  reading from {nm_root}")

    images, labels, missing = [], [], 0
    for i, r in enumerate(rows, 1):
        ipath, mpath, is_npy = _resolve(nm_root, r)
        try:
            if is_npy:
                if not ipath.exists():
                    raise FileNotFoundError(ipath)
                d = np.load(ipath, allow_pickle=True).item()
                img, msk = d["img"], d["masks"]
            else:
                if not (ipath.exists() and mpath.exists()):
                    raise FileNotFoundError(ipath if not ipath.exists() else mpath)
                img, msk = _imread(ipath), _imread(mpath)
            images.append(np.asarray(img))
            labels.append(np.asarray(msk).astype(np.int32))
        except Exception as e:
            missing += 1
            print(f"    [skip {i}] {r.get('image_name','?')}: {type(e).__name__} {e}")
        if i % 20 == 0:
            print(f"    ...processed {i}/{len(rows)}")

    print(f"  Loaded {len(images)} pairs from {csv_path.name}"
          + (f"  ({missing} skipped)" if missing else ""))
    if not images:
        raise RuntimeError(
            f"No pairs loaded from {csv_path.name}. Check data.nmslices_root in params.yaml "
            f"(currently {nm_root}) points at the mounted NM_Slices folder.")
    return images, labels
