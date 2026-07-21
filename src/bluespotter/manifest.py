"""Load a BlueSpotter dataset from a manifest CSV (links, not copies).

Each manifest row points at an image + its mask by Google Drive file-ID. The
pixels stay in Drive; here we download each referenced file to local disk (once)
and load it into memory for Cellpose. Two mask conventions are supported:

  * cp_masks_png : image is a .tif, mask is a Cellpose `_cp_masks.png` (2 files)
  * seg_npy      : image AND mask live inside one Cellpose `_seg.npy`
                   (d = np.load(...).item(); img=d['img']; masks=d['masks'])

The download uses the Drive API with your Colab identity, so it works for
private and shared files (a one-time auth popup may appear).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


def get_drive_service():
    """Authenticated Drive API client using the Colab user's identity."""
    from google.colab import auth
    auth.authenticate_user()
    from googleapiclient.discovery import build
    return build("drive", "v3")


def _download(service, file_id: str, dest: Path) -> Path:
    """Download one Drive file by ID to dest (skips if already present)."""
    from googleapiclient.http import MediaIoBaseDownload
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = service.files().get_media(fileId=file_id)
    with open(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return dest


def _imread(path: Path):
    """Read a .tif/.png image or mask into a numpy array."""
    from skimage.io import imread
    return imread(str(path))


def load_manifest(csv_path, cache_dir, service=None):
    """Return (images, labels) lists ready for cellpose.train.train_seg.

    csv_path : mounted-Drive path to train.csv / test.csv
    cache_dir: local folder to download the pixel files into
    """
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if service is None:
        service = get_drive_service()
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    images, labels = [], []
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"  Manifest: {csv_path.name}  ({len(rows)} rows)")
    for i, r in enumerate(rows, 1):
        mask_type = r["mask_type"]
        try:
            if mask_type.startswith("seg_npy"):
                npy = _download(service, r["image_id"], cache_dir / f"{r['image_id']}.npy")
                d = np.load(npy, allow_pickle=True).item()
                img, msk = d["img"], d["masks"]
            else:  # cp_masks_png: separate image (.tif) + mask (.png)
                img_p = _download(service, r["image_id"], cache_dir / f"{r['image_id']}.tif")
                msk_p = _download(service, r["mask_id"], cache_dir / f"{r['mask_id']}.png")
                img, msk = _imread(img_p), _imread(msk_p)
            images.append(np.asarray(img))
            labels.append(np.asarray(msk).astype(np.int32))
        except Exception as e:  # noqa: BLE001 - skip a bad row rather than abort
            print(f"    [skip row {i}] {r.get('image_name','?')}: {e}")
        if i % 10 == 0:
            print(f"    ...loaded {i}/{len(rows)}")

    print(f"  Loaded {len(images)} image/mask pairs from {csv_path.name}")
    return images, labels
