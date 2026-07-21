"""Data handling: cache from Drive to local scratch, then load for Cellpose.

The Drive mount is slow and flaky for a multi-hour training loop, so we copy the
working dataset to the Colab VM's local SSD once, and train against that.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from cellpose import io


def cache_from_drive(data_dir: Path, local_cache: Path, force: bool = False) -> Path:
    """Copy the training dataset from Drive to fast local storage.

    Returns the local directory that now holds the paired image/mask files.
    """
    data_dir = Path(data_dir)
    local_cache = Path(local_cache)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Drive data folder not found: {data_dir}\n"
            "Check `drive.root` in params.yaml and that Drive is mounted."
        )

    local_cache.mkdir(parents=True, exist_ok=True)
    if force and local_cache.exists():
        shutil.rmtree(local_cache)
        local_cache.mkdir(parents=True, exist_ok=True)

    n = 0
    for src in sorted(data_dir.iterdir()):
        if src.is_file():
            dst = local_cache / src.name
            if force or not dst.exists():
                shutil.copy2(src, dst)
            n += 1
    if n == 0:
        raise RuntimeError(
            f"No files found in {data_dir}. Upload paired *_img/*_masks files first."
        )
    print(f"Cached {n} files to {local_cache}")
    return local_cache


def load_dataset(
    local_dir: Path,
    image_filter: str = "_img",
    mask_filter: str = "_masks",
    test_dir: Path | None = None,
):
    """Load paired images/labels via Cellpose's loader.

    Returns (images, labels, test_images, test_labels). Test lists may be empty
    if no separate test_dir is provided — caller can split manually.
    """
    out = io.load_train_test_data(
        str(local_dir),
        str(test_dir) if test_dir else None,
        image_filter=image_filter,
        mask_filter=mask_filter,
        look_one_level_down=False,
    )
    images, labels, image_names, test_images, test_labels, test_names = out
    print(f"Loaded {len(images)} training images, {len(test_images or [])} test images")
    return images, labels, test_images, test_labels
