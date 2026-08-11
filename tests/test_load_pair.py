"""Loading a manifest row into pixels.

The regression these pin: newer Cellpose `_seg.npy` files do not embed the raw
image. The old loader read `d["img"]`, caught the KeyError and printed a skip,
so ~680 of 1,386 training rows silently vanished while every report still said
1,386. The DVC index counted the files as present — they are — it just never
checked they were readable.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from bluespotter.manifest import find_image_for_seg, load_manifest, load_pair

LOW = "NM_lowtiter_histology_brains"


def _row(image_name: str, **kw) -> dict:
    base = {
        "split": "train", "source": LOW, "microscope": "Olympus",
        "mouse": "DHC-0464", "slice": "", "channel": "TH", "side": "L",
        "image_name": image_name, "image_id": "x",
        "mask_name": image_name, "mask_id": "x",
        "mask_type": "seg_npy (image+mask in one file)",
    }
    base.update(kw)
    return base


def _masks():
    m = np.zeros((40, 40), dtype=np.int32)
    m[5:15, 5:15] = 1
    m[20:30, 20:30] = 2
    return m


@pytest.fixture
def nm_root(tmp_path):
    (tmp_path / LOW / "masks" / "npy_masks").mkdir(parents=True)
    return tmp_path


def test_old_cellpose_files_still_load(nm_root):
    # The generation that embedded the image. Must keep working.
    d = nm_root / LOW / "masks" / "npy_masks"
    image = np.arange(1600, dtype=np.uint8).reshape(40, 40)
    np.save(d / "old_seg.npy", {"img": image, "masks": _masks()}, allow_pickle=True)

    img, msk = load_pair(nm_root, _row("old_seg.npy"))
    np.testing.assert_array_equal(img, image)
    assert set(np.unique(msk)) == {0, 1, 2}


def test_new_cellpose_files_find_the_image_beside_them(nm_root):
    # Cellpose >= 3 drops `img` and records only `filename`. The image sits next
    # to the .npy under the same stem.
    from skimage.io import imsave

    d = nm_root / LOW / "masks" / "npy_masks"
    image = (np.arange(1600) % 255).astype(np.uint8).reshape(40, 40)
    imsave(str(d / "slice1.tif"), image, check_contrast=False)
    np.save(d / "slice1_seg.npy",
            {"masks": _masks(), "filename": "/home/someone/else/slice1.tif",
             "outlines": np.zeros((40, 40))},
            allow_pickle=True)

    img, msk = load_pair(nm_root, _row("slice1_seg.npy"))
    assert img.shape == (40, 40)
    assert set(np.unique(msk)) == {0, 1, 2}


def test_image_is_found_elsewhere_in_the_cohort(nm_root):
    # Cohorts split masks and images across sibling folders, so a name match
    # anywhere under the cohort is the last resort before giving up.
    from skimage.io import imsave

    processed = nm_root / LOW / "processed" / "cropped" / "TH"
    processed.mkdir(parents=True)
    imsave(str(processed / "far.tif"), np.zeros((40, 40), np.uint8), check_contrast=False)

    d = nm_root / LOW / "masks" / "npy_masks"
    np.save(d / "far_seg.npy",
            {"masks": _masks(), "filename": "C:/annotator/far.tif"}, allow_pickle=True)

    img, _msk = load_pair(nm_root, _row("far_seg.npy"))
    assert img.shape == (40, 40)


def test_a_missing_image_raises_with_a_useful_message(nm_root):
    # It must say *why*, and quote the recorded filename, or the next person
    # gets a bare KeyError and no idea where to look.
    d = nm_root / LOW / "masks" / "npy_masks"
    np.save(d / "orphan_seg.npy",
            {"masks": _masks(), "filename": "/gone/orphan.tif"}, allow_pickle=True)

    with pytest.raises(FileNotFoundError, match="Cellpose >=3 drops it"):
        load_pair(nm_root, _row("orphan_seg.npy"))


def test_a_file_without_masks_is_rejected(nm_root):
    d = nm_root / LOW / "masks" / "npy_masks"
    np.save(d / "nomask_seg.npy", {"flows": [], "filename": "x.tif"}, allow_pickle=True)

    with pytest.raises(KeyError, match="no 'masks' key"):
        load_pair(nm_root, _row("nomask_seg.npy"))


def test_find_image_prefers_the_sibling_over_the_recorded_name(nm_root):
    # The recorded filename can point at a file that also exists elsewhere under
    # a different slice. The one sitting next to the .npy is the right one.
    from skimage.io import imsave

    d = nm_root / LOW / "masks" / "npy_masks"
    imsave(str(d / "twin.tif"), np.zeros((10, 10), np.uint8), check_contrast=False)
    npy = d / "twin_seg.npy"
    np.save(npy, {"masks": _masks()}, allow_pickle=True)

    found = find_image_for_seg(npy, {"filename": "/elsewhere/twin.tif"}, nm_root)
    assert found == d / "twin.tif"


def test_load_manifest_refuses_a_silently_halved_dataset(nm_root, tmp_path):
    # The core regression. Ten rows, most unreadable: the old code returned the
    # survivors and printed skips. Training on a fraction of the manifest while
    # reporting the manifest's dataset_hash makes every metric a lie.
    from skimage.io import imsave

    d = nm_root / LOW / "masks" / "npy_masks"
    rows = []
    for i in range(10):
        name = f"s{i}_seg.npy"
        if i < 2:
            imsave(str(d / f"s{i}.tif"), np.zeros((40, 40), np.uint8), check_contrast=False)
        np.save(d / name, {"masks": _masks(), "filename": f"/gone/s{i}.tif"},
                allow_pickle=True)
        rows.append(_row(name))

    manifest = tmp_path / "train.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with pytest.raises(RuntimeError, match="Refusing to train"):
        load_manifest(manifest, nm_root)


def test_load_manifest_tolerates_a_couple_of_bad_rows(nm_root, tmp_path):
    # One corrupt slice out of many should not stop a run — the bar is a
    # proportion, not perfection.
    from skimage.io import imsave

    d = nm_root / LOW / "masks" / "npy_masks"
    rows = []
    for i in range(40):
        name = f"g{i}_seg.npy"
        if i != 0:
            imsave(str(d / f"g{i}.tif"), np.zeros((40, 40), np.uint8), check_contrast=False)
        np.save(d / name, {"masks": _masks(), "filename": f"/gone/g{i}.tif"},
                allow_pickle=True)
        rows.append(_row(name))

    manifest = tmp_path / "train.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    images, labels = load_manifest(manifest, nm_root)
    assert len(images) == len(labels) == 39
