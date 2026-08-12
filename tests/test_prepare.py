"""The on-disk cache that makes lazy training possible.

What these pin is mostly about *failure*: this pass reads every file in the
dataset off a Drive mount that drops under sustained load, so it has to be
resumable and it has to report why rows were lost rather than quietly dropping
them.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from bluespotter.prepare import FLOWS_SUFFIX, cache_name, is_cached, prepare_split


def fake_flows(mask, device=None):
    """Stand-in for dynamics.labels_to_flows.

    cellpose is a heavy import and is not installed in CI, but the contract that
    matters here is ours: a 4-channel float32 array, because _get_batch does
    io.imread(labels_file)[1:] and drops the first channel.
    """
    m = np.asarray(mask, np.float32)
    return np.stack([m, (m > 0).astype(np.float32), m * 0, m * 0])

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
    m = np.zeros((64, 64), dtype=np.int32)
    m[8:24, 8:24] = 1
    m[36:56, 36:56] = 2
    return m


@pytest.fixture
def tree(tmp_path):
    d = tmp_path / "nm" / LOW / "masks" / "npy_masks"
    d.mkdir(parents=True)
    return tmp_path / "nm", d


def _manifest(tmp_path, rows):
    path = tmp_path / "train.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_cache_names_cannot_collide_across_cohorts():
    # Two labs both have a slice1_seg.npy. Without the row index they would
    # share one cached pair and half the dataset would silently be the wrong
    # image.
    a = cache_name({"image_name": "slice1_seg.npy"}, 3)
    b = cache_name({"image_name": "slice1_seg.npy"}, 91)
    assert a != b


def test_a_prepared_row_yields_an_image_and_a_flows_file(tree, tmp_path):
    nm_root, d = tree
    np.save(d / "a_seg.npy", {"img": np.zeros((64, 64), np.uint8), "masks": _masks()},
            allow_pickle=True)
    mf = _manifest(tmp_path, [_row("a_seg.npy")])

    r = prepare_split(mf, nm_root, tmp_path / "cache", flows_fn=fake_flows)
    assert r["n_usable"] == 1 and r["n_failed"] == 0
    img, flows = r["image_files"][0], r["label_files"][0]
    assert flows.endswith(FLOWS_SUFFIX)

    # Cellpose does io.imread(labels_file)[1:], so the file must hold the
    # 4-channel labels_to_flows output, not a raw mask. Handing it a mask
    # trains on garbage without erroring.
    # skimage.io.imsave silently moves a leading axis of 4 to the end, so a
    # (4, H, W) flows array round-trips as (H, W, 4) and Cellpose slices the
    # wrong axis. This caught that; do not relax it.
    import tifffile
    assert tifffile.imread(flows).shape == (4, 64, 64)
    assert tifffile.imread(img).shape == (64, 64)


def test_a_second_run_skips_what_is_already_cached(tree, tmp_path):
    # The pass takes hours over Drive and the mount drops. Re-running must
    # continue, not start again.
    nm_root, d = tree
    np.save(d / "b_seg.npy", {"img": np.zeros((64, 64), np.uint8), "masks": _masks()},
            allow_pickle=True)
    mf = _manifest(tmp_path, [_row("b_seg.npy")])
    cache = tmp_path / "cache"

    first = prepare_split(mf, nm_root, cache, flows_fn=fake_flows)
    second = prepare_split(mf, nm_root, cache, flows_fn=fake_flows)
    assert first["n_already_cached"] == 0
    assert second["n_already_cached"] == 1
    assert second["n_usable"] == 1


def test_a_half_written_entry_is_not_treated_as_cached(tree, tmp_path):
    # Writes are atomic via a .part rename precisely so a crash cannot leave a
    # truncated file that the skip logic then trusts forever.
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "00000_x.tif").write_bytes(b"")
    (cache / f"00000_x{FLOWS_SUFFIX}").write_bytes(b"")
    assert not is_cached(cache, "00000_x")


def test_failures_are_reported_with_a_cause_not_silently_dropped(tree, tmp_path):
    nm_root, d = tree
    np.save(d / "good_seg.npy", {"img": np.zeros((64, 64), np.uint8), "masks": _masks()},
            allow_pickle=True)
    np.save(d / "empty_seg.npy",
            {"img": np.zeros((64, 64), np.uint8), "masks": np.zeros((64, 64), np.int32)},
            allow_pickle=True)
    mf = _manifest(tmp_path, [_row("good_seg.npy"), _row("empty_seg.npy"),
                              _row("missing_seg.npy")])

    r = prepare_split(mf, nm_root, tmp_path / "cache", flows_fn=fake_flows)
    assert r["n_usable"] == 1
    assert r["n_failed"] == 2
    # Grouped by cause, so ~90 failures across the corpus become a few
    # actionable categories rather than a wall of tracebacks.
    assert set(r["causes"]) == {"ValueError", "FileNotFoundError"}
    assert all(f["mouse"] == "DHC-0464" for f in r["failures"])
