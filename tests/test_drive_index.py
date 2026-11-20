"""Tests for the Drive content index — the layer that makes Drive drift visible."""
from __future__ import annotations

import csv

import pytest

from bluespotter.drive_index import build_index
from bluespotter.manifest import resolve_row

from .conftest import ED, LOW, _npy_row, _png_row


def _hash(manifest, drive, tmp_path, mode="full", name="idx.csv"):
    return build_index(manifest, drive, tmp_path / name, hash_mode=mode,
                       cache_path=tmp_path / "cache.json")


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #
def test_resolver_png_layout(drive):
    row = dict(zip(  # noqa: B905 - fixture rows are known to be aligned
        "split source microscope mouse slice channel side image_name image_id "
        "mask_name mask_id mask_type".split(),
        _png_row("train", "M1", "slice1", "L", "stem").split(","),
    ))
    img, msk, is_npy = resolve_row(drive, row)
    assert is_npy is False
    assert img == drive / ED / "processed" / "cropped" / "TH" / "stem.tif"
    assert msk == drive / ED / "masks" / "TH" / "stem_cp_masks.png"


def test_resolver_npy_points_image_and_mask_at_one_file(drive):
    row = dict(zip(  # noqa: B905 - fixture rows are known to be aligned
        "split source microscope mouse slice channel side image_name image_id "
        "mask_name mask_id mask_type".split(),
        _npy_row("train", "M1", "L", "stem").split(","),
    ))
    img, msk, is_npy = resolve_row(drive, row)
    assert is_npy is True
    assert img == msk == drive / LOW / "masks" / "npy_masks" / "stem.npy"


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #
def test_index_counts_two_files_per_png_row_and_one_per_npy(
        drive, write_manifest, make_png_pair, make_npy, tmp_path):
    make_png_pair("p1")
    make_npy("n1")
    mf = write_manifest("train.csv", [
        _png_row("train", "M1", "slice1", "L", "p1"),
        _npy_row("train", "M2", "L", "n1"),
    ])
    s = _hash(mf, drive, tmp_path)
    assert s["rows"] == 2
    assert s["files_indexed"] == 3      # 2 for the png row, 1 for the npy row
    assert s["files_present"] == 3
    assert s["files_missing"] == 0


def test_missing_file_is_reported_not_raised(drive, write_manifest, tmp_path):
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "ghost")])
    s = _hash(mf, drive, tmp_path)
    assert s["files_missing"] == 2
    assert s["files_present"] == 0
    assert any("ghost" in p for p in s["missing_examples"])


def test_hash_is_stable_across_runs(drive, write_manifest, make_png_pair, tmp_path):
    make_png_pair("p1")
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "p1")])
    a = _hash(mf, drive, tmp_path, name="a.csv")["dataset_hash"]
    b = _hash(mf, drive, tmp_path, name="b.csv")["dataset_hash"]
    assert a == b


def test_silent_content_change_changes_dataset_hash(
        drive, write_manifest, make_png_pair, tmp_path):
    """Same filename, same length, different bytes — the case that breaks naive setups."""
    make_png_pair("p1", mask_bytes=b"AAA")
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "p1")])
    before = _hash(mf, drive, tmp_path, name="a.csv")["dataset_hash"]

    make_png_pair("p1", mask_bytes=b"BBB")   # identical size
    after = _hash(mf, drive, tmp_path, name="b.csv")["dataset_hash"]
    assert before != after


def test_row_order_in_manifest_does_not_change_dataset_hash(
        drive, write_manifest, make_png_pair, tmp_path):
    """Re-sorting the CSV is not a data change and must not look like one."""
    make_png_pair("p1")
    make_png_pair("p2")
    r1 = _png_row("train", "M1", "slice1", "L", "p1")
    r2 = _png_row("train", "M2", "slice1", "L", "p2")
    a = _hash(write_manifest("a.csv", [r1, r2]), drive, tmp_path, name="ia.csv")
    b = _hash(write_manifest("b.csv", [r2, r1]), drive, tmp_path, name="ib.csv")
    assert a["dataset_hash"] == b["dataset_hash"]


def test_index_csv_has_expected_columns(drive, write_manifest, make_png_pair, tmp_path):
    make_png_pair("p1")
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "p1")])
    out = tmp_path / "idx.csv"
    build_index(mf, drive, out, hash_mode="full", cache_path=tmp_path / "c.json")
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["role"] for r in rows} == {"image", "mask"}
    assert all(r["rel_path"].startswith(ED) for r in rows)
    assert all(int(r["exists"]) == 1 and r["content_hash"] for r in rows)


def _backdate(*paths, seconds=3600):
    """Age files so they fall outside the racily-clean window."""
    import os
    import time
    old = time.time() - seconds
    for p in paths:
        os.utime(p, (old, old))


def test_hash_cache_avoids_rereading_unchanged_files(
        drive, write_manifest, make_png_pair, tmp_path, monkeypatch):
    make_png_pair("p1")
    _backdate(drive / ED / "processed" / "cropped" / "TH" / "p1.tif",
              drive / ED / "masks" / "TH" / "p1_cp_masks.png")
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "p1")])
    cache = tmp_path / "cache.json"
    build_index(mf, drive, tmp_path / "a.csv", hash_mode="full", cache_path=cache)
    assert cache.exists()

    # Second pass must not hash anything: make the hasher explode if called.
    import bluespotter.drive_index as di
    monkeypatch.setattr(di, "_md5_full", lambda p: pytest.fail("re-hashed an unchanged file"))
    build_index(mf, drive, tmp_path / "b.csv", hash_mode="full", cache_path=cache)


def test_cache_does_not_trust_a_file_edited_in_the_same_tick(
        drive, write_manifest, make_png_pair, tmp_path):
    """A same-second, same-size edit must still change the dataset hash.

    Drive-backed mounts often round mtime to whole seconds, so size+mtime can be
    byte-for-byte identical before and after a fast edit. The cache therefore
    refuses to trust entries whose file mtime is not comfortably older than the
    moment the hash was recorded, and re-reads instead. This test drives that
    path with a shared cache and no artificial ageing.

    Residual limitation, stated plainly: nothing keyed on (size, mtime) can catch
    a writer that deliberately restores the original mtime after editing. Drive
    and Cellpose do not do that; if you ever need certainty, delete the cache and
    re-run with hash_mode: full.
    """
    make_png_pair("p1", mask_bytes=b"AAA")
    cache = tmp_path / "cache.json"
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "p1")])
    before = build_index(mf, drive, tmp_path / "a.csv", hash_mode="full",
                         cache_path=cache)["dataset_hash"]

    make_png_pair("p1", mask_bytes=b"BBB")   # identical size, mtime ~ now
    after = build_index(mf, drive, tmp_path / "b.csv", hash_mode="full",
                        cache_path=cache)["dataset_hash"]

    assert before != after, "stale cache hit hid a real content change"


def test_partial_mode_can_miss_a_mid_file_edit_but_full_mode_catches_it(
        drive, write_manifest, big_file_factory, tmp_path):
    """Documents the documented trade-off of hash_mode, so it stays deliberate."""
    img = drive / ED / "processed" / "cropped" / "TH" / "big.tif"
    msk = drive / ED / "masks" / "TH" / "big_cp_masks.png"
    big_file_factory(img, 12 * 1024 * 1024)
    big_file_factory(msk, 1024)
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "big")])

    partial_before = _hash(mf, drive, tmp_path, "partial", "p1.csv")["dataset_hash"]
    full_before = _hash(mf, drive, tmp_path, "full", "f1.csv")["dataset_hash"]

    big_file_factory.mutate_middle(img)
    # mtime changes invalidate the cache, so both modes re-read the file.
    partial_after = _hash(mf, drive, tmp_path, "partial", "p2.csv")["dataset_hash"]
    full_after = _hash(mf, drive, tmp_path, "full", "f2.csv")["dataset_hash"]

    assert full_before != full_after, "full mode must catch any byte change"
    assert partial_before == partial_after, (
        "partial mode is head+tail only — this is the known blind spot documented "
        "in params.yaml; use hash_mode: full when it matters"
    )


def test_rejects_unknown_hash_mode(drive, write_manifest, tmp_path):
    mf = write_manifest("train.csv", [_png_row("train", "M1", "slice1", "L", "p")])
    with pytest.raises(ValueError, match="hash_mode"):
        _hash(mf, drive, tmp_path, mode="sha512")


def test_missing_manifest_raises(drive, tmp_path):
    with pytest.raises(FileNotFoundError):
        _hash(tmp_path / "nope.csv", drive, tmp_path)
