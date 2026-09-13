"""Tests for manifest validation, especially the split-contamination checks."""
from __future__ import annotations

from bluespotter.validate import validate

from .conftest import _npy_row, _png_row


def test_clean_split_passes(write_manifest):
    train = write_manifest("train.csv", [
        _png_row("train", "DHC-0001", "slice1", "L", "a1"),
        _png_row("train", "DHC-0001", "slice2", "R", "a2"),
        _npy_row("train", "DHC-0002", "L", "a3"),
    ])
    test = write_manifest("test.csv", [
        _png_row("test", "DHC-0009", "slice1", "L", "b1"),
    ])
    r = validate(train, test)
    assert r["ok"] is True
    assert r["n_errors"] == 0
    assert r["n_mouse_overlap"] == 0
    assert r["n_slice_overlap"] == 0
    assert r["_detail"]["warnings"] == []
    assert r["n_train"] == 3 and r["n_test"] == 1
    assert r["n_train_mice"] == 2


def test_same_image_in_both_splits_is_an_error(write_manifest):
    row = _png_row("train", "DHC-0001", "slice1", "L", "dup")
    train = write_manifest("train.csv", [row])
    test = write_manifest("test.csv", [row.replace("train,", "test,", 1)])
    r = validate(train, test)
    assert r["ok"] is False
    assert r["image_id_overlap"] == 1
    assert any("BOTH train and test" in e for e in r["_detail"]["errors"])


def test_shared_mouse_across_splits_warns(write_manifest):
    """Different sections of one animal in train and test — optimistic scores."""
    train = write_manifest("train.csv", [
        _png_row("train", "DHC-0929", "slice1", "L", "t1"),
        _png_row("train", "DHC-0929", "slice2", "L", "t2"),
    ])
    test = write_manifest("test.csv", [
        _png_row("test", "DHC-0929", "slice3", "R", "v1"),
    ])
    r = validate(train, test)
    assert r["ok"] is True                    # not fatal by default
    assert r["n_mouse_overlap"] == 1
    assert r["n_slice_overlap"] == 0          # different section
    assert "DHC-0929" in r["_detail"]["leakage"]["mouse_overlap"]
    assert any("also appear in train" in w for w in r["_detail"]["warnings"])


def test_hemispheres_of_one_slice_split_across_train_and_test(write_manifest):
    """The strongest leak: left/right of the same physical section."""
    train = write_manifest("train.csv", [
        _png_row("train", "DHC-0935", "slice4", "L", "L4"),
    ])
    test = write_manifest("test.csv", [
        _png_row("test", "DHC-0935", "slice4", "R", "R4"),
    ])
    r = validate(train, test)
    assert r["n_slice_overlap"] == 1
    assert r["_detail"]["leakage"]["slice_overlap"] == ["DHC-0935|slice4"]


def test_duplicate_row_within_split_is_an_error(write_manifest):
    row = _png_row("train", "DHC-0001", "slice1", "L", "same")
    train = write_manifest("train.csv", [row, row])
    test = write_manifest("test.csv", [_png_row("test", "DHC-0002", "slice1", "L", "o")])
    r = validate(train, test)
    assert r["ok"] is False
    assert any("more than once" in e for e in r["_detail"]["errors"])


def test_unknown_mask_type_is_an_error(write_manifest):
    bad = _png_row("train", "DHC-0001", "slice1", "L", "x").replace(
        "cp_masks_png", "some_new_format")
    train = write_manifest("train.csv", [bad])
    test = write_manifest("test.csv", [_png_row("test", "DHC-0002", "slice1", "L", "o")])
    r = validate(train, test)
    assert r["ok"] is False
    assert any("unknown mask_type" in e for e in r["_detail"]["errors"])


def test_blank_drive_id_is_an_error(write_manifest):
    bad = _png_row("train", "DHC-0001", "slice1", "L", "x").replace(",img_x,", ",,")
    train = write_manifest("train.csv", [bad])
    test = write_manifest("test.csv", [_png_row("test", "DHC-0002", "slice1", "L", "o")])
    r = validate(train, test)
    assert r["ok"] is False
    assert any("blank image_id" in e for e in r["_detail"]["errors"])


def test_metrics_payload_is_flat(write_manifest):
    """`dvc metrics show` flattens JSON; nested values here would wreck the table."""
    train = write_manifest("train.csv", [_png_row("train", "DHC-0001", "slice1", "L", "a")])
    test = write_manifest("test.csv", [_png_row("test", "DHC-0002", "slice1", "L", "b")])
    r = validate(train, test)
    metrics = {k: v for k, v in r.items() if k != "_detail"}
    assert all(isinstance(v, (int, float, bool, str)) for v in metrics.values()), metrics
