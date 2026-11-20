"""The AP manifest exists to carry a trustworthy bregma coordinate. These tests
pin the failure modes that would make it untrustworthy without being obvious."""
from __future__ import annotations

import csv
from pathlib import Path

from bluespotter.discover import AP_FIELDS, MANIFEST_FIELDS
from bluespotter.drive_links import enrich
from bluespotter.validate import validate_ap


def _write(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _seg(name, mouse="DHC-2076", split="train"):
    rel = f"NM_tyr_titering/masks/{name}"
    return {"split": split, "source": "NM_tyr_titering", "mouse": mouse,
            "channel": "TH", "side": "R", "image_name": name, "image_id": rel,
            "mask_name": name, "mask_id": rel, "mask_type": "seg_npy",
            "rel_path": rel}


def _ap(name, ap="-5.35", **kw):
    return {**_seg(name, **kw), "ap_mm": ap, "ap_source": "filename"}


def _files(tmp_path, seg_rows, ap_rows, test_rows=()):
    _write(tmp_path / "train.csv", MANIFEST_FIELDS, seg_rows)
    _write(tmp_path / "test.csv", MANIFEST_FIELDS, test_rows)
    _write(tmp_path / "ap_position.csv", AP_FIELDS, ap_rows)
    return (tmp_path / "ap_position.csv", tmp_path / "train.csv", tmp_path / "test.csv")


def test_clean_manifest_passes(tmp_path):
    n = "TH_DHC-2076-5.35_R_seg.npy"
    r = validate_ap(*_files(tmp_path, [_seg(n)], [_ap(n)]))
    assert r["errors"] == [] and r["warnings"] == []
    assert r["ap_mm_min"] == r["ap_mm_max"] == -5.35


def test_coordinate_outside_the_lc_is_an_error(tmp_path):
    # -1.2 mm is nowhere near the locus coeruleus; the name was misread.
    n = "TH_DHC-2076-5.35_R_seg.npy"
    r = validate_ap(*_files(tmp_path, [_seg(n)], [_ap(n, ap="-1.20")]))
    assert any("outside the LC range" in e for e in r["errors"])


def test_positive_coordinate_is_an_error(tmp_path):
    # Names write the distance positive; forgetting to negate it is the single
    # most likely parsing bug, so it must not pass silently.
    n = "TH_DHC-2076-5.35_R_seg.npy"
    r = validate_ap(*_files(tmp_path, [_seg(n)], [_ap(n, ap="5.35")]))
    assert any("outside the LC range" in e for e in r["errors"])


def test_blank_coordinate_is_an_error(tmp_path):
    n = "TH_DHC-2076-5.35_R_seg.npy"
    r = validate_ap(*_files(tmp_path, [_seg(n)], [_ap(n, ap="")]))
    assert any("blank ap_mm" in e for e in r["errors"])


def test_animal_in_both_ap_splits_is_an_error(tmp_path):
    a, b = "TH_DHC-2076-5.35_R_seg.npy", "TH_DHC-2076-5.45_R_seg.npy"
    r = validate_ap(*_files(
        tmp_path,
        [_seg(a)], [_ap(a), _ap(b, split="test")], test_rows=[_seg(b, split="test")],
    ))
    assert any("both splits" in e for e in r["errors"])


def test_ap_row_missing_from_the_segmentation_manifests_warns(tmp_path):
    # The AP file is a view over train/test. Drift means someone edited one and
    # not the other.
    r = validate_ap(*_files(tmp_path, [], [_ap("TH_DHC-2076-5.35_R_seg.npy")]))
    assert any("neither train.csv nor test.csv" in w for w in r["warnings"])


def test_adding_drive_links_does_not_make_every_row_look_orphaned(tmp_path):
    # Regression: enrichment rewrites image_id in ap_position.csv only. Matching
    # on image_id would then flag all 500 rows as orphans.
    n = "TH_DHC-2076-5.35_R_seg.npy"
    ap_csv, train_csv, test_csv = _files(tmp_path, [_seg(n)], [_ap(n)])
    enrich(ap_csv, cache_path=tmp_path / "cache.json", lookup=lambda _: "FILEID")

    r = validate_ap(ap_csv, train_csv, test_csv)
    assert r["warnings"] == []
    assert r["errors"] == []


def test_absent_manifest_is_not_a_failure(tmp_path):
    r = validate_ap(tmp_path / "nope.csv", tmp_path / "t.csv", tmp_path / "s.csv")
    assert r == {"present": False, "n_rows": 0, "errors": [], "warnings": []}
