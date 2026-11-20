"""Link enrichment is optional by design: it must improve the manifest when the
Drive API is reachable and leave it usable when it is not."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from bluespotter.discover import AP_FIELDS
from bluespotter.drive_links import enrich


def _write_ap(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(AP_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in AP_FIELDS})


def _row(name: str, **kw) -> dict:
    base = {
        "split": "train", "source": "NM_tyr_titering", "mouse": "DHC-2076",
        "channel": "TH", "side": "R", "image_name": name,
        "image_id": f"NM_tyr_titering/masks/{name}", "mask_name": name,
        "mask_id": f"NM_tyr_titering/masks/{name}", "mask_type": "seg_npy",
        "rel_path": f"NM_tyr_titering/masks/{name}", "ap_mm": "-5.35",
    }
    base.update(kw)
    return base


def test_links_are_written_and_ids_upgraded(tmp_path):
    p = tmp_path / "ap_position.csv"
    _write_ap(p, [_row("TH_DHC-2076-5.35_R_seg.npy")])

    summary = enrich(p, cache_path=tmp_path / "cache.json",
                     lookup=lambda name: "FILEID123")

    assert summary["resolved"] == 1
    with open(p, newline="") as fh:
        row = next(iter(csv.DictReader(fh)))
    assert row["drive_url"] == "https://drive.google.com/file/d/FILEID123/view"
    # The path placeholder is replaced by the real ID once we know it.
    assert row["image_id"] == "FILEID123"
    assert row["mask_id"] == "FILEID123"


def test_existing_drive_ids_are_not_clobbered(tmp_path):
    # A hand-authored row already carries a real ID. A name lookup can collide
    # across cohorts, so the ID we already trust wins.
    p = tmp_path / "ap_position.csv"
    _write_ap(p, [_row("TH_DHC-2076-5.35_R_seg.npy",
                       image_id="REAL_ID", mask_id="REAL_ID")])

    enrich(p, cache_path=tmp_path / "cache.json", lookup=lambda name: "GUESSED")

    with open(p, newline="") as fh:
        row = next(iter(csv.DictReader(fh)))
    assert row["image_id"] == "REAL_ID"
    assert row["drive_url"].endswith("GUESSED/view")  # link still offered


def test_unresolved_names_leave_the_row_intact(tmp_path):
    p = tmp_path / "ap_position.csv"
    _write_ap(p, [_row("TH_DHC-2076-5.35_R_seg.npy")])

    summary = enrich(p, cache_path=tmp_path / "cache.json", lookup=lambda name: "")

    assert summary["unresolved"] == 1
    with open(p, newline="") as fh:
        row = next(iter(csv.DictReader(fh)))
    assert row["drive_url"] == ""
    assert row["image_id"].endswith("TH_DHC-2076-5.35_R_seg.npy")  # still resolvable


def test_lookups_are_cached_across_runs(tmp_path):
    p = tmp_path / "ap_position.csv"
    cache = tmp_path / "cache.json"
    _write_ap(p, [_row("a_seg.npy", image_name="a_seg.npy"),
                  _row("a_seg.npy", image_name="a_seg.npy")])

    calls: list[str] = []

    def lookup(name):
        calls.append(name)
        return "X"

    enrich(p, cache_path=cache, lookup=lookup)
    assert len(calls) == 1                       # two rows, one distinct name
    assert json.loads(cache.read_text()) == {"a_seg.npy": "X"}

    enrich(p, cache_path=cache, lookup=lookup)
    assert len(calls) == 1                       # second run hits the cache


def test_missing_file_is_not_an_error(tmp_path):
    # No AP manifest yet is a normal state, not a pipeline failure.
    assert enrich(tmp_path / "nope.csv")["status"] == "missing"
