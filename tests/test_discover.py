"""Discovery turns a Drive tree into manifests. The properties worth pinning are
the ones that are expensive to notice later: that no animal straddles the split,
that duplicate copies of a slice collapse to one row, and that the AP manifest
contains only real coordinates."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bluespotter.discover import assign_splits, build, dedupe, walk


def _touch(p: Path, data: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.fixture
def nm_root(tmp_path: Path) -> Path:
    """A miniature of the real NM_Slices tree: three cohorts, three conventions."""
    root = tmp_path / "NM_Slices"

    # Cohort A — AP in the name, image+mask in one .npy.
    a = root / "NM_tyr_titering" / "masks"
    for mouse in ("2030", "2076"):
        for ap in ("5.20", "5.40", "5.60"):
            for side in ("L", "R"):
                _touch(a / f"TH_DHC-{mouse}-{ap}_{side}_seg.npy")

    # Cohort B — Zeiss slices, separate image and cp_masks, no AP.
    b_img = root / "NM_hightiter_histology_brains_ED" / "processed" / "cropped" / "TH"
    b_msk = root / "NM_hightiter_histology_brains_ED" / "masks" / "TH"
    for sl in ("slice1", "slice2"):
        for side in ("L", "R"):
            _touch(b_img / f"DHC_0929_{sl}_TH_{side}.tif")
            _touch(b_msk / f"DHC_0929_{sl}_TH_{side}_cp_masks.png")

    # Cohort C — Olympus vsi, ordinal AP only.
    c = root / "NM_lowtiter_histology_brains" / "masks" / "npy_masks"
    _touch(c / "DHC_0464_02.vsi - ap2_TH_left_seg.npy")
    _touch(c / "DHC_0464_05.vsi - ap3_TH_left_seg.npy")

    return root


def test_walk_finds_all_three_conventions(nm_root):
    pairs, unparsed = walk(nm_root)
    assert not unparsed
    kinds = {p.meta.kind for p in pairs}
    assert kinds == {"ap_mm", "slice_index", "ap_ordinal"}
    assert len(pairs) == 12 + 4 + 2


def test_cp_masks_pair_across_sibling_directories(nm_root):
    # The ED cohort keeps images and masks in different subtrees; a pairing that
    # only looked in the same directory would silently drop the whole cohort.
    pairs, _ = walk(nm_root)
    ed = [p for p in pairs if p.mask_type == "cp_masks_png"]
    assert len(ed) == 4
    assert all(p.image != p.mask for p in ed)
    assert all(p.image.suffix == ".tif" and p.mask.suffix == ".png" for p in ed)


def test_duplicate_copies_collapse_to_the_canonical_folder(nm_root):
    # The same slice often exists in both masks/ and processed/cropped2/.
    dup = nm_root / "NM_tyr_titering" / "processed" / "cropped2" / "TH"
    (dup).mkdir(parents=True, exist_ok=True)
    (dup / "TH_DHC-2030-5.20_L_seg.npy").write_bytes(b"x")

    pairs, _ = walk(nm_root)
    kept, dropped = dedupe(pairs)
    assert dropped == 1
    survivor = next(p for p in kept
                    if p.meta.mouse == "DHC-2030" and p.meta.ap_mm == -5.20
                    and p.meta.side == "L")
    assert "masks" in survivor.rel_dir


def test_no_mouse_appears_in_both_splits(nm_root, tmp_path):
    out = tmp_path / "out"
    build(nm_root, out, test_fraction=0.3)

    def mice(name):
        with open(out / name, newline="") as fh:
            return {r["mouse"] for r in csv.DictReader(fh)}

    assert mice("train.csv") & mice("test.csv") == set()


def test_split_is_deterministic(nm_root, tmp_path):
    # Two runs on two machines must agree, or every comparison between
    # experiments is confounded by a reshuffled test set.
    pairs, _ = walk(nm_root)
    pairs, _ = dedupe(pairs)
    assert assign_splits(pairs, 0.3) == assign_splits(pairs, 0.3)


def test_ap_manifest_holds_only_real_coordinates(nm_root, tmp_path):
    out = tmp_path / "out"
    summary = build(nm_root, out, test_fraction=0.3)

    with open(out / "ap_position.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # 12 files from cohort A carry a bregma coordinate. The ordinal `ap2` rows
    # and the slice-index rows must not be here: neither is a millimetre value,
    # and mixing them would train the AP model on two incompatible scales.
    assert len(rows) == 12
    assert summary["ap_rows"] == 12
    assert all(r["mask_type"] == "seg_npy" for r in rows)
    assert all(-6.5 <= float(r["ap_mm"]) <= -4.5 for r in rows)
    assert not any("vsi" in r["image_name"] for r in rows)


def test_ap_rows_are_also_in_the_segmentation_manifests(nm_root, tmp_path):
    # The AP subset is a *view*, not a separate dataset: those slices should
    # still train the segmenter.
    out = tmp_path / "out"
    build(nm_root, out, test_fraction=0.3)

    def ids(name):
        with open(out / name, newline="") as fh:
            return {r["image_id"] for r in csv.DictReader(fh)}

    assert ids("ap_position.csv") <= (ids("train.csv") | ids("test.csv"))


def test_unparsed_files_are_reported_not_silently_dropped(nm_root, tmp_path):
    _touch(nm_root / "NM_tyr_titering" / "masks" / "mystery_thing_seg.npy")
    summary = build(nm_root, tmp_path / "out", test_fraction=0.3, dry_run=True)
    assert summary["unparsed_files"] == 1
    assert "mystery_thing_seg.npy" in summary["unparsed_examples"][0]


def test_dry_run_writes_nothing(nm_root, tmp_path):
    out = tmp_path / "out"
    build(nm_root, out, dry_run=True)
    assert not out.exists()


def test_missing_mount_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Mount Drive first"):
        walk(tmp_path / "nope")
