"""The file-name grammar is load-bearing: it is the only record of which slice
sits where. These tests pin the real names seen on Drive, including the ugly
ones, and — more importantly — pin the cases we must *refuse* to parse, because
a silently misread coordinate is worse than a missing one."""
from __future__ import annotations

import pytest

from bluespotter.naming import parse, strip_role_suffix


@pytest.mark.parametrize("name,mouse,channel,side,ap", [
    ("TH_DHC-2076-5.35_R_seg.npy",        "DHC-2076", "TH", "R", -5.35),
    ("TH_DHC-2030-5.50_L_cp_masks.png",   "DHC-2030", "TH", "L", -5.50),
    ("HA_DHC_1351-5.30_R_seg.npy",        "DHC-1351", "HA", "R", -5.30),   # underscore
    ("HA_DHC-0632-5.20_R .seg.npy",       "DHC-0632", "HA", "R", -5.20),   # stray space
    ("HA_DHC-0542-4.84_R_seg.npy",        "DHC-0542", "HA", "R", -4.84),
    ("HA_DHC-1668-5.45_R_seg_2.npy",      "DHC-1668", "HA", "R", -5.45),   # re-segmented
])
def test_ap_in_millimetres(name, mouse, channel, side, ap):
    p = parse(name)
    assert p.kind == "ap_mm"
    assert (p.mouse, p.channel, p.side) == (mouse, channel, side)
    assert p.ap_mm == pytest.approx(ap)
    assert p.has_ap


def test_ap_is_stored_negative_because_the_atlas_reads_that_way():
    # Names write the distance behind bregma as a positive number; every atlas
    # and every paper writes it negative. Normalise once, here.
    assert parse("TH_DHC-2076-5.35_R_seg.npy").ap_mm < 0


@pytest.mark.parametrize("name", [
    "TH_DHC-2076-10.0_R_seg.npy",   # magnification, not a coordinate
    "TH_DHC-2076-0.25_R_seg.npy",   # far outside the LC
])
def test_out_of_range_numbers_are_not_read_as_ap(name):
    # Refusing is the whole point: an LC slice is never at bregma -0.25 mm, so
    # accepting it would poison the AP dataset with confident nonsense.
    assert parse(name).ap_mm is None


def test_ordinal_ap_is_kept_separate_from_millimetres():
    p = parse("DHC_0464_02.vsi - ap2_TH_left_seg.npy")
    assert p.kind == "ap_ordinal"
    assert (p.mouse, p.channel, p.side, p.ap_index) == ("DHC-0464", "TH", "L", 2)
    # `ap2` orders sections within one animal; it is not comparable across
    # animals, so it must never masquerade as a real coordinate.
    assert p.ap_mm is None
    assert not p.has_ap


def test_slice_index_cohort_has_no_ap():
    p = parse("DHC_0929_slice4_TH_R.tif")
    assert p.kind == "slice_index"
    assert (p.mouse, p.slice, p.channel, p.side) == ("DHC-0929", "slice4", "TH", "R")
    assert not p.has_ap


def test_slidescanner_tiles():
    p = parse("M1_PM_13_merged.ome_CH2_reduced_seg.npy")
    assert p.kind == "slidescanner"
    assert (p.mouse, p.slice, p.channel) == ("M1", "PM_13", "CH2")


def test_unknown_names_are_reported_not_guessed():
    assert parse("scratch_notes_final_v2.png").kind == "unknown"


@pytest.mark.parametrize("name,stem,role", [
    ("DHC_0929_slice4_TH_R.tif",              "DHC_0929_slice4_TH_R", "image"),
    ("DHC_0929_slice4_TH_R_cp_masks.png",     "DHC_0929_slice4_TH_R", "mask"),
    ("TH_DHC-2076-5.35_R_seg.npy",            "TH_DHC-2076-5.35_R",   "mask"),
    ("HA_DHC-0632-5.20_R .seg.npy",           "HA_DHC-0632-5.20_R",   "mask"),
])
def test_stem_and_role(name, stem, role):
    # Pairing a mask back to its image depends entirely on both collapsing to
    # the same stem.
    assert strip_role_suffix(name) == (stem, role)


def test_left_right_spellings_all_normalise():
    sides = {parse(n).side for n in (
        "DHC_0464_01.vsi - ap6_TH_leftG_seg.npy",
        "DHC_0464_02.vsi - ap2_TH_left_seg.npy",
    )}
    assert sides == {"L"}
