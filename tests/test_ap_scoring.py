"""AP levels: recovering Csilla's bins, and scoring agreement on an ordinal scale."""
from __future__ import annotations

import csv

import pytest

from bluespotter.ap_scoring import (
    Bin,
    ap_agreement,
    assign_bin,
    coverage,
    derive_bins,
    read_ap_manifest,
    read_wt_mice,
)


def test_bins_are_recovered_from_the_values_actually_used():
    # Csilla worked to a fixed set of levels. Seven distinct values, each
    # recorded a few times with slight rounding -> seven bins, not a continuum.
    vals = ([-4.84] * 3 + [-4.85] + [-5.02] * 4 + [-5.20] * 6 + [-5.34] * 5
            + [-5.52] * 4 + [-5.68] * 3 + [-5.80] * 2)
    bins = derive_bins(vals)
    assert len(bins) == 7
    assert [b.label for b in bins] == [f"AP{i}" for i in range(1, 8)]


def test_ap1_is_the_most_rostral_level():
    # Values are negative, so rostral is the least negative. Getting this
    # backwards would silently invert the whole scale.
    bins = derive_bins([-5.80, -4.84, -5.20])
    assert bins[0].centre_mm == -4.84
    assert bins[-1].centre_mm == -5.80


def test_values_within_tolerance_are_one_level_not_two():
    assert len(derive_bins([-5.30, -5.35])) == 1     # 0.05 apart
    assert len(derive_bins([-5.30, -5.55])) == 2     # 0.25 apart


def test_a_value_is_assigned_to_its_nearest_level():
    bins = derive_bins([-4.84, -5.20, -5.80])
    assert assign_bin(-5.22, bins) == "AP2"
    assert assign_bin(None, bins) is None


def test_wt_filtering_is_explicit_and_never_guessed_from_cohort(tmp_path):
    # The manifest has no condition column, and cohort folder names only *look*
    # like they encode genotype. Guessing would put degenerated LCs into a model
    # that learns shape -> position. It must come from a supplied table.
    ap = tmp_path / "ap.csv"
    rows = [
        {"mouse": "DHC-0001", "image_name": "a.tif", "ap_mm": "-5.20",
         "source": "NM_hightiter_behavior_brains"},
        {"mouse": "DHC-0002", "image_name": "b.tif", "ap_mm": "-5.20",
         "source": "NM_hightiter_behavior_brains"},
    ]
    with open(ap, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    assert len(read_ap_manifest(ap)) == 2
    # Same cohort, different conditions: only the supplied table can tell them
    # apart, and it does.
    assert len(read_ap_manifest(ap, wt_mice={"DHC-0001"})) == 1


def test_rows_without_a_bregma_value_are_not_invented(tmp_path):
    ap = tmp_path / "ap.csv"
    rows = [{"mouse": "M1", "image_name": "a.tif", "ap_mm": "-5.20", "source": "x"},
            {"mouse": "M1", "image_name": "b.tif", "ap_mm": "", "source": "x"},
            {"mouse": "M1", "image_name": "c.tif", "ap_mm": "junk", "source": "x"}]
    with open(ap, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    assert len(read_ap_manifest(ap)) == 1


def test_a_condition_table_with_no_wt_animals_is_an_error_not_an_empty_filter(tmp_path):
    # Returning an empty set would filter the whole dataset away and look like
    # missing data rather than a misconfigured file.
    p = tmp_path / "cond.csv"
    p.write_text("mouse,condition\nDHC-0001,NM-hightiter\nDHC-0002,NM-lowtiter\n")
    with pytest.raises(ValueError, match="none with condition"):
        read_wt_mice(p)


def test_wt_table_accepts_the_usual_spellings(tmp_path):
    p = tmp_path / "cond.csv"
    p.write_text("mouse,condition\nA,WT\nB,wild-type\nC,control\nD,NM-hightiter\n")
    assert read_wt_mice(p) == {"A", "B", "C"}


def test_coverage_reports_what_is_missing():
    allr = [{"image_name": f"{i}.tif", "mouse": f"M{i % 3}", "source": "coh"}
            for i in range(9)]
    ap = [{"image_name": "0.tif", "mouse": "M0"}, {"image_name": "3.tif", "mouse": "M0"}]
    c = coverage(allr, ap)
    assert c["n_sections"] == 9 and c["n_with_ap"] == 2 and c["n_without_ap"] == 7
    assert set(c["mice_without_any_ap"]) == {"M1", "M2"}


# --- agreement ------------------------------------------------------------- #
def _bins(n=7):
    return [Bin(i + 1, f"AP{i + 1}", -4.8 - 0.16 * i, 5) for i in range(n)]


def test_perfect_ap_agreement():
    s = {f"LC-{i:03d}": f"AP{(i % 7) + 1}" for i in range(1, 21)}
    r = ap_agreement({"a": s, "b": dict(s)}, _bins())
    assert r["pairwise"]["a vs b"]["weighted_kappa"] == pytest.approx(1.0)
    assert r["sections_with_full_agreement_pct"] == 100.0


def test_adjacent_bin_errors_score_better_than_distant_ones():
    # The reason for quadratic weighting: a near miss on an ordinal scale is
    # not the same mistake as calling rostral caudal, and plain accuracy
    # cannot tell them apart.
    truth = {f"LC-{i:03d}": "AP4" for i in range(1, 31)}
    near = {k: "AP5" for k in truth}
    far = {k: "AP1" for k in truth}
    k_near = ap_agreement({"a": truth, "b": near}, _bins())["pairwise"]["a vs b"]
    k_far = ap_agreement({"a": truth, "b": far}, _bins())["pairwise"]["a vs b"]
    assert k_near["within_one_bin_pct"] == 100.0
    assert k_far["within_one_bin_pct"] == 0.0
    assert k_near["mean_signed_bins"] == pytest.approx(-1.0)
    assert k_far["mean_signed_bins"] == pytest.approx(3.0)


def test_csillas_values_enter_as_the_reference_rater():
    s = {f"LC-{i:03d}": "AP3" for i in range(1, 11)}
    r = ap_agreement({"a": s}, _bins(), reference={k: "AP3" for k in s})
    assert "reference" in r["raters"]
    assert "a vs reference" in r["pairwise"]
