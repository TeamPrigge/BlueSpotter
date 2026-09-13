"""The blind counting export and the agreement statistics.

The export exists to measure how much human counts differ. Two things would
quietly ruin it: somata too small to count, and filenames that leak which
animal a section came from. Both are pinned here.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from bluespotter.scoring_set import (
    MIN_SOMA_PX,
    TARGET_SOMA_PX,
    agreement,
    anonymous_ids,
    choose_scale,
    soma_diameter,
)


def _mask(n_cells=10, radius=15, shape=(600, 600)):
    m = np.zeros(shape, np.int32)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    for i in range(n_cells):
        cy, cx = 40 + (i // 5) * 120, 40 + (i % 5) * 120
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = i + 1
    return m


def test_soma_diameter_is_measured_not_assumed():
    # Equivalent diameter from area: a radius-15 disc is 30 px across.
    assert soma_diameter(_mask(radius=15)) == pytest.approx(30, abs=1.5)


def test_big_somata_are_scaled_down_to_the_countable_target():
    scale, ok = choose_scale(_mask(radius=25))       # 50 px across
    assert ok
    assert 50 * scale == pytest.approx(TARGET_SOMA_PX, abs=1.5)


def test_small_somata_are_never_upscaled():
    # Interpolating a 6 px soma up to 15 px does not add information, it just
    # makes a larger blur, and would hide that the section is hard to count.
    scale, ok = choose_scale(_mask(radius=3))
    assert scale == 1.0
    assert ok is (6 >= MIN_SOMA_PX)


def test_a_section_that_cannot_reach_the_minimum_is_flagged_not_silently_shipped():
    # The failure the assertion set hit at x0.15: a huge slide downscaled to fit
    # a box leaves 3 px somata that nobody can count. It must be declared.
    huge = np.zeros((200, 40000), np.int32)
    huge[90:110, 100:120] = 1                       # 20 px soma on a vast slide
    scale, ok = choose_scale(huge)
    assert not ok
    assert 20 * scale < MIN_SOMA_PX


def test_ids_are_shuffled_so_filenames_leak_no_ordering():
    # Sequential IDs would tell a scorer which sections are neighbours in the
    # same animal, which is exactly the context blinding is meant to remove.
    ids = anonymous_ids(50)
    assert sorted(ids) == [f"LC-{i:03d}" for i in range(1, 51)]
    assert ids != sorted(ids)
    assert anonymous_ids(50) == ids                 # deterministic


def test_perfect_agreement_scores_one():
    counts = {f"LC-{i:03d}": i * 3 for i in range(1, 21)}
    r = agreement({"a": counts, "b": dict(counts), "c": dict(counts)})
    assert r["icc_2_1"] == pytest.approx(1.0, abs=1e-6)
    assert r["mean_cv_pct"] == pytest.approx(0.0, abs=1e-6)


def test_a_constant_offset_is_caught_even_though_it_correlates_perfectly():
    # A rater who counts 10 more every time correlates at r=1.0. Reporting
    # correlation would call that perfect agreement; ICC and bias must not.
    base = {f"LC-{i:03d}": i * 3 for i in range(1, 21)}
    off = {k: v + 10 for k, v in base.items()}
    r = agreement({"a": base, "b": off})
    assert r["icc_2_1"] < 0.95
    assert r["pairwise"]["a vs b"]["bias"] == pytest.approx(-10.0)


def test_the_reference_masks_are_treated_as_one_more_rater():
    base = {f"LC-{i:03d}": 20 for i in range(1, 11)}
    r = agreement({"a": base, "b": base}, reference={k: 25 for k in base})
    assert "reference" in r["raters"]
    assert "a vs reference" in r["pairwise"]


def test_sections_not_scored_by_everyone_are_excluded():
    a = {"LC-001": 5, "LC-002": 7}
    b = {"LC-001": 6}
    assert agreement({"a": a, "b": b})["n_sections"] == 1


def test_the_outline_subset_is_spread_across_animals():
    # A flat random 50-of-248 would by chance pile several sections onto a few
    # mice, and outline agreement would then partly measure how hard that one
    # animal is rather than how much scorers differ.
    key = [{"scoring_id": f"LC-{i:03d}", "mouse": f"M{i % 20}"} for i in range(1, 249)]
    from bluespotter.scoring_set import choose_label_subset
    picked = choose_label_subset(key, 50)
    assert len(picked) == 50
    mice = {r["mouse"] for r in key if r["scoring_id"] in set(picked)}
    assert len(mice) == 20                       # every animal represented
    assert choose_label_subset(key, 50) == picked  # deterministic


def test_asking_for_more_sections_than_exist_returns_all_of_them():
    from bluespotter.scoring_set import choose_label_subset
    key = [{"scoring_id": f"LC-{i:03d}", "mouse": "M1"} for i in range(1, 6)]
    assert len(choose_label_subset(key, 50)) == 5
