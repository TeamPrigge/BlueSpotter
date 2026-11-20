"""The metrics are the thing everything else is judged by, so they get judged
first. A bug here is worse than no metric at all, because a number gets believed."""
from __future__ import annotations

import numpy as np
import pytest

from bluespotter.metrics import (
    aggregate,
    match_instances,
    precision_recall_curve,
    score_one,
)


def _disc(canvas, cx, cy, r, label):
    yy, xx = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
    canvas[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
    return canvas


def _three_cells():
    m = np.zeros((100, 100), dtype=np.int32)
    _disc(m, 20, 20, 8, 1)
    _disc(m, 50, 50, 8, 2)
    _disc(m, 80, 80, 8, 3)
    return m


def test_perfect_prediction_scores_one():
    m = _three_cells()
    tp, fp, fn = match_instances(m, m, 0.5)
    assert (tp, fp, fn) == (3, 0, 0)


def test_instance_ids_need_not_agree():
    # The model has no way to know a human called this cell "2". Only the
    # partition matters, not the numbering.
    m = _three_cells()
    relabelled = np.where(m > 0, 100 - m, 0)
    tp, fp, fn = match_instances(m, relabelled, 0.5)
    assert (tp, fp, fn) == (3, 0, 0)


def test_a_missed_cell_is_a_false_negative():
    truth = _three_cells()
    pred = truth.copy()
    pred[pred == 3] = 0
    tp, fp, fn = match_instances(truth, pred, 0.5)
    assert (tp, fp, fn) == (2, 0, 1)


def test_an_invented_cell_is_a_false_positive():
    truth = _three_cells()
    pred = _disc(truth.copy(), 20, 80, 8, 4)
    tp, fp, fn = match_instances(truth, pred, 0.5)
    assert (tp, fp, fn) == (3, 1, 0)


def test_two_cells_merged_into_one_costs_both_ways():
    # Under-segmentation is the classic Cellpose failure on crowded LC fields.
    # It must show as one hit, one miss and no free pass.
    truth = np.zeros((60, 30), dtype=np.int32)
    truth[5:25, 5:25] = 1
    truth[30:50, 5:25] = 2
    merged = np.zeros_like(truth)
    merged[5:50, 5:25] = 1

    tp, _fp, fn = match_instances(truth, merged, 0.5)
    assert tp <= 1 and fn >= 1


def test_tighter_threshold_is_stricter():
    # A sloppy but roughly-right outline should pass at 0.5 and fail at 0.9,
    # which is the entire point of reporting more than one threshold.
    truth = np.zeros((40, 40), dtype=np.int32)
    truth[10:30, 10:30] = 1          # 400 px
    loose = np.zeros_like(truth)
    loose[10:30, 10:26] = 1          # 320 px, IoU = 0.8

    assert match_instances(truth, loose, 0.5)[0] == 1
    assert match_instances(truth, loose, 0.9)[0] == 0


def test_empty_ground_truth_and_empty_prediction():
    empty = np.zeros((20, 20), dtype=np.int32)
    assert match_instances(empty, empty, 0.5) == (0, 0, 0)

    s = score_one(empty, empty)
    # No cells, none predicted, nothing invented. Scores must be defined, not NaN.
    assert s["n_true"] == s["n_pred"] == 0
    assert s["per_threshold"]["0.5"]["f1"] == 0.0


def test_hallucinating_on_an_empty_section_is_penalised():
    # A model that finds cells where there is no LC must not score well just
    # because there was nothing to miss.
    empty = np.zeros((40, 40), dtype=np.int32)
    invented = _disc(empty.copy(), 20, 20, 6, 1)
    s = score_one(empty, invented)
    assert s["per_threshold"]["0.5"]["precision"] == 0.0
    assert s["count_error"] == 1


def test_count_error_is_signed():
    # Systematic over- and under-counting are different problems with different
    # causes; an absolute value would hide which one you have.
    truth = _three_cells()
    under = truth.copy()
    under[under == 3] = 0
    assert score_one(truth, under)["count_error"] == -1
    assert score_one(under, truth)["count_error"] == 1


def test_aggregate_pools_counts_rather_than_averaging_ratios():
    # One image with 100 cells and one with 1 must not carry equal weight: LC
    # cell number varies along the rostrocaudal axis, so averaging per-image
    # ratios would bias the score toward the small end of the structure.
    big = {"n_true": 100, "n_pred": 100, "count_error": 0,
           "per_threshold": {"0.5": {"tp": 50, "fp": 50, "fn": 50,
                                     "precision": 0.5, "recall": 0.5,
                                     "f1": 0.5, "ap": 0.33}}}
    small = {"n_true": 1, "n_pred": 1, "count_error": 0,
             "per_threshold": {"0.5": {"tp": 1, "fp": 0, "fn": 0,
                                       "precision": 1.0, "recall": 1.0,
                                       "f1": 1.0, "ap": 1.0}}}

    pooled = aggregate([big, small], (0.5,))
    # Pooled: 51 tp / 50 fp -> 0.505. Averaging the ratios would give 0.75.
    assert pooled["precision@0.5"] == pytest.approx(51 / 101, abs=1e-6)


def test_count_bias_is_reported_as_a_percentage():
    s = [{"n_true": 100, "n_pred": 110, "count_error": 10,
          "per_threshold": {"0.5": {"tp": 100, "fp": 10, "fn": 0,
                                    "precision": 0.0, "recall": 0.0,
                                    "f1": 0.0, "ap": 0.0}}}]
    assert aggregate(s, (0.5,))["count_bias_pct"] == pytest.approx(10.0)


def test_pr_curve_degrades_monotonically_with_threshold():
    # Demanding tighter boundary agreement can never find more cells.
    truth = _three_cells()
    pred = truth.copy()
    scores = [score_one(truth, pred, tuple(round(t, 2) for t in np.arange(0.5, 1.0, 0.05)))]
    rows = precision_recall_curve(scores)
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
