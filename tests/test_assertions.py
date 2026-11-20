"""The assertion loop CI runs on every push.

These score the committed crops in `assertions/` — real held-out slices, real
human-drawn masks — against deliberately good and deliberately broken
predictions. No GPU, no Drive, no model download: this checks that the *scoring*
is trustworthy, which is the precondition for the quality gate meaning anything.

Nothing here hardcodes how many crops there are or what is in them. The contract
lives in `assertions/expected.json`, so regenerating the set with a different
sample changes the fixtures and not the tests.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bluespotter.metrics import aggregate, score_one
from bluespotter.quality_gate import check, check_freshness

ASSERTIONS = Path(__file__).resolve().parents[1] / "assertions"
EXPECTED = ASSERTIONS / "expected.json"

_NO_SET = pytest.mark.skipif(
    not EXPECTED.exists(),
    reason=("no assertion set committed yet — run "
            "`python -m bluespotter.make_assertions` in Colab and commit assertions/"),
)


def _contract() -> dict:
    return json.loads(EXPECTED.read_text())


def _cases():
    """Yield (case, image, masks) for every committed crop."""
    from skimage.io import imread

    for case in _contract()["cases"]:
        image = imread(str(ASSERTIONS / f"{case['name']}_image.png"))
        masks = imread(str(ASSERTIONS / f"{case['name']}_masks.png")).astype(np.int32)
        yield case, image, masks


@_NO_SET
def test_every_declared_case_is_present_and_matches_its_contract():
    # If someone regenerates the set and a cell count moves, that is a change to
    # what the gate promises and should be a deliberate, visible commit.
    cases = list(_cases())
    assert cases, "expected.json declares no cases"
    for case, image, masks in cases:
        assert image.shape[:2] == masks.shape[:2], case["name"]
        assert int(masks.max()) == case["n_cells"], case["name"]


@_NO_SET
def test_the_set_stays_small_enough_to_live_in_git():
    # Real image data in git is a deliberate, bounded exception to the project
    # rule. Bounded is the operative word: if this grows it has stopped being an
    # assertion set and become a dataset, and belongs back on Drive.
    total = sum(p.stat().st_size for p in ASSERTIONS.glob("*.png"))
    assert total < 20_000_000, f"assertion set is {total / 1e6:.1f} MB — too big for git"


def test_review_figures_are_not_committed():
    # They are a local eyeball check, rewritten on every model change. Committing
    # them puts 34 MB of derived panels into history per iteration, permanently,
    # to show something a person looks at once. The numbers CI actually gates on
    # are in reports/segmentation_metrics.json.
    figures = Path(__file__).resolve().parents[1] / "reports" / "figures"
    committed = [p.name for p in figures.glob("*.png")] if figures.is_dir() else []
    tracked = subprocess.run(
        ["git", "ls-files", "reports/figures"],
        cwd=figures.parents[1], capture_output=True, text=True,
    ).stdout.split()
    assert not tracked, (
        f"{len(tracked)} figure(s) are tracked by git — reports/figures is "
        f"gitignored on purpose; run `git rm -r --cached reports/figures`. "
        f"(untracked files present locally: {len(committed)}, which is fine)")


@_NO_SET
def test_the_set_covers_more_than_one_animal_and_includes_a_background_crop():
    # A gate built on one animal measures that animal. A gate with no empty crop
    # cannot catch a model that invents cells where there is no LC.
    contract = _contract()
    mice = {c["mouse"] for c in contract["cases"]}
    assert len(mice) >= 2, f"only {len(mice)} animal(s) in the assertion set"
    assert any(c["n_cells"] == 0 for c in contract["cases"]), \
        "no background crop — hallucination cannot be detected"

    # Both crop kinds must be present: a detail window tests boundary accuracy on
    # touching somata, a whole-LC view tests shape and total count. Neither
    # substitutes for the other.
    kinds = {c["kind"] for c in contract["cases"]}
    assert {"detail", "whole_lc"} <= kinds, f"missing crop kinds, have {kinds}"


@_NO_SET
def test_a_perfect_model_scores_perfectly():
    for case, _image, masks in _cases():
        s = score_one(masks, masks)
        assert s["n_pred"] == case["n_cells"], case["name"]
        assert s["count_error"] == 0, case["name"]
        if case["n_cells"]:
            assert s["per_threshold"]["0.5"]["f1"] == pytest.approx(1.0), case["name"]


@_NO_SET
def test_a_model_that_finds_nothing_scores_zero():
    # The degenerate optimum — predict background everywhere. No metric we report
    # may reward it.
    for case, _image, masks in _cases():
        if not case["n_cells"]:
            continue
        s = score_one(masks, np.zeros_like(masks))
        assert s["per_threshold"]["0.5"]["recall"] == 0.0, case["name"]
        assert s["count_error"] == -case["n_cells"], case["name"]


@_NO_SET
def test_under_segmentation_is_caught():
    # Every labelled neuron fused into one blob. On crowded LC fields this is
    # Cellpose's characteristic failure, it looks fine in a thumbnail, and it
    # destroys a cell count.
    for case, _image, masks in _cases():
        if case["n_cells"] < 3:
            continue
        fused = (masks > 0).astype(np.int32)
        s = score_one(masks, fused)
        assert s["n_pred"] == 1, case["name"]
        assert s["per_threshold"]["0.5"]["recall"] < 0.5, case["name"]


@_NO_SET
def test_hallucination_on_a_background_crop_is_penalised():
    for case, _image, masks in _cases():
        if case["n_cells"]:
            continue
        invented = np.zeros_like(masks)
        invented[10:40, 10:40] = 1
        s = score_one(masks, invented)
        assert s["per_threshold"]["0.5"]["precision"] == 0.0
        assert s["count_error"] == 1


@_NO_SET
def test_aggregate_over_the_whole_set_matches_the_contract():
    scores = [score_one(m, m) for _c, _i, m in _cases()]
    overall = aggregate(scores)
    contract = _contract()
    assert overall["n_images"] == contract["n_cases"]
    assert overall["n_true_total"] == contract["total_cells"]
    assert overall["count_error_total"] == 0


# --- the gate itself -------------------------------------------------------- #

_GATE = {
    "enabled": True,
    "f1": {"metric": "f1@0.5", "min": 0.7},
    "count_bias": {"metric": "count_bias_pct", "max": 10.0},
}


def test_gate_passes_good_metrics():
    ok, lines = check({"f1@0.5": 0.85, "count_bias_pct": 3.0}, _GATE)
    assert ok
    assert all("FAIL" not in line for line in lines)


def test_gate_fails_a_regression():
    ok, lines = check({"f1@0.5": 0.61, "count_bias_pct": 3.0}, _GATE)
    assert not ok
    assert any("FAIL" in line and "f1@0.5" in line for line in lines)


def test_gate_fails_a_missing_metric_rather_than_passing_silently():
    # A renamed metric must break the build, not quietly stop being checked.
    ok, lines = check({"count_bias_pct": 1.0}, _GATE)
    assert not ok
    assert any("MISSING" in line for line in lines)


def test_gate_reports_every_check_not_only_failures():
    _ok, lines = check({"f1@0.5": 0.9, "count_bias_pct": 1.0}, _GATE)
    assert len(lines) == 2       # a gate that is silent when happy gets ignored


def test_stale_metrics_are_rejected(tmp_path):
    # The failure this exists to catch: training code changes, nobody re-runs
    # evaluate, and CI goes green on numbers describing a model that is gone.
    (tmp_path / "test_index_summary.json").write_text(
        json.dumps({"dataset_hash": "NEWHASH0000"}))

    ok, lines = check_freshness({"dataset_hash_test": "OLDHASH0000"},
                                tmp_path / "dvc.lock", tmp_path)
    assert not ok
    assert any("re-run" in line for line in lines)


def test_fresh_metrics_are_accepted(tmp_path):
    (tmp_path / "test_index_summary.json").write_text(
        json.dumps({"dataset_hash": "SAMEHASH000"}))
    ok, _lines = check_freshness({"dataset_hash_test": "SAMEHASH000"},
                                 tmp_path / "dvc.lock", tmp_path)
    assert ok


def test_metrics_without_a_dataset_stamp_are_rejected(tmp_path):
    ok, _lines = check_freshness({}, tmp_path / "dvc.lock", tmp_path)
    assert not ok
