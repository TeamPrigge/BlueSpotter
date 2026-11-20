"""Enforce a minimum quality bar on the last recorded evaluation.

WHAT CI CAN AND CANNOT DO HERE
------------------------------
The honest constraint first: a GitHub runner cannot re-run the model. It has no
GPU, no Drive mount, and pulling Cellpose-SAM weights on every push would be
minutes of download for a check that still could not see the test images. So
"run the assertions on every push" cannot mean "re-segment 248 slices in CI".

What it can mean, and what this does, is the standard model-registry gate:

  1. Colab runs `evaluate`, which segments the held-out set and writes
     reports/segmentation_metrics.json. That file is committed.
  2. CI checks, on every push, that the committed numbers clear the thresholds
     in params.yaml, and that they were produced by the dataset and code
     currently in the tree — not by an older run left lying around.

Point 2 is the one that does the real work. A stale metrics file is how a model
quietly gets worse without anyone noticing: someone changes the training code,
does not re-evaluate, and the green tick comes from numbers that describe a model
that no longer exists. So the freshness check is a failure, not a warning.

The per-image assertion set (see `assertions/`) covers the other half — that the
evaluation logic itself is right — and *that* runs for real in CI, on small
committed crops, with no GPU needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Metric name -> (comparison, human description). Thresholds come from
# params.yaml so the bar is versioned alongside everything else.
_COMPARISONS = {
    "min": (lambda value, bar: value >= bar, ">="),
    "max": (lambda value, bar: value <= bar, "<="),
}


def check(metrics: dict[str, Any], gate: dict[str, Any]) -> tuple[bool, list[str]]:
    """Compare recorded metrics against the configured bar.

    Returns (passed, lines) where `lines` is human-readable and always includes
    every check, passing or failing — a gate that only prints failures teaches
    people to ignore it when it is quiet.
    """
    lines: list[str] = []
    ok = True

    for key, spec in sorted(gate.items()):
        if key.startswith("_") or not isinstance(spec, dict):
            continue
        metric = spec.get("metric", key)
        if metric not in metrics:
            lines.append(f"  MISSING  {metric} not in the metrics file")
            ok = False
            continue
        value = float(metrics[metric])
        for kind, (compare, symbol) in _COMPARISONS.items():
            if kind not in spec:
                continue
            bar = float(spec[kind])
            passed = compare(value, bar)
            ok &= passed
            lines.append(
                f"  {'PASS' if passed else 'FAIL'}     {metric} = {value:.4f} "
                f"({symbol} {bar})"
            )
    return ok, lines


def check_freshness(metrics: dict[str, Any], dvc_lock: Path,
                    reports_dir: Path) -> tuple[bool, list[str]]:
    """Verify the metrics describe the dataset currently pinned in dvc.lock.

    `evaluate` records the test-split dataset hash it scored against. If that no
    longer matches what the index says, the numbers describe different data and
    are not evidence about this commit.
    """
    lines: list[str] = []
    recorded = metrics.get("dataset_hash_test")
    summary = reports_dir / "test_index_summary.json"

    if not recorded:
        lines.append("  FAIL     metrics file has no dataset_hash_test — "
                     "re-run `dvc repro evaluate`")
        return False, lines
    if not summary.exists():
        lines.append("  SKIP     no test_index_summary.json to compare against")
        return True, lines

    current = json.loads(summary.read_text()).get("dataset_hash", "")
    if recorded != current:
        lines.append(
            f"  FAIL     metrics were computed on dataset {recorded[:12]} but the "
            f"index now pins {current[:12]} — re-run `dvc repro evaluate` in Colab"
        )
        return False, lines

    lines.append(f"  PASS     metrics match the pinned test dataset {current[:12]}")
    return True, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail the build if model quality regressed.")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--metrics", default="reports/segmentation_metrics.json")
    ap.add_argument("--allow-missing", action="store_true",
                    help="warn instead of failing when no evaluation exists yet")
    args = ap.parse_args(argv)

    import yaml

    gate = yaml.safe_load(Path(args.params).read_text()).get("quality_gate", {})
    if not gate.get("enabled", False):
        print("[gate] quality_gate.enabled is false — skipping")
        return 0

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        msg = (f"no evaluation found at {metrics_path}. Run `dvc repro evaluate` "
               f"in Colab and commit the report.")
        if args.allow_missing:
            print(f"::warning title=No evaluation::{msg}")
            return 0
        print(f"[gate] FAILED: {msg}")
        return 1

    metrics = json.loads(metrics_path.read_text())
    print(f"[gate] checking {metrics_path}")

    fresh_ok, fresh_lines = check_freshness(
        metrics, Path("dvc.lock"), metrics_path.parent)
    gate_ok, gate_lines = check(metrics, gate)

    for line in fresh_lines + gate_lines:
        print(line)

    if fresh_ok and gate_ok:
        print("[gate] all quality checks passed")
        return 0
    print("[gate] FAILED — see the lines marked FAIL above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
