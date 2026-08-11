# The assertion set

Small image/mask pairs with **exactly known** ground truth, committed to git so
CI can run them on every push with no GPU, no Drive mount and no model download.

They are not real LC sections, and that is deliberate. Real images cannot go in
git (the project rule, and 185 GiB of them), and an assertion whose expected
answer is itself a human judgement cannot be asserted on. These are synthetic
crops where the number of cells and their exact boundaries are known by
construction, so a check against them is a check, not an opinion.

| Case | Cells | What it pins |
|---|---|---|
| `dense_24_cells` | 24 | Crowded field — the case where merging two touching neurons into one is easy to do and hard to notice. |
| `sparse_5_cells` | 5 | The ordinary case. |
| `empty_no_cells` | 0 | A section that missed the LC. Metrics must not divide by zero, and a model that hallucinates here must score 0, not be excused. |

## What CI checks with these

Two different things, and it is worth keeping them apart:

1. **The metrics code is correct** — `tests/test_metrics.py` and
   `tests/test_assertions.py` score these fixtures against known-good and
   known-bad predictions and assert the resulting numbers. This runs on every
   push. No model involved.

2. **The model still clears the bar** — `bluespotter.quality_gate` reads
   `reports/segmentation_metrics.json`, which Colab produced by segmenting the
   real 248-image held-out split, and fails the build if it regressed or if the
   report is stale. CI cannot recompute those numbers; it can and does refuse to
   accept old ones.

The second is the gate that protects the science. The first is what stops the
gate itself from being wrong — a quality check with a bug in its arithmetic is
worse than none, because it is believed.

## Regenerating

```bash
python -m bluespotter.make_assertions   # deterministic; seeded
```

Only regenerate if you are deliberately changing what the set covers. The
expected values in `expected.json` are the contract.
