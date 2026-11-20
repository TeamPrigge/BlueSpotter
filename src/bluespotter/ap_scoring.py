"""Rostrocaudal (AP) level as a human-scorable, model-trainable label.

WHERE THE BINS COME FROM
------------------------
They are not invented here. Csilla already assigned a bregma coordinate to most
sections, encoded in the filename (`TH_DHC-2076-5.35_R` -> -5.35 mm), and those
values are discrete — she worked to a fixed set of levels, not a continuum. So
`derive_bins()` reads the values actually present and recovers that set. Picking
a round number of bins instead would either split levels she treated as one or
merge levels she distinguished, and would make her existing labels and the
students' new ones incomparable.

WHY BINS AND NOT MILLIMETRES
----------------------------
A scorer judging one section from LC shape and the sliver of fourth ventricle
cannot resolve 0.05 mm, and asking for a number invites false precision that
then has to be un-invented during analysis. An ordinal level is what the eye
actually delivers. It also makes the eventual model an ordinal classification,
which can be evaluated honestly — adjacent-bin errors are not the same as
rostral-called-caudal, and weighted kappa says so while accuracy does not.

TWO DIFFERENT USES, DO NOT MIX THEM
-----------------------------------
- Csilla's filename values are **labels**: they come from the sectioning record,
  not from looking at the picture. These train the model.
- The students' judgements are a **human baseline**: how well a trained eye
  recovers AP from the image alone. These must never become training labels, or
  the model learns to reproduce human guessing rather than the sectioning truth.

THE WT RESTRICTION IS NOT ENFORCEABLE YET
-----------------------------------------
Neuromelanin overexpression changes the LC outline, so shape->AP may only be
learned from non-degenerated animals. The manifest has no genotype or condition
column, so nothing here can filter on it. Every entry point takes an explicit
`wt_mice` set and refuses to guess from cohort folder names: `NM_hightiter_*`
*looks* like an overexpression cohort, but acting on that hunch would silently
poison the training set in exactly the way that must not happen.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Two AP values closer than this are the same level recorded with slightly
# different precision, not two levels. 0.1 mm is well below section spacing and
# well above rounding noise in the filenames.
BIN_TOLERANCE_MM = 0.1


@dataclass(frozen=True)
class Bin:
    index: int          # 1 = most rostral
    label: str          # "AP1"
    centre_mm: float    # negative, atlas convention
    n_sections: int

    def describe(self) -> str:
        return f"{self.label}  ({self.centre_mm:+.2f} mm)  n={self.n_sections}"


def derive_bins(ap_values, tolerance: float = BIN_TOLERANCE_MM) -> list[Bin]:
    """Recover the discrete AP levels present in the data, rostral -> caudal.

    Single-linkage over sorted values with a fixed gap: values within
    `tolerance` of the running group join it, otherwise a new group starts.
    Deliberately not k-means — there is no k to choose, and the whole point is
    to let the data say how many levels there are.
    """
    vals = sorted(float(v) for v in ap_values if v is not None and v == v)
    if not vals:
        return []

    groups: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])

    # vals are negative and sorted ascending, i.e. caudal first. Rostral is the
    # least negative, so reverse to make AP1 the most rostral level.
    groups.reverse()
    return [Bin(index=i + 1, label=f"AP{i + 1}",
                centre_mm=round(float(np.median(g)), 2), n_sections=len(g))
            for i, g in enumerate(groups)]


def assign_bin(ap_mm: float | None, bins: list[Bin]) -> str | None:
    """Nearest bin for a bregma value, or None if it has no AP."""
    if ap_mm is None or not bins:
        return None
    return min(bins, key=lambda b: abs(b.centre_mm - float(ap_mm))).label


def read_ap_manifest(path: Path, wt_mice: set[str] | None = None) -> list[dict]:
    """Rows with a real bregma value, optionally restricted to WT animals.

    `wt_mice` must be supplied explicitly. There is no inference from cohort
    name: see the module docstring.
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            raw = (r.get("ap_mm") or "").strip()
            if not raw:
                continue
            if wt_mice is not None and r.get("mouse") not in wt_mice:
                continue
            try:
                r["_ap"] = float(raw)
            except ValueError:
                continue
            rows.append(r)
    return rows


def read_wt_mice(path: Path) -> set[str]:
    """Load a mouse -> condition table and return the WT animals.

    Expects columns `mouse` and `condition`; anything whose condition starts
    with 'wt' or equals 'control' counts. Raises if the file names no WT animals
    at all, because silently returning an empty set would filter the entire
    dataset away and look like a data problem rather than a config one.
    """
    wt = set()
    seen = 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            seen += 1
            cond = (r.get("condition") or "").strip().lower()
            if cond.startswith("wt") or cond in {"control", "wildtype", "wild-type"}:
                wt.add((r.get("mouse") or "").strip())
    if not wt:
        raise ValueError(
            f"{path} lists {seen} animals but none with condition wt/control. "
            f"Expected columns: mouse,condition")
    return wt


def coverage(all_rows, ap_rows, wt_mice: set[str] | None = None) -> dict:
    """How much of the dataset is usable for an AP model, and what is missing."""
    have = {r["image_name"] for r in ap_rows}
    total = len(all_rows)
    mice_all = {r["mouse"] for r in all_rows}

    missing = [r for r in all_rows if r["image_name"] not in have]
    by_cohort: dict[str, list[int]] = {}
    for r in all_rows:
        c = by_cohort.setdefault(r.get("source", "?"), [0, 0])
        c[0] += 1
        c[1] += r["image_name"] in have

    out = {
        "n_sections": total,
        "n_with_ap": len(have),
        "n_without_ap": total - len(have),
        "n_mice": len(mice_all),
        "by_cohort": {k: {"total": v[0], "with_ap": v[1]} for k, v in sorted(by_cohort.items())},
        "mice_without_any_ap": sorted({r["mouse"] for r in missing}
                                      - {r["mouse"] for r in ap_rows}),
    }
    if wt_mice is not None:
        out["n_wt_mice"] = len(wt_mice & mice_all)
        out["n_wt_sections_with_ap"] = sum(1 for r in ap_rows if r["mouse"] in wt_mice)
        out["wt_mice_not_in_dataset"] = sorted(wt_mice - mice_all)
    return out


# --------------------------------------------------------------------------- #
# agreement on an ordinal scale
# --------------------------------------------------------------------------- #
def _weighted_kappa(a: list[int], b: list[int], n_bins: int) -> float:
    """Quadratic-weighted Cohen's kappa.

    Quadratic weights because the scale is ordinal: calling AP3 when the answer
    is AP4 is a near miss, calling AP7 is not, and unweighted kappa scores both
    as simply wrong. Plain accuracy has the same blindness.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    obs = np.zeros((n_bins, n_bins))
    for x, y in zip(a, b, strict=True):
        obs[x, y] += 1
    obs /= n
    pa, pb = obs.sum(axis=1), obs.sum(axis=0)
    exp = np.outer(pa, pb)

    idx = np.arange(n_bins)
    w = (idx[:, None] - idx[None, :]) ** 2 / max(1, (n_bins - 1) ** 2)
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom else float("nan")


def ap_agreement(sheets: dict[str, dict[str, str]], bins: list[Bin],
                 reference: dict[str, str] | None = None) -> dict:
    """Agreement between scorers on AP bin.

    sheets    {rater: {scoring_id: "AP3"}}
    reference {scoring_id: "AP3"} from Csilla's bregma values — the closest
              thing to truth here, since it comes from the sectioning record
              rather than from looking at the image.
    """
    raters = dict(sheets)
    if reference:
        raters["reference"] = reference

    order = {b.label: b.index - 1 for b in bins}
    ids = sorted(set.intersection(*(set(v) for v in raters.values())))
    ids = [i for i in ids if all(raters[r].get(i) in order for r in raters)]
    if not ids:
        raise ValueError("no sections with a valid AP bin from every rater")

    names = sorted(raters)
    cols = {r: [order[raters[r][i]] for i in ids] for r in names}

    pairs = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = cols[names[x]], cols[names[y]]
            d = np.array(a) - np.array(b)
            pairs[f"{names[x]} vs {names[y]}"] = {
                "weighted_kappa": round(_weighted_kappa(a, b, len(bins)), 3),
                "exact_match_pct": round(100 * float((d == 0).mean()), 1),
                "within_one_bin_pct": round(100 * float((np.abs(d) <= 1).mean()), 1),
                "mean_signed_bins": round(float(d.mean()), 2),
            }

    m = np.array([cols[r] for r in names])          # raters x sections
    spread = m.max(axis=0) - m.min(axis=0)
    return {
        "n_sections": len(ids),
        "n_bins": len(bins),
        "raters": names,
        "pairwise": pairs,
        "mean_spread_bins": round(float(spread.mean()), 2),
        "sections_with_full_agreement_pct": round(100 * float((spread == 0).mean()), 1),
        "worst_sections": [ids[i] for i in np.argsort(-spread)[:10]],
    }
