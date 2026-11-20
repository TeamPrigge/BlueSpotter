"""Structural validation of the BlueSpotter manifests, run as a DVC stage.

This is the gate between "a CSV exists" and "a dataset you can trust". It checks
things that are cheap to verify and expensive to discover late:

  1. Schema      — required columns present, no blank image/mask names or IDs.
  2. Duplicates  — the same file referenced twice within a split.
  3. Overlap     — the same file appearing in BOTH train and test.
  4. Group leak  — different images from the same *mouse* (and the same physical
                   *slice*) split across train and test.
  5. Mask types  — only known conventions (`cp_masks_png`, `seg_npy`).
  6. Balance     — how the splits distribute over cohort, microscope, channel.

Check 4 deserves a word, because it is the one that quietly inflates results.
Two hemispheres of one slice, or two slices from one animal, share the animal's
biology, the staining batch, and the imaging session. Putting one in train and
the other in test means the test score partly measures memorisation of that
animal rather than generalisation to a new one. The honest unit of splitting for
this kind of histology is the *mouse*. This stage reports the leak rather than
silently rejecting it — set `dvc.fail_on_group_leak: true` in params.yaml to make
it a hard failure once you have re-split.

Run:

    python -m bluespotter.validate
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import Config, load_config

REQUIRED_COLUMNS = {
    "split", "source", "microscope", "mouse", "slice", "channel", "side",
    "image_name", "image_id", "mask_name", "mask_id", "mask_type",
}

KNOWN_MASK_TYPES = ("cp_masks_png", "seg_npy")


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _slice_key(r: dict[str, str]) -> str:
    """Identify a physical section: mouse + slice (ignoring hemisphere)."""
    return f"{r.get('mouse','')}|{r.get('slice','')}"


def _counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(r.get(field, "") for r in rows).items()))


def validate_ap(ap_csv: Path, train_csv: Path, test_csv: Path) -> dict[str, Any]:
    """Checks specific to the AP-position manifest.

    This file is the seed for a second model — predicting rostrocaudal position
    from LC shape — so its failure modes are different from segmentation's. What
    matters here is that the coordinate is real and that the *AP* split does not
    leak animals, which it would silently inherit if it just tagged along with
    the segmentation split.
    """
    if not ap_csv.exists():
        return {"present": False, "n_rows": 0, "errors": [], "warnings": []}

    rows = _read(ap_csv)
    errors: list[str] = []
    warnings: list[str] = []
    values: list[float] = []

    for i, r in enumerate(rows, 2):
        raw = (r.get("ap_mm") or "").strip()
        if not raw:
            errors.append(f"ap_position line {i}: blank ap_mm — the whole point of "
                          f"this manifest is the coordinate")
            continue
        try:
            v = float(raw)
        except ValueError:
            errors.append(f"ap_position line {i}: ap_mm {raw!r} is not a number")
            continue
        # Mouse LC spans roughly -5.0 to -5.9 mm from bregma. A positive value or
        # one far outside that window means the filename was misread, not that
        # the slice is exotic.
        if not (-6.5 <= v <= -4.5):
            errors.append(f"ap_position line {i}: ap_mm {v} is outside the LC range "
                          f"(-6.5..-4.5 mm) — likely a misparsed file name")
        values.append(v)

    # Every AP row must also be a real segmentation row, otherwise the two
    # datasets have drifted apart and the AP file is citing files nobody indexes.
    # Match on rel_path, not image_id: `drive_links` rewrites image_id from a
    # path to a Drive file-ID in this manifest only, so comparing IDs would
    # report every row as an orphan the moment someone adds the links.
    def _identity(r: dict[str, str]) -> str:
        return (r.get("rel_path") or r.get("image_id") or "").strip()

    seg_ids = {_identity(r) for r in _read(train_csv)} | \
              {_identity(r) for r in _read(test_csv)}
    orphans = [r.get("image_name", "?") for r in rows
               if _identity(r) and _identity(r) not in seg_ids]
    if orphans:
        warnings.append(f"{len(orphans)} AP row(s) reference files that are in neither "
                        f"train.csv nor test.csv (e.g. {', '.join(orphans[:3])})")

    ap_tr = {r.get("mouse", "") for r in rows if r.get("split") == "train"}
    ap_te = {r.get("mouse", "") for r in rows if r.get("split") == "test"}
    ap_leak = sorted(m for m in (ap_tr & ap_te) if m)
    if ap_leak:
        errors.append(f"AP manifest: {len(ap_leak)} mouse/mice in both splits "
                      f"({', '.join(ap_leak)}) — an AP model would be scored on "
                      f"animals it trained on")

    return {
        "present": True,
        "n_rows": len(rows),
        "n_mice": len({r.get("mouse", "") for r in rows if r.get("mouse")}),
        "ap_mm_min": round(min(values), 2) if values else None,
        "ap_mm_max": round(max(values), 2) if values else None,
        "n_ap_mouse_overlap": len(ap_leak),
        "errors": errors,
        "warnings": warnings,
    }


def validate(train_csv: Path, test_csv: Path, ap_csv: Path | None = None) -> dict[str, Any]:
    train, test = _read(train_csv), _read(test_csv)
    errors: list[str] = []
    warnings: list[str] = []

    # --- 1. schema -------------------------------------------------------- #
    for name, rows in (("train", train), ("test", test)):
        if not rows:
            errors.append(f"{name}: manifest is empty")
            continue
        missing_cols = REQUIRED_COLUMNS - set(rows[0].keys())
        if missing_cols:
            errors.append(f"{name}: missing columns {sorted(missing_cols)}")
        for i, r in enumerate(rows, 2):  # 2 = first data row in a 1-indexed file
            for col in ("image_name", "image_id", "mask_name", "mask_id"):
                if not (r.get(col) or "").strip():
                    errors.append(f"{name} line {i}: blank {col}")
            if not str(r.get("mask_type", "")).startswith(KNOWN_MASK_TYPES):
                errors.append(f"{name} line {i}: unknown mask_type "
                              f"{r.get('mask_type')!r}")
            declared = (r.get("split") or "").strip()
            if declared and declared != name:
                warnings.append(f"{name} line {i}: split column says {declared!r}")

    # --- 2. duplicates within a split ------------------------------------- #
    for name, rows in (("train", train), ("test", test)):
        dupes = [k for k, n in Counter(r.get("image_id", "") for r in rows).items()
                 if n > 1 and k]
        if dupes:
            errors.append(f"{name}: {len(dupes)} image_id(s) appear more than once")

    # --- 3. exact file overlap between splits ----------------------------- #
    tr_ids = {r.get("image_id", "") for r in train if r.get("image_id")}
    te_ids = {r.get("image_id", "") for r in test if r.get("image_id")}
    id_overlap = sorted(tr_ids & te_ids)
    if id_overlap:
        errors.append(f"{len(id_overlap)} image(s) present in BOTH train and test — "
                      f"direct test-set contamination")

    # --- 4. group leakage -------------------------------------------------- #
    tr_mice = {r.get("mouse", "") for r in train if r.get("mouse")}
    te_mice = {r.get("mouse", "") for r in test if r.get("mouse")}
    mouse_leak = sorted(tr_mice & te_mice)

    tr_slices = {_slice_key(r) for r in train}
    te_slices = {_slice_key(r) for r in test}
    slice_leak = sorted(s for s in (tr_slices & te_slices) if s.strip("|"))

    if mouse_leak:
        warnings.append(
            f"{len(mouse_leak)}/{len(te_mice)} test mice also appear in train "
            f"({', '.join(mouse_leak)}). Test scores will be optimistic: the model "
            f"has seen other sections from these animals. Consider a "
            f"grouped (leave-mice-out) split."
        )
    if slice_leak:
        warnings.append(
            f"{len(slice_leak)} physical slice(s) are split across train and test "
            f"({', '.join(slice_leak)}) — typically left/right hemisphere of the same "
            f"section. This is the strongest form of the leak above."
        )

    # --- 5/6. composition -------------------------------------------------- #
    composition = {
        name: {
            "n_rows": len(rows),
            "n_mice": len({r.get("mouse", "") for r in rows if r.get("mouse")}),
            "by_source": _counts(rows, "source"),
            "by_microscope": _counts(rows, "microscope"),
            "by_channel": _counts(rows, "channel"),
            "by_mask_type": _counts(rows, "mask_type"),
        }
        for name, rows in (("train", train), ("test", test))
    }

    n_tr, n_te = len(train), len(test)
    per_mouse: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "test": 0})
    for r in train:
        per_mouse[r.get("mouse", "")]["train"] += 1
    for r in test:
        per_mouse[r.get("mouse", "")]["test"] += 1

    # Two files, on purpose. `dvc metrics show` flattens nested JSON into one
    # column per leaf, so putting the per-mouse breakdown in the metrics file
    # would produce a table hundreds of columns wide and unreadable on a PR.
    # Metrics stay scalar; the breakdown goes next door.
    ap = validate_ap(ap_csv, train_csv, test_csv) if ap_csv else {"present": False,
                                                                  "n_rows": 0,
                                                                  "errors": [],
                                                                  "warnings": []}
    errors.extend(ap["errors"])
    warnings.extend(ap["warnings"])

    metrics = {
        "ok": not errors,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "n_train": n_tr,
        "n_test": n_te,
        "n_ap": ap["n_rows"],
        "test_fraction": round(n_te / (n_tr + n_te), 4) if (n_tr + n_te) else 0.0,
        "n_train_mice": composition["train"]["n_mice"],
        "n_test_mice": composition["test"]["n_mice"],
        "image_id_overlap": len(id_overlap),
        "n_mouse_overlap": len(mouse_leak),
        "n_slice_overlap": len(slice_leak),
    }
    detail = {
        "errors": errors,
        "warnings": warnings,
        "leakage": {
            "image_id_overlap": sorted(id_overlap),
            "mouse_overlap": mouse_leak,
            "slice_overlap": slice_leak,
        },
        "composition": composition,
        "rows_per_mouse": dict(sorted(per_mouse.items())),
        "ap_position": ap,
    }
    return {**metrics, "_detail": detail}


def _print(report: dict[str, Any]) -> None:
    detail = report["_detail"]
    print(f"  train={report['n_train']} rows ({report['n_train_mice']} mice)   "
          f"test={report['n_test']} rows ({report['n_test_mice']} mice)   "
          f"(test fraction {report['test_fraction']:.1%})")
    print(f"  leakage: {report['image_id_overlap']} identical images, "
          f"{report['n_mouse_overlap']} shared mice, "
          f"{report['n_slice_overlap']} shared slices")
    apr = detail.get("ap_position", {})
    if apr.get("present"):
        span = (f"{apr['ap_mm_min']}..{apr['ap_mm_max']} mm"
                if apr.get("ap_mm_min") is not None else "no coordinates")
        print(f"  AP subset: {apr['n_rows']} rows ({apr.get('n_mice', 0)} mice), "
              f"bregma {span}")
    for e in detail["errors"]:
        print(f"  ERROR   {e}")
    for w in detail["warnings"]:
        print(f"  WARNING {w}")
    if report["ok"] and not detail["warnings"]:
        print("  All checks passed.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate BlueSpotter manifests.")
    ap.add_argument("--params", default=None)
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.params)
    dvc_cfg = cfg.raw.get("dvc", {})
    repo = cfg.repo_root
    mdir = repo / dvc_cfg["manifest_dir"]

    print(f"[validate] manifests in {mdir}")
    report = validate(mdir / "train.csv", mdir / "test.csv", mdir / "ap_position.csv")
    _print(report)

    rdir = repo / dvc_cfg["report_dir"]
    rdir.mkdir(parents=True, exist_ok=True)
    detail = report.pop("_detail")
    (rdir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (rdir / "validation_detail.json").write_text(json.dumps(detail, indent=2) + "\n")
    print(f"  metrics -> {dvc_cfg['report_dir']}/validation.json"
          f"   detail -> {dvc_cfg['report_dir']}/validation_detail.json")

    if detail["errors"]:
        return 1
    leaky = report["n_mouse_overlap"] or report["n_slice_overlap"]
    if leaky and dvc_cfg.get("fail_on_group_leak", False):
        print("[validate] FAILED: group leakage present and dvc.fail_on_group_leak=true")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
