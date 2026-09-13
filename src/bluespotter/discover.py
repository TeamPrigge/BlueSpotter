"""Discover every labelled LC image on the Drive mount and write the manifests.

WHY THIS EXISTS
---------------
Until now train.csv / test.csv were hand-authored on Drive, which capped the
training set at whatever someone had time to type out (68 rows) while several
hundred more labelled slices sat in cohort folders nobody had transcribed. A
hand-maintained index of a growing dataset is a losing game: it goes stale the
moment a colleague segments another brain.

So this module makes the manifests *derived* rather than authored. It walks the
mounted NM_Slices tree, finds every image/label pair, reads the metadata out of
the file names (see `naming.py`), and writes:

    train.csv         segmentation training split
    test.csv          held-out split, grouped by animal
    ap_position.csv   the subset whose names carry a real bregma coordinate

`ap_position.csv` is deliberately a separate file. Those rows are the seed for a
second model — predicting rostrocaudal position from LC shape — and that model
needs a different split (by animal *and* by AP band) and different validation
than segmentation does. Keeping it apart means neither pipeline can quietly
contaminate the other, and the AP rows can still appear in train.csv for
segmentation, where their AP label is simply unused.

The split is grouped by mouse. Two hemispheres of one section, or two sections
from one animal, share biology, staining batch and imaging session; splitting
them across train/test measures memorisation of that animal, not generalisation
to a new one. Assignment is a deterministic hash of the mouse ID, so the split
is stable across runs and machines without storing a seed file.

Run:

    python -m bluespotter.discover               # write manifests to Drive
    python -m bluespotter.discover --dry-run     # report only, touch nothing
    python -m bluespotter.discover --out-dir X   # write somewhere else
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .naming import Parsed, parse, strip_role_suffix

# Columns of train.csv / test.csv. The first twelve are the historical schema
# that validate.py enforces; `rel_path` is new.
MANIFEST_FIELDS = (
    "split", "source", "microscope", "mouse", "slice", "channel", "side",
    "image_name", "image_id", "mask_name", "mask_id", "mask_type",
    "rel_path",
)

# ap_position.csv carries everything the manifest does plus the coordinate and a
# clickable link, because this file gets eyeballed by humans checking whether a
# slice really sits where its name claims.
AP_FIELDS = (
    *MANIFEST_FIELDS,
    "ap_mm", "ap_index", "ap_source", "drive_url",
)

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
_DRIVE_FILE_URL = "https://drive.google.com/file/d/{id}/view"

# Folders that hold working copies rather than the canonical export. When the
# same (mouse, channel, side, ap) turns up in several places we keep the copy
# from the folder with the lowest rank.
_PREFERENCE = ("masks", "npy", "cropped_bilat", "cropped2", "processed")


@dataclass
class Pair:
    """One image + label pair, wherever it lives on the mount."""

    image: Path
    mask: Path
    meta: Parsed
    mask_type: str        # cp_masks_png | seg_npy
    cohort: str           # top-level folder under nmslices_root
    rel_dir: str          # directory relative to nmslices_root

    @property
    def key(self) -> tuple:
        """Biological identity — used to collapse duplicate copies."""
        m = self.meta
        return (m.mouse, m.channel, m.side, m.ap_mm, m.ap_index, m.slice, self.cohort)

    @property
    def rank(self) -> int:
        parts = {p.lower() for p in Path(self.rel_dir).parts}
        for i, pref in enumerate(_PREFERENCE):
            if pref in parts:
                return i
        return len(_PREFERENCE)


# --------------------------------------------------------------------------- #
# walking
# --------------------------------------------------------------------------- #
def _microscope(cohort: str, mask_type: str) -> str:
    """Best-effort acquisition label; the manifests already record it as free text."""
    if mask_type == "seg_npy" and "lowtiter_histology" in cohort:
        return "Olympus (vsi->seg.npy)"
    if "tyr_titering" in cohort or "hightiter" in cohort:
        return "Zeiss (czi->tif)"
    if "slidescanner" in cohort:
        return "Slidescanner (ome.tif)"
    return "unknown"


def walk(nm_root: Path) -> tuple[list[Pair], list[Path]]:
    """Return (pairs, unparsed) for everything labelled under `nm_root`."""
    nm_root = Path(nm_root)
    if not nm_root.is_dir():
        raise FileNotFoundError(
            f"NM_Slices root not found: {nm_root}\n"
            "Mount Drive first, and check data.nmslices_root in params.yaml."
        )

    # Index candidate images by stem per directory so masks can find their image.
    images_by_dir: dict[Path, dict[str, Path]] = defaultdict(dict)
    mask_files: list[Path] = []
    npy_files: list[Path] = []

    for path in nm_root.rglob("*"):
        if not path.is_file():
            continue
        low = path.name.lower()
        if low.endswith("_seg.npy") or low.endswith(" .seg.npy") or low.endswith("_seg_2.npy"):
            npy_files.append(path)
        elif "_cp_masks" in low and low.endswith(".png"):
            mask_files.append(path)
        elif low.endswith(IMAGE_EXTS):
            stem, role = strip_role_suffix(path.name)
            if role == "image":
                images_by_dir[path.parent].setdefault(stem, path)

    pairs: list[Pair] = []
    unparsed: list[Path] = []

    def _cohort_of(p: Path) -> tuple[str, str]:
        rel = p.parent.relative_to(nm_root)
        parts = rel.parts
        return (parts[0] if parts else ""), str(rel)

    # seg.npy holds image and mask in one file.
    for p in npy_files:
        meta = parse(p.name)
        if meta.kind == "unknown":
            unparsed.append(p)
            continue
        cohort, rel_dir = _cohort_of(p)
        pairs.append(Pair(p, p, meta, "seg_npy", cohort, rel_dir))

    # cp_masks.png needs its sibling image.
    for p in mask_files:
        stem, _ = strip_role_suffix(p.name)
        img = images_by_dir.get(p.parent, {}).get(stem)
        if img is None:
            # Masks and images are sometimes split into masks/<ch> and
            # processed/cropped/<ch>; search the cohort for the stem.
            cohort, _ = _cohort_of(p)
            for cand_dir, stems in images_by_dir.items():
                try:
                    if cand_dir.relative_to(nm_root).parts[:1] == (cohort,) and stem in stems:
                        img = stems[stem]
                        break
                except ValueError:
                    continue
        if img is None:
            unparsed.append(p)
            continue
        meta = parse(p.name)
        if meta.kind == "unknown":
            unparsed.append(p)
            continue
        cohort, rel_dir = _cohort_of(img)
        pairs.append(Pair(img, p, meta, "cp_masks_png", cohort, rel_dir))

    return pairs, unparsed


def _one_per_dir(paths: list[Path], nm_root: Path, limit: int = 30) -> list[str]:
    """One representative path per directory, so distinct conventions all show."""
    seen: dict[Path, str] = {}
    for p in paths:
        seen.setdefault(p.parent, str(p.relative_to(nm_root)))
    return list(seen.values())[:limit]


def dedupe(pairs: Iterable[Pair]) -> tuple[list[Pair], int]:
    """Collapse copies of the same slice, preferring the canonical folder."""
    best: dict[tuple, Pair] = {}
    dropped = 0
    for p in pairs:
        cur = best.get(p.key)
        if cur is None:
            best[p.key] = p
        else:
            dropped += 1
            if p.rank < cur.rank:
                best[p.key] = p
    return list(best.values()), dropped


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def assign_splits(pairs: list[Pair], test_fraction: float,
                  salt: str = "bluespotter") -> dict[str, str]:
    """Map mouse -> 'train'|'test', grouped by animal and deterministic.

    Hashing the mouse ID rather than shuffling means the same animal lands in the
    same split on every machine and every rerun, with no seed file to lose. We
    then walk animals in hash order and stop assigning to test once the target
    fraction of *images* (not animals) is met, so the split is balanced by data
    volume even though large and small animals are mixed.
    """
    per_mouse: dict[str, int] = Counter(p.meta.mouse for p in pairs)
    total = sum(per_mouse.values())
    target = round(total * test_fraction)

    def h(mouse: str) -> str:
        return hashlib.md5(f"{salt}:{mouse}".encode()).hexdigest()  # split choice only

    order = sorted(per_mouse, key=h)
    assignment: dict[str, str] = {}
    acc = 0
    for mouse in order:
        n = per_mouse[mouse]
        # Take the animal only if doing so lands closer to the target than
        # stopping here. A plain `while acc < target` overshoots badly when
        # animals are large: with 14 images each and a target of 20 it would
        # take two animals and hand back a 21% test set.
        if acc < target and abs(acc + n - target) <= abs(acc - target):
            assignment[mouse] = "test"
            acc += n
        else:
            assignment[mouse] = "train"

    # A test split with no animals in it is useless; take the smallest one.
    if acc == 0 and order:
        smallest = min(order, key=lambda m: (per_mouse[m], h(m)))
        assignment[smallest] = "test"
    return assignment


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _row(p: Pair, split: str, nm_root: Path) -> dict[str, Any]:
    rel_img = str(p.image.relative_to(nm_root))
    rel_msk = str(p.mask.relative_to(nm_root))
    return {
        "split": split,
        "source": p.cohort,
        "microscope": _microscope(p.cohort, p.mask_type),
        "mouse": p.meta.mouse,
        "slice": p.meta.slice,
        "channel": p.meta.channel,
        "side": p.meta.side,
        "image_name": p.image.name,
        # IDs are filled in by the Drive API when available; the mount gives us
        # paths, and `rel_path` is what actually resolves the file. We put the
        # relative path in the ID column too so validate.py's "no blank ID" rule
        # still means something: every row can be traced to a unique object.
        "image_id": rel_img,
        "mask_name": p.mask.name,
        "mask_id": rel_msk,
        "mask_type": p.mask_type,
        "rel_path": rel_img,
    }


def _ap_row(p: Pair, split: str, nm_root: Path) -> dict[str, Any]:
    row = _row(p, split, nm_root)
    row.update(
        ap_mm=f"{p.meta.ap_mm:.2f}" if p.meta.ap_mm is not None else "",
        ap_index="" if p.meta.ap_index is None else str(p.meta.ap_index),
        ap_source="filename",
        drive_url="",
    )
    return row


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def build(nm_root: Path, out_dir: Path | None = None, test_fraction: float = 0.15,
          dry_run: bool = False, out_paths: dict[str, Path] | None = None,
          ) -> dict[str, Any]:
    """Discover, split and write the three manifests. Returns a summary dict.

    `out_paths` gives an explicit destination per manifest and takes precedence
    over `out_dir`. That matters because the three files do *not* live in one
    folder on Drive — train.csv is under data/training_data/ and test.csv under
    data/test_data/. Writing all three next to train.csv silently orphaned the
    new test.csv while `sync-manifests` kept reading the stale one, so train was
    rebuilt and test was not. Passing the configured paths through removes the
    chance of that happening again.
    """
    nm_root = Path(nm_root)
    pairs, unparsed = walk(nm_root)
    pairs, duplicates = dedupe(pairs)

    # Deterministic order so the CSV hash depends on content, not on the order
    # the filesystem happened to hand us directory entries.
    pairs.sort(key=lambda p: (p.cohort, p.meta.mouse, p.meta.channel,
                              p.meta.ap_mm if p.meta.ap_mm is not None else 0.0,
                              p.meta.slice, p.meta.side, p.image.name))

    assignment = assign_splits(pairs, test_fraction)

    train_rows, test_rows, ap_rows = [], [], []
    for p in pairs:
        split = assignment[p.meta.mouse]
        row = _row(p, split, nm_root)
        (test_rows if split == "test" else train_rows).append(row)
        if p.meta.has_ap:
            ap_rows.append(_ap_row(p, split, nm_root))

    summary = {
        "nmslices_root": str(nm_root),
        "pairs_found": len(pairs) + duplicates,
        "pairs_kept": len(pairs),
        "duplicates_collapsed": duplicates,
        "unparsed_files": len(unparsed),
        # One example per directory, not the first N paths. Unparsed files come
        # in families — a whole folder shares a naming convention — so a flat
        # head() shows the same mistake ten times and hides the other nine.
        "unparsed_examples": _one_per_dir(unparsed, nm_root),
        "unparsed_by_dir": dict(sorted(
            Counter(str(p.parent.relative_to(nm_root)) for p in unparsed).items(),
            key=lambda kv: -kv[1])),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "ap_rows": len(ap_rows),
        "mice_total": len(assignment),
        "mice_test": sum(1 for v in assignment.values() if v == "test"),
        "by_cohort": dict(sorted(Counter(p.cohort for p in pairs).items())),
        "by_mask_type": dict(sorted(Counter(p.mask_type for p in pairs).items())),
        "by_channel": dict(sorted(Counter(p.meta.channel for p in pairs).items())),
        "ap_mm_range": (
            [min(r["ap_mm"] for r in ap_rows), max(r["ap_mm"] for r in ap_rows)]
            if ap_rows else []
        ),
        "dry_run": dry_run,
    }

    if out_paths is None:
        if out_dir is None:
            raise ValueError("pass either out_dir or out_paths")
        d = Path(out_dir)
        out_paths = {"train": d / "train.csv", "test": d / "test.csv",
                     "ap": d / "ap_position.csv"}
    summary["destinations"] = {k: str(v) for k, v in out_paths.items()}

    if not dry_run:
        _write(out_paths["train"], MANIFEST_FIELDS, train_rows)
        _write(out_paths["test"], MANIFEST_FIELDS, test_rows)
        _write(out_paths["ap"], AP_FIELDS, ap_rows)
        summary["written"] = [str(p) for p in out_paths.values()]

    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=None)
    ap.add_argument("--nm-root", default=None,
                    help="override data.nmslices_root (useful off Colab)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the manifests (default: the Drive authoring dir)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="run even when dvc.discover is false in params.yaml")
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.params)
    repo = cfg.repo_root
    prov = repo / cfg.dvc.get("report_dir", "reports") / "discovery.json"
    prov.parent.mkdir(parents=True, exist_ok=True)

    # Gating lives here rather than in a shell one-liner in dvc.yaml, so the
    # skip is visible in the report instead of hidden behind an `||`.
    if not (cfg.dvc.get("discover") or args.force or args.nm_root):
        print("[discover] skipped (dvc.discover=false in params.yaml). "
              "Set it true on a machine with the Drive mount, or pass --force.")
        prov.write_text(json.dumps({"status": "skipped",
                                    "reason": "dvc.discover=false"}, indent=2) + "\n")
        return 0

    nm_root = Path(args.nm_root) if args.nm_root else cfg.nmslices_root

    # Each manifest goes where params.yaml says it lives — they are in three
    # different Drive folders, and guessing one folder for all three is what
    # left a stale test.csv in place on the first real run.
    if args.out_dir:
        d = Path(args.out_dir)
        out_paths = {"train": d / "train.csv", "test": d / "test.csv",
                     "ap": d / "ap_position.csv"}
    else:
        out_paths = {
            "train": cfg.train_manifest,
            "test": cfg.test_manifest,
            "ap": cfg.drive_root / cfg.data["ap_manifest"],
        }

    print(f"[discover] walking {nm_root}")
    summary = build(
        nm_root=nm_root,
        out_paths=out_paths,
        test_fraction=float(cfg.data.get("test_split", 0.15)),
        dry_run=args.dry_run,
    )

    print(f"  pairs      : {summary['pairs_kept']} kept "
          f"({summary['duplicates_collapsed']} duplicate copies collapsed)")
    print(f"  splits     : {summary['train_rows']} train / {summary['test_rows']} test "
          f"across {summary['mice_total']} mice")
    print(f"  AP subset  : {summary['ap_rows']} rows with a bregma coordinate")
    if summary["unparsed_files"]:
        print(f"  WARNING: {summary['unparsed_files']} labelled file(s) whose names "
              f"could not be parsed — they are NOT in the manifests:")
        for p in summary["unparsed_examples"]:
            print(f"    - {p}")

    prov.write_text(json.dumps({"status": "ok", **summary}, indent=2) + "\n")
    print(f"  summary -> {prov.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
