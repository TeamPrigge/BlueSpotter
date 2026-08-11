"""Load a BlueSpotter dataset from a manifest CSV (links, not copies).

The referenced image/mask files already live in the mounted Google Drive
(TeamPrigge/Data/NM_Slices/...), so we read them straight off the mount by path
- no download, no auth, no storage duplication. If a file can't be found on the
mount we fall back to downloading it by Drive file-ID via the API.

Two mask conventions are supported:
  * cp_masks_png : image is a .tif, mask is a Cellpose `_cp_masks.png`
  * seg_npy      : a Cellpose `_seg.npy`, which may or may not contain the image

That "may or may not" is not a hedge — it is two generations of Cellpose, and
getting it wrong cost us most of the training set once already. Older versions
embedded the raw image under the key `img`. Newer ones dropped it to keep the
files small and record only `filename`, a path on whoever ran the annotation.
Roughly 680 of the 1,386 training rows are the newer kind.

The old loader did `d["img"]`, caught the KeyError, and printed `[skip]`. So the
pipeline reported 1,386 training images and trained on about 700, with nothing
anywhere saying so: the DVC index counted the files as *present*, because they
are — it never checked they were *loadable*. `load_manifest` now raises if the
loss is large rather than continuing quietly, because a training set that halves
without telling you is worse than one that fails.
"""
from __future__ import annotations

import contextlib
import csv
from pathlib import Path

import numpy as np

# Source cohort -> folder layout under the NM_Slices root.
_ED_COHORT = "NM_hightiter_histology_brains_ED"
_LOW_COHORT = "NM_lowtiter_histology_brains"

_IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Colab's FUSE Drive mount drops under sustained reads of large files — it comes
# back as OSError 107, "Transport endpoint is not connected", and every
# subsequent path fails until it is remounted. Our .npy slices are 200-500 MB
# each, so a run that touches a few dozen of them will hit this. Remounting and
# retrying the file recovers it; without this a single blip discards the rest of
# the dataset and the run reports a number that describes whatever it managed to
# read before the mount died.
_MOUNT_ERRNOS = {107, 5, 103}   # not connected, I/O error, connection aborted


def _remount_drive() -> bool:
    """Best effort remount of the Colab Drive mount. False if not in Colab."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return False
    with contextlib.suppress(Exception):   # already broken is fine
        drive.flush_and_unmount()
    try:
        drive.mount("/content/drive", force_remount=True)
        return True
    except Exception:
        return False


def _is_mount_failure(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno in _MOUNT_ERRNOS

# name -> path index per cohort, built once on demand. Walking a Drive mount is
# slow, so we pay for it at most once per cohort per process.
_NAME_INDEX: dict[Path, dict[str, Path]] = {}


def _cohort_index(root: Path) -> dict[str, Path]:
    """Map lowercased file name -> path for every image under `root`."""
    if root not in _NAME_INDEX:
        index: dict[str, Path] = {}
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    index.setdefault(p.name.lower(), p)
        _NAME_INDEX[root] = index
    return _NAME_INDEX[root]


def find_image_for_seg(npy_path: Path, blob: dict, nm_root: Path) -> Path | None:
    """Locate the image belonging to a `_seg.npy` that does not embed one.

    Tried in order of how much we trust them:

      1. `<stem>.tif` (or .png/...) sitting next to the .npy — the usual layout,
         and unambiguous.
      2. The basename recorded in the blob's `filename` key, looked up next to
         the .npy. The stored path itself is an absolute path on the annotator's
         own machine and will not resolve here, but the file name survives.
      3. The same basename anywhere under the cohort. Cohorts split images and
         masks across sibling folders (masks/ vs processed/cropped/), so this
         catches the common case at the cost of one directory walk.

    Returns None rather than guessing if nothing matches.
    """
    stem = npy_path.name
    for suffix in ("_seg.npy", " .seg.npy", "_seg_2.npy"):
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = npy_path.stem

    for ext in _IMAGE_EXTS:
        cand = npy_path.parent / f"{stem}{ext}"
        if cand.exists():
            return cand

    recorded = str(blob.get("filename") or "")
    names = [Path(recorded).name] if recorded else []
    names += [f"{stem}{ext}" for ext in _IMAGE_EXTS]

    for name in names:
        if not name:
            continue
        cand = npy_path.parent / name
        if cand.exists():
            return cand

    # Fall back to a cohort-wide lookup: everything under the first directory
    # below nm_root that contains this file.
    try:
        cohort = nm_root / npy_path.relative_to(nm_root).parts[0]
    except (ValueError, IndexError):
        return None
    index = _cohort_index(cohort)
    for name in names:
        hit = index.get(name.lower())
        if hit is not None:
            return hit
    return None


def load_pair(nm_root: Path, row: dict, _retry: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (image, mask) for one manifest row. Raises if either is unusable.

    The single place that knows how to turn a row into pixels — training,
    evaluation and the assertion builder all go through here, so a format
    surprise like the missing `img` key can only ever be fixed once.

    Retries once through a Drive remount if the mount drops mid-read.
    """
    nm_root = Path(nm_root)
    try:
        return _load_pair_inner(nm_root, row)
    except OSError as exc:
        if not (_retry and _is_mount_failure(exc)):
            raise
        print(f"    [drive] mount lost ({exc.errno}) — remounting and retrying "
              f"{row.get('image_name', '?')}")
        _NAME_INDEX.clear()          # the cached paths are stale after a remount
        if not _remount_drive():
            raise
        return load_pair(nm_root, row, _retry=False)


def _load_pair_inner(nm_root: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    ipath, mpath, is_npy = resolve_row(nm_root, row)

    if not is_npy:
        if not ipath.exists():
            raise FileNotFoundError(ipath)
        if not mpath.exists():
            raise FileNotFoundError(mpath)
        return np.asarray(_imread(ipath)), np.asarray(_imread(mpath)).astype(np.int32)

    if not ipath.exists():
        raise FileNotFoundError(ipath)
    blob = np.load(ipath, allow_pickle=True).item()
    if "masks" not in blob:
        raise KeyError(f"{ipath.name} has no 'masks' key (keys: {sorted(blob)})")
    mask = np.asarray(blob["masks"]).astype(np.int32)

    if "img" in blob:
        return np.asarray(blob["img"]), mask

    found = find_image_for_seg(ipath, blob, nm_root)
    if found is None:
        raise FileNotFoundError(
            f"{ipath.name} stores no image (Cellpose >=3 drops it) and no matching "
            f"image file was found next to it or under the cohort. "
            f"Recorded filename was {blob.get('filename', '<none>')!r}."
        )
    return np.asarray(_imread(found)), mask


def resolve_row(nm_root: Path, row: dict):
    """Return (image_path, mask_path, is_npy) on the mounted Drive for a row.

    This is the single source of truth for manifest-row -> Drive-path mapping.
    Both the training loader and the DVC indexing stage (`drive_index.py`) call
    it, so a layout change only ever has to be made here.

    Two generations of manifest are supported. Rows written by
    `bluespotter.discover` carry an explicit `rel_path` (and a relative path in
    `mask_id`), which we trust directly — that is the only form that scales, as
    the labelled cohorts live in a dozen differently-shaped folder layouts.
    Older hand-authored rows have no `rel_path`, so we fall back to the two
    hard-coded cohort layouts below.
    """
    is_npy = str(row.get("mask_type", "")).startswith("seg_npy")

    rel_img = (row.get("rel_path") or "").strip()
    if rel_img:
        img = nm_root / rel_img
        # `mask_id` holds either a Drive file-ID (legacy) or a relative path
        # (discover). A Drive ID never contains a separator, so that is the test.
        rel_msk = (row.get("mask_id") or "").strip()
        msk = nm_root / rel_msk if "/" in rel_msk else img
        return img, (img if is_npy else msk), is_npy

    ch = row.get("channel", "TH")
    iname, mname = row["image_name"], row["mask_name"]
    if row["mask_type"].startswith("seg_npy"):
        p = nm_root / _LOW_COHORT / "masks" / "npy_masks" / iname
        return p, p, True
    # cp_masks_png (ED cohort): image in processed/cropped/<ch>, mask in masks/<ch>
    img = nm_root / _ED_COHORT / "processed" / "cropped" / ch / iname
    msk = nm_root / _ED_COHORT / "masks" / ch / mname
    return img, msk, False


# Backwards-compatible private alias (older call sites).
_resolve = resolve_row


def _imread(path: Path):
    from skimage.io import imread
    return imread(str(path))


def load_manifest(csv_path, nm_root, cache_dir=None, service=None):
    """Return (images, labels) lists ready for cellpose.train.train_seg.

    csv_path : mounted-Drive path to train.csv / test.csv
    nm_root  : mounted-Drive path to .../Data/NM_Slices
    """
    csv_path, nm_root = Path(csv_path), Path(nm_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  Manifest: {csv_path.name}  ({len(rows)} rows)  reading from {nm_root}")

    images, labels = [], []
    failures: list[str] = []
    for i, r in enumerate(rows, 1):
        try:
            img, msk = load_pair(nm_root, r)
            images.append(img)
            labels.append(msk)
        except Exception as e:
            failures.append(f"{r.get('image_name', '?')}: {type(e).__name__} {e}")
        if i % 20 == 0:
            print(f"    ...processed {i}/{len(rows)}")

    n_bad = len(failures)
    print(f"  Loaded {len(images)}/{len(rows)} pairs from {csv_path.name}"
          + (f"  ({n_bad} unreadable)" if n_bad else ""))
    for line in failures[:10]:
        print(f"    [skip] {line}")
    if n_bad > 10:
        print(f"    ... and {n_bad - 10} more")

    if not images:
        raise RuntimeError(
            f"No pairs loaded from {csv_path.name}. Check data.nmslices_root in params.yaml "
            f"(currently {nm_root}) points at the mounted NM_Slices folder.")

    # Refuse to train on a silently halved dataset. This exact failure — newer
    # Cellpose .npy files not embedding the image — cost ~680 of 1,386 training
    # rows while every report still said 1,386. A run that quietly drops half its
    # data is not a run you can compare against anything.
    if n_bad > max(5, 0.05 * len(rows)):
        raise RuntimeError(
            f"{n_bad}/{len(rows)} rows in {csv_path.name} could not be loaded "
            f"({n_bad / len(rows):.0%}). Refusing to train on a dataset this much "
            f"smaller than the manifest claims — the metrics would describe "
            f"something other than the recorded dataset_hash. See the skips above."
        )
    return images, labels
