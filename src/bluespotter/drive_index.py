"""Content-index the Google Drive files referenced by a BlueSpotter manifest.

WHY THIS EXISTS
---------------
The manifests (train.csv / test.csv) are lists of *links* — file names plus Drive
file-IDs. They say nothing about the bytes on the other end of those links. If
someone re-exports a .czi, re-runs Cellpose to regenerate a `_cp_masks.png`, or
swaps a file while keeping the name, the manifest is unchanged and git sees
nothing. The dataset silently mutated underneath the experiment.

This module closes that hole. For every row it resolves the image and mask on the
mounted Drive, records `size / mtime / md5`, and writes a deterministic index
CSV. That index is a DVC output, so its hash lands in `dvc.lock` and is committed
to git. A single changed pixel anywhere in the dataset now changes the index
hash, which changes `dvc.lock`, which shows up as a diff on a pull request.

That is what makes "Drive is the source of truth" safe to build on.

HASHING COST
------------
Full md5 over a Drive mount is I/O-bound and can be slow for large stacks, so
results are memoised in a sidecar cache keyed by (path, size, mtime_ns) in the
local scratch dir. Re-runs only hash files that actually changed. Set
`dvc.hash_mode` in params.yaml to trade rigour for speed:

  full        md5 over the whole file          (default; exact)
  partial     md5 over first+last 4 MiB + size (fast; catches re-exports)
  size_mtime  no content read at all           (fastest; catches almost nothing)

Run as a DVC stage:

    python -m bluespotter.drive_index --split train
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .manifest import resolve_row

_PARTIAL_WINDOW = 4 * 1024 * 1024  # 4 MiB from each end
_CHUNK = 1024 * 1024

# Columns carried through from the manifest, in order, so the index is readable
# on its own without joining back to the manifest.
_CARRY = ("split", "source", "microscope", "mouse", "slice", "channel", "side")

INDEX_FIELDS = (
    *_CARRY,
    "role",         # image | mask
    "name",
    "drive_id",
    "rel_path",     # path relative to nmslices_root — stable across machines
    "exists",
    "size_bytes",
    "mtime_utc",
    "content_hash",
    "hash_mode",
)


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
def _md5_full(path: Path) -> str:
    h = hashlib.md5()  # integrity/versioning only, not security
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _md5_partial(path: Path, size: int) -> str:
    """Hash the head and tail of a file plus its length.

    Cheap but strong in practice: image formats put headers/metadata at the front
    and (for TIFF) IFD offsets near the end, so a re-export essentially always
    perturbs one of the two windows or the length.
    """
    h = hashlib.md5()  # integrity only, not security
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(_PARTIAL_WINDOW))
        if size > 2 * _PARTIAL_WINDOW:
            fh.seek(-_PARTIAL_WINDOW, os.SEEK_END)
            h.update(fh.read(_PARTIAL_WINDOW))
    return h.hexdigest()


# Filesystem timestamp granularity is not always nanoseconds — network and
# Drive-backed mounts routinely round to 1 s (and FAT-derived ones to 2 s). If a
# file is modified in the same tick that we hashed it, size and mtime are both
# unchanged and a naive cache silently returns a stale hash. Git calls such
# entries "racily clean" and re-reads them; we do the same, treating any entry
# whose file mtime is not strictly older than the moment we recorded it as
# untrusted. Cheap insurance: it only ever costs a re-hash.
_RACE_WINDOW_NS = 2_000_000_000  # 2 s
_CACHE_VERSION = 2


class _HashCache:
    """Memoise hashes across runs, keyed by (path, size, mtime_ns).

    Entries are stored as [hash, recorded_at_ns] so the race guard above can
    decide whether a hit is trustworthy.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, list] = {}
        if path.exists():
            try:
                blob = json.loads(path.read_text())
                if isinstance(blob, dict) and blob.get("version") == _CACHE_VERSION:
                    self._data = blob.get("entries", {})
                # Older cache formats are simply discarded rather than trusted.
            except (OSError, json.JSONDecodeError):
                self._data = {}
        self._dirty = False

    @staticmethod
    def _key(path: Path, st: os.stat_result, mode: str) -> str:
        return f"{mode}:{st.st_size}:{st.st_mtime_ns}:{path}"

    def get_or_compute(self, path: Path, st: os.stat_result, mode: str) -> str:
        if mode == "size_mtime":
            return f"sm-{st.st_size}-{st.st_mtime_ns}"

        key = self._key(path, st, mode)
        entry = self._data.get(key)
        if entry is not None:
            value, recorded_at = entry[0], entry[1]
            if st.st_mtime_ns + _RACE_WINDOW_NS < recorded_at:
                return value  # comfortably older than our reading: trustworthy

        value = _md5_full(path) if mode == "full" else _md5_partial(path, st.st_size)
        self._data[key] = [value, time.time_ns()]
        self._dirty = True
        return value

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"version": _CACHE_VERSION, "entries": self._data}))


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #
def _iso(mtime: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(mtime, dt.timezone.utc).isoformat(timespec="seconds")


def _entry(row: dict, role: str, path: Path, nm_root: Path,
           cache: _HashCache, mode: str) -> dict[str, Any]:
    name = row["image_name"] if role == "image" else row["mask_name"]
    drive_id = row.get("image_id" if role == "image" else "mask_id", "")
    try:
        rel = str(path.relative_to(nm_root))
    except ValueError:
        rel = str(path)

    out: dict[str, Any] = {k: row.get(k, "") for k in _CARRY}
    out.update(role=role, name=name, drive_id=drive_id, rel_path=rel)

    if not path.exists():
        out.update(exists=0, size_bytes="", mtime_utc="", content_hash="", hash_mode=mode)
        return out

    st = path.stat()
    out.update(
        exists=1,
        size_bytes=st.st_size,
        mtime_utc=_iso(st.st_mtime),
        content_hash=cache.get_or_compute(path, st, mode),
        hash_mode=mode,
    )
    return out


def build_index(manifest_csv: Path, nm_root: Path, out_csv: Path,
                hash_mode: str = "full", cache_path: Path | None = None,
                ) -> dict[str, Any]:
    """Index every file referenced by `manifest_csv`. Returns a summary dict."""
    manifest_csv, nm_root, out_csv = Path(manifest_csv), Path(nm_root), Path(out_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    if hash_mode not in {"full", "partial", "size_mtime"}:
        raise ValueError(f"hash_mode must be full|partial|size_mtime, got {hash_mode!r}")

    with open(manifest_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    cache = _HashCache(cache_path or Path("/tmp/bluespotter_hash_cache.json"))
    entries: list[dict[str, Any]] = []

    for i, row in enumerate(rows, 1):
        img, msk, is_npy = resolve_row(nm_root, row)
        entries.append(_entry(row, "image", img, nm_root, cache, hash_mode))
        # For seg_npy rows image and mask are the same file; indexing it once is
        # enough and avoids double-hashing a large .npy.
        if not is_npy:
            entries.append(_entry(row, "mask", msk, nm_root, cache, hash_mode))
        if i % 20 == 0:
            print(f"    indexed {i}/{len(rows)} rows")

    cache.flush()

    # Deterministic order so the output hash depends only on content, never on
    # dict/filesystem iteration order.
    entries.sort(key=lambda e: (e["rel_path"], e["role"]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(INDEX_FIELDS))
        w.writeheader()
        w.writerows(entries)

    present = [e for e in entries if e["exists"]]
    missing = [e for e in entries if not e["exists"]]
    total_bytes = sum(int(e["size_bytes"]) for e in present)

    # A single hash standing for "this exact dataset content".
    roll = hashlib.md5()  # integrity only, not security
    for e in entries:
        roll.update(f"{e['rel_path']}|{e['role']}|{e['content_hash']}".encode())

    summary = {
        "manifest": manifest_csv.name,
        "rows": len(rows),
        "files_indexed": len(entries),
        "files_present": len(present),
        "files_missing": len(missing),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 3),
        "hash_mode": hash_mode,
        "dataset_hash": roll.hexdigest(),
        "missing_examples": [e["rel_path"] for e in missing[:10]],
    }
    print(f"  {manifest_csv.name}: {len(present)}/{len(entries)} files present, "
          f"{summary['total_gib']} GiB, dataset_hash={summary['dataset_hash'][:12]}")
    if missing:
        print(f"  WARNING: {len(missing)} referenced file(s) missing from Drive:")
        for p in summary["missing_examples"]:
            print(f"    - {p}")
    return summary


# --------------------------------------------------------------------------- #
# CLI (DVC stage entry point)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Content-index Drive files for a manifest split.")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--params", default=None, help="path to params.yaml")
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.params)
    dvc_cfg = cfg.raw.get("dvc", {})
    repo = cfg.repo_root

    manifest = repo / dvc_cfg["manifest_dir"] / f"{args.split}.csv"
    # Big index CSV -> data/ (DVC-cached, gitignored).
    out_csv = repo / dvc_cfg["index_dir"] / f"{args.split}_index.csv"
    # Small summary -> reports/ (committed to git, so it shows up in PR diffs
    # and `dvc metrics diff` works on GitHub).
    out_json = repo / dvc_cfg["report_dir"] / f"{args.split}_index_summary.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"[index] split={args.split}  manifest={manifest}")
    summary = build_index(
        manifest_csv=manifest,
        nm_root=cfg.nmslices_root,
        out_csv=out_csv,
        hash_mode=dvc_cfg.get("hash_mode", "full"),
        cache_path=cfg.local_cache / "hash_cache.json",
    )
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    if summary["files_missing"] and dvc_cfg.get("fail_on_missing", True):
        print("\n[index] FAILED: manifest references files that are not on the Drive mount.")
        print("        Fix the manifest or restore the files, or set dvc.fail_on_missing=false.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
