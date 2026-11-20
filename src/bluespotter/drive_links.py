"""Attach Google Drive file-IDs (and therefore shareable links) to manifest rows.

Discovery walks a *mounted* Drive, which gives paths but no file-IDs — the FUSE
layer hides them. That is fine for training, which reads by path, but not for
`ap_position.csv`: those rows get opened by hand, by people checking whether a
slice really sits at the bregma coordinate its name claims. A path they cannot
click is a path they will not check.

So this module fills the IDs in from the Drive API when credentials happen to be
available (they are, inside Colab, after `google.colab.auth.authenticate_user()`),
and does nothing at all when they are not. Enrichment is strictly optional: a
manifest without links is still a valid, trainable manifest, and no pipeline
stage is allowed to fail because the API was unreachable.

Lookups are cached to `reports/drive_ids.json`, keyed by name, so a second run
costs no API calls for files that have not moved.

    python -m bluespotter.drive_links data/manifests/ap_position.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

FILE_URL = "https://drive.google.com/file/d/{id}/view"


def _colab_service():  # pragma: no cover - requires Colab runtime
    """Return an authenticated Drive v3 client, or None if unavailable."""
    try:
        from googleapiclient.discovery import build as _build  # type: ignore
    except ImportError:
        return None
    try:
        from google.colab import auth  # type: ignore

        auth.authenticate_user()
        return _build("drive", "v3")
    except Exception:
        pass
    try:
        import google.auth  # type: ignore

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        return _build("drive", "v3", credentials=creds)
    except Exception:
        return None


def _lookup_factory(service) -> Callable[[str], str]:  # pragma: no cover
    def lookup(name: str) -> str:
        # Escape single quotes the way the Drive query language wants.
        safe = name.replace("\\", "\\\\").replace("'", "\\'")
        try:
            res = service.files().list(
                q=f"name = '{safe}' and trashed = false",
                fields="files(id)", pageSize=1,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
        except Exception:
            return ""
        files = res.get("files", [])
        return files[0]["id"] if files else ""

    return lookup


def enrich(csv_path: Path, cache_path: Path | None = None,
           lookup: Callable[[str], str] | None = None) -> dict[str, Any]:
    """Fill image_id / mask_id / drive_url in `csv_path`. Returns a summary.

    `lookup` is injectable so the logic is testable without a network or a
    Google account; production passes the Drive-API-backed one.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {"status": "missing", "path": str(csv_path), "resolved": 0}

    cache: dict[str, str] = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}

    if lookup is None:  # pragma: no cover - depends on ambient credentials
        service = _colab_service()
        if service is None:
            return {"status": "no_credentials", "path": str(csv_path), "resolved": 0,
                    "hint": "run inside Colab after google.colab.auth.authenticate_user()"}
        lookup = _lookup_factory(service)

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    for extra in ("drive_url",):
        if extra not in fields:
            fields.append(extra)

    resolved = misses = 0
    for r in rows:
        name = r.get("image_name", "")
        if not name:
            continue
        if name not in cache:
            cache[name] = lookup(name)
        fid = cache[name]
        if not fid:
            misses += 1
            continue
        resolved += 1
        r["drive_url"] = FILE_URL.format(id=fid)
        # Only overwrite the ID columns when they still hold a path; a real
        # Drive ID that is already there is more trustworthy than a name lookup,
        # which can collide when two cohorts reuse a file name.
        if "/" in (r.get("image_id") or ""):
            r["image_id"] = fid
        if r.get("mask_name") == name and "/" in (r.get("mask_id") or ""):
            r["mask_id"] = fid

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, indent=2, sort_keys=True))

    return {"status": "ok", "path": str(csv_path), "rows": len(rows),
            "resolved": resolved, "unresolved": misses}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", default="data/manifests/ap_position.csv")
    ap.add_argument("--cache", default="reports/drive_ids.json")
    args = ap.parse_args(argv)

    summary = enrich(Path(args.csv), Path(args.cache))
    print(f"[links] {summary['status']}: "
          f"{summary.get('resolved', 0)} resolved, "
          f"{summary.get('unresolved', 0)} not found")
    if summary["status"] == "no_credentials":
        print(f"        {summary['hint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
