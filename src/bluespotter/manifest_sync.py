"""Snapshot the authored manifests from Drive into the repo, under DVC control.

The division of labour:

  Drive  is the *authoring* surface. You (or a collaborator) edit
         data/training_data/train.csv and data/test_data/test.csv there, in a
         sheet, with the images next to them.
  Repo   is the *versioning* surface. This stage copies those CSVs to
         data/manifests/, where DVC hashes them and git records the hash.

So Drive stays convenient and browsable, while every training run can still name
the exact manifest revision it used. Editing Drive alone changes nothing about a
past experiment; it only shows up once someone re-runs this stage and commits the
resulting `dvc.lock` diff — which is precisely the reviewable moment you want.

Run:

    python -m bluespotter.manifest_sync
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .config import Config, load_config


def _md5(path: Path) -> str:
    h = hashlib.md5()  # integrity only, not security
    h.update(path.read_bytes())
    return h.hexdigest()


def _n_rows(path: Path) -> int:
    with open(path, newline="") as fh:
        return max(0, sum(1 for _ in fh) - 1)  # minus header


def sync_one(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(
            f"Source manifest not found on the Drive mount:\n  {src}\n"
            "Is Drive mounted, and does drive.root in params.yaml point at it?"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)

    old = _md5(dst) if dst.exists() else None
    shutil.copy2(src, dst)
    new = _md5(dst)

    status = "unchanged" if old == new else ("created" if old is None else "updated")
    info = {
        "source": str(src),
        "dest": str(dst),
        "md5": new,
        "previous_md5": old,
        "status": status,
        "rows": _n_rows(dst),
        "bytes": dst.stat().st_size,
    }
    print(f"  {dst.name:<12} {status:<10} {info['rows']:>4} rows  md5={new[:12]}")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Copy manifests from Drive into the repo.")
    ap.add_argument("--params", default=None)
    args = ap.parse_args(argv)

    cfg: Config = load_config(args.params)
    dvc_cfg = cfg.raw.get("dvc", {})
    repo = cfg.repo_root
    mdir = repo / dvc_cfg["manifest_dir"]

    print(f"[sync] Drive root : {cfg.drive_root}")
    print(f"[sync] repo dest  : {mdir.relative_to(repo)}")

    result = {
        "drive_root": str(cfg.drive_root),
        "train": sync_one(cfg.train_manifest, mdir / "train.csv"),
        "test": sync_one(cfg.test_manifest, mdir / "test.csv"),
    }

    # The AP manifest is the seed for the shape->rostrocaudal-position model. It
    # is versioned exactly like the segmentation manifests, but kept in its own
    # file so the two datasets can be split and validated independently. Absent
    # on older Drive layouts, so its absence is not an error.
    ap_rel = cfg.raw["data"].get("ap_manifest")
    if ap_rel:
        ap_dst = mdir / "ap_position.csv"
        ap_src = cfg.drive_root / ap_rel
        if ap_src.exists():
            result["ap_position"] = sync_one(ap_src, ap_dst)
        else:
            # The stage declares this file as an output, so it has to exist.
            # A header-only CSV is honest: schema present, zero rows, and the
            # provenance record below says exactly why.
            from .discover import AP_FIELDS

            ap_dst.parent.mkdir(parents=True, exist_ok=True)
            ap_dst.write_text(",".join(AP_FIELDS) + "\n")
            result["ap_position"] = {
                "source": str(ap_src), "dest": str(ap_dst), "status": "absent",
                "rows": 0,
                "hint": "run `python -m bluespotter.discover` to generate it",
            }
            print("  ap_position  absent on Drive — wrote header-only stub; run "
                  "`python -m bluespotter.discover` to populate it")

    # Provenance goes to reports/ so it is committed to git — the manifests
    # themselves are DVC-cached and therefore gitignored.
    prov = repo / dvc_cfg["report_dir"] / "manifest_source.json"
    prov.parent.mkdir(parents=True, exist_ok=True)
    prov.write_text(json.dumps(result, indent=2) + "\n")
    print(f"  provenance -> {prov.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
