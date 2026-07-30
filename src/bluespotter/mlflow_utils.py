"""MLflow setup for BlueSpotter.

Phase 0 logs into a Drive-synced `mlruns/` folder so runs survive Colab session
resets. Set the MLFLOW_TRACKING_URI env var to point at a hosted tracking server
later (e.g. DagsHub) - no code change required.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from .config import Config


def start_tracking(cfg: Config) -> str:
    """Configure MLflow tracking + experiment. Returns the resolved tracking URI."""
    uri = cfg.tracking_uri()

    # MLflow 3.x disabled the local/Drive file store ("file://.../mlruns") by
    # default and raises unless you opt in. We *intentionally* use a Drive file
    # store in phase 0 so runs persist across Colab resets, so opt back in here.
    # (When you move MLFLOW_TRACKING_URI to a real server this branch is skipped.)
    if uri.startswith("file://"):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        Path(uri.replace("file://", "")).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.mlflow["experiment_name"])
    print(f"  MLflow tracking URI : {uri}")
    print(f"  MLflow experiment   : {cfg.mlflow['experiment_name']}")
    return uri


def log_params_from_config(cfg: Config) -> None:
    """Log the training + model config so every run is fully reproducible."""
    mlflow.log_params(
        {
            "pretrained": cfg.model["pretrained"],
            "n_epochs": cfg.train["n_epochs"],
            "learning_rate": cfg.train["learning_rate"],
            "weight_decay": cfg.train["weight_decay"],
            "batch_size": cfg.train["batch_size"],
            "nchan": cfg.train["nchan"],
            "image_filter": cfg.data["image_filter"],
            "mask_filter": cfg.data["mask_filter"],
        }
    )


# --------------------------------------------------------------------------- #
# Data provenance: tie every MLflow run to an exact DVC dataset version
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> str | None:
    """Run a git command in `repo`, returning stripped stdout or None."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def log_data_provenance(cfg: Config) -> dict[str, Any]:
    """Log which code revision and which dataset version this run actually used.

    Without this, "MLflow run #47 scored 0.82" is an orphan number — you cannot
    later establish which images produced it. With it, every run carries the git
    commit plus the content hash of the indexed Drive files, so a result can be
    traced back to bytes even months later.

    Everything here is best-effort: a missing git binary or an un-run DVC stage
    degrades to fewer tags, never to a failed training run.
    """
    repo = cfg.repo_root
    dvc_cfg = cfg.dvc
    tags: dict[str, str] = {}
    params: dict[str, Any] = {}

    # --- code revision ---------------------------------------------------- #
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = _git(repo, "status", "--porcelain")
    if commit:
        tags["git_commit"] = commit
        tags["git_commit_short"] = commit[:12]
    if branch:
        tags["git_branch"] = branch
    # A dirty tree means the recorded commit does not fully describe this run.
    tags["git_dirty"] = "true" if dirty else "false"
    origin = _git(repo, "config", "--get", "remote.origin.url")
    if origin:
        tags["git_remote"] = origin

    # --- dataset version, from the DVC index summaries -------------------- #
    report_dir = repo / dvc_cfg.get("report_dir", "reports")
    for split in ("train", "test"):
        summary = _read_json(report_dir / f"{split}_index_summary.json")
        if not summary:
            continue
        tags[f"dataset_hash_{split}"] = str(summary.get("dataset_hash", ""))[:32]
        params[f"{split}_files_present"] = summary.get("files_present")
        params[f"{split}_files_missing"] = summary.get("files_missing")
        params[f"{split}_gib"] = summary.get("total_gib")
        params[f"{split}_hash_mode"] = summary.get("hash_mode")

    # --- manifest revision ------------------------------------------------ #
    src = _read_json(report_dir / "manifest_source.json")
    if src:
        for split in ("train", "test"):
            if md5 := (src.get(split) or {}).get("md5"):
                tags[f"manifest_md5_{split}"] = md5

    # --- data-quality state at training time ------------------------------ #
    report = _read_json(report_dir / "validation.json")
    if report:
        leak = report.get("leakage", {})
        params["validation_ok"] = report.get("ok")
        params["n_mouse_overlap"] = leak.get("n_mouse_overlap")
        params["n_slice_overlap"] = leak.get("n_slice_overlap")
        # Surfaced as a tag so leaky runs are filterable in the MLflow UI and can
        # never be mistaken for clean held-out results.
        tags["grouped_split"] = "false" if (
            leak.get("n_mouse_overlap") or leak.get("n_slice_overlap")
        ) else "true"

    if tags:
        mlflow.set_tags(tags)
    if params:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

    # Keep the actual manifests + indexes with the run, so it stays readable even
    # if the repo moves.
    for rel in (dvc_cfg.get("manifest_dir", "data/manifests"),
                dvc_cfg.get("index_dir", "data/index")):
        d = repo / rel
        if d.is_dir():
            mlflow.log_artifacts(str(d), artifact_path=f"data_version/{Path(rel).name}")

    print(f"  Provenance: commit={tags.get('git_commit_short', '?')} "
          f"dirty={tags.get('git_dirty')} "
          f"train_hash={tags.get('dataset_hash_train', '?')[:12]}")
    return {"tags": tags, "params": params}


def write_dvc_metrics(cfg: Config, metrics: dict[str, Any],
                      losses: dict[str, list[float]] | None = None) -> None:
    """Write metrics/plots where dvc.yaml expects them, for `dvc metrics diff`.

    MLflow is for exploring many runs interactively; DVC metrics are for the diff
    on a pull request. Writing both costs nothing and they answer different
    questions.
    """
    report_dir = cfg.repo_root / cfg.dvc.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    if losses:
        import csv as _csv

        train_l = losses.get("train", [])
        test_l = losses.get("test", [])
        n = max(len(train_l), len(test_l))
        with open(report_dir / "loss_curve.csv", "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["epoch", "train_loss", "test_loss"])
            for i in range(n):
                w.writerow([
                    i,
                    train_l[i] if i < len(train_l) else "",
                    test_l[i] if i < len(test_l) else "",
                ])
