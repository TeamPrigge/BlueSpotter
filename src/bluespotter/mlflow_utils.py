"""MLflow setup for BlueSpotter: tracking, data provenance, model registration.

Runs are recorded in a SQLite backend store whose durable home is Google Drive but
which SQLite only ever opens on local disk — see `bluespotter.mlflow_store` for why
that split is necessary and `docs/MLFLOW.md` for the whole picture. A database
backend (rather than the legacy `mlruns/` file store) is what makes the Model
Registry available at all.

Set the MLFLOW_TRACKING_URI env var to point at a hosted tracking server instead;
it takes precedence over everything here, so no code change is required.
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
    """Configure MLflow tracking + experiment. Returns the resolved tracking URI.

    For the `sqlite` backend this first restores the database from Drive to local
    disk and brings its schema up to date, because SQLite cannot safely operate on
    the Drive FUSE mount (see bluespotter.mlflow_store for the reasoning).
    Artifacts still live on Drive — they are write-once blobs, which the mount
    handles perfectly well.
    """
    uri = cfg.tracking_uri()
    artifact_location = None

    if uri.startswith("sqlite:"):
        from . import mlflow_store
        mlflow_store.restore(cfg)
        mlflow_store.db_upgrade(cfg)
        artifact_location = mlflow_store.artifact_root(cfg)
        artifact_location.mkdir(parents=True, exist_ok=True)
        artifact_location = artifact_location.as_uri()

    elif uri.startswith("file://"):
        # Legacy path. MLflow 3.x turned the file store off by default and it
        # cannot support the Model Registry, so this is kept only so old configs
        # keep working. Prefer mlflow.backend: sqlite.
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        Path(uri.replace("file://", "")).mkdir(parents=True, exist_ok=True)
        print("  NOTE: file:// backend is legacy and has no Model Registry. "
              "Set mlflow.backend: sqlite in params.yaml.")

    mlflow.set_tracking_uri(uri)

    # set_experiment() cannot set artifact_location on an experiment that already
    # exists, so create it explicitly the first time to pin artifacts onto Drive.
    name = cfg.mlflow["experiment_name"]
    if artifact_location and mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)

    print(f"  MLflow tracking URI : {uri}")
    print(f"  MLflow experiment   : {name}")
    if artifact_location:
        print(f"  MLflow artifacts    : {artifact_location}")
    return uri


_DB_BACKED = ("sqlite:", "http://", "https://", "postgresql:", "postgres:", "mysql:", "mssql:")


def register_model(cfg: Config, run_id: str, weights_path: Path | str | None = None,
                   artifact_path: str = "model") -> str | None:
    """Register the run's weights in the Model Registry, returning the version.

    The registry is the piece the file store could not provide, and it is what lets
    `deploy/` refer to `models:/bluespotter-lc/3` instead of a filename on Drive
    that someone can overwrite.

    Cellpose weights are not an MLflow "flavor", and since MLflow 3 you cannot
    register a bare artifact directory — `runs:/<id>/<path>` now resolves only to a
    *logged model*. The right tool is `mlflow.create_external_model()`, which exists
    precisely for models whose artifacts live outside MLflow (here: on Drive). It
    creates a LoggedModel that carries params and tags but no payload, and that can
    be registered by its model id. The weights themselves remain a run artifact and
    a file on Drive; the registry records the version, the lineage and where the
    bytes are.
    """
    name = cfg.mlflow.get("registered_model_name")
    if not name:
        return None
    if not str(cfg.tracking_uri()).startswith(_DB_BACKED):
        print("  Registry skipped: needs a database-backed or served tracking store "
              "(mlflow.backend: sqlite, or MLFLOW_TRACKING_URI to a server).")
        return None

    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
        # Carry the data/code provenance onto the model itself, so a registry entry
        # answers "which dataset and commit produced this?" without a join.
        keys = ("dataset_hash_train", "dataset_hash_test", "git_commit_short",
                "git_commit", "grouped_split")
        tags = {k: v for k in keys if (v := run.data.tags.get(k))}
        params = {"weights_path": str(weights_path)} if weights_path else None

        model_uri = None
        if hasattr(mlflow, "create_external_model"):
            lm = mlflow.create_external_model(
                name=f"{name}-{run_id[:8]}", source_run_id=run_id,
                model_type="cellpose-sam", params=params, tags=tags,
            )
            model_uri = f"models:/{lm.model_id}"
        else:  # MLflow 2.x
            model_uri = f"runs:/{run_id}/{artifact_path}"

        mv = mlflow.register_model(model_uri=model_uri, name=name)
        for k, v in tags.items():
            client.set_model_version_tag(name, mv.version, k, v)
        if weights_path:
            client.set_model_version_tag(name, mv.version, "weights_path", str(weights_path))
        print(f"  Registered model    : {name} v{mv.version}   ({model_uri})")
        return str(mv.version)
    except Exception as e:  # a registry hiccup must not fail training
        print(f"  Registry skipped: {type(e).__name__}: {e}")
        return None


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
