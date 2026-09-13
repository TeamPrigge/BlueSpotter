"""Configuration loading and path resolution for BlueSpotter.

All tunables live in params.yaml. This module loads that file once and exposes
resolved, absolute paths so the rest of the code never hard-codes a location.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _find_params(start: Path | None = None) -> Path:
    """Locate params.yaml by walking up from this file (repo root)."""
    here = (start or Path(__file__).resolve()).parent
    for candidate in [here, *here.parents]:
        p = candidate / "params.yaml"
        if p.exists():
            return p
    raise FileNotFoundError("params.yaml not found in any parent directory")


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    params_path: Path | None = None

    # --- Repo layout (git + DVC live here) ---
    @property
    def repo_root(self) -> Path:
        """Directory containing params.yaml — i.e. the git/DVC repo root."""
        if self.params_path is not None:
            return Path(self.params_path).resolve().parent
        return _find_params().parent

    @property
    def dvc(self) -> dict[str, Any]:
        return self.raw.get("dvc", {})

    def repo_manifest(self, split: str) -> Path:
        """The DVC-tracked manifest snapshot inside the repo."""
        return self.repo_root / self.dvc.get("manifest_dir", "data/manifests") / f"{split}.csv"

    def manifest_for(self, split: str) -> Path:
        """Manifest to actually train from.

        Prefers the DVC-tracked repo snapshot (so the run is pinned to a commit)
        and falls back to reading the live copy on Drive if the snapshot has not
        been created yet.
        """
        snap = self.repo_manifest(split)
        if snap.exists():
            return snap
        return self.train_manifest if split == "train" else self.test_manifest

    # --- Drive paths (source of truth for data + model) ---
    @property
    def drive_root(self) -> Path:
        return Path(self.raw["drive"]["root"])

    @property
    def data_dir(self) -> Path:
        return self.drive_root / self.raw["drive"]["data_subdir"]

    @property
    def model_dir(self) -> Path:
        return self.drive_root / self.raw["drive"]["model_subdir"]

    @property
    def mlruns_dir(self) -> Path:
        return self.drive_root / self.raw["drive"]["mlruns_subdir"]

    # --- Local scratch (fast training I/O) ---
    @property
    def local_cache(self) -> Path:
        return Path(self.raw["data"]["local_cache"])

    @property
    def use_manifest(self) -> bool:
        return bool(self.raw["data"].get("use_manifest", False))

    @property
    def train_manifest(self) -> Path:
        return self.drive_root / self.raw["data"]["train_manifest"]

    @property
    def test_manifest(self) -> Path:
        return self.drive_root / self.raw["data"]["test_manifest"]

    @property
    def nmslices_root(self) -> Path:
        return Path(self.raw["data"]["nmslices_root"])

    # --- Convenience accessors ---
    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def train(self) -> dict[str, Any]:
        return self.raw["train"]

    @property
    def mlflow(self) -> dict[str, Any]:
        return self.raw["mlflow"]

    def tracking_uri(self) -> str:
        """MLflow tracking URI.

        Precedence: MLFLOW_TRACKING_URI env var (so a hosted server can be dropped
        in without touching code) > params.yaml mlflow.tracking_uri > the backend
        selected by mlflow.backend.

        For the `sqlite` backend this returns the *local* working copy, never the
        path on Drive — SQLite must not operate on a FUSE mount. The Drive copy is
        the durable home, managed by bluespotter.mlflow_store.
        """
        env = os.environ.get("MLFLOW_TRACKING_URI")
        if env:
            return env
        cfg = self.raw["mlflow"].get("tracking_uri") or ""
        if cfg:
            return cfg
        if self.raw["mlflow"].get("backend", "file") == "sqlite":
            from .mlflow_store import local_db, sqlite_uri
            return sqlite_uri(local_db(self))
        return f"file://{self.mlruns_dir}"


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else _find_params()
    with open(p) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, params_path=Path(p).resolve())
