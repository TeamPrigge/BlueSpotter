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
        """MLflow tracking URI. Env var wins; else params.yaml; else Drive mlruns."""
        env = os.environ.get("MLFLOW_TRACKING_URI")
        if env:
            return env
        cfg = self.raw["mlflow"].get("tracking_uri") or ""
        if cfg:
            return cfg
        return f"file://{self.mlruns_dir}"


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else _find_params()
    with open(p, "r") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
