"""MLflow setup for BlueSpotter.

Phase 0 logs into a Drive-synced `mlruns/` folder so runs survive Colab session
resets. Set the MLFLOW_TRACKING_URI env var to point at a hosted tracking server
later — no code change required.
"""
from __future__ import annotations

from pathlib import Path

import mlflow

from .config import Config


def start_tracking(cfg: Config) -> str:
    """Configure MLflow tracking + experiment. Returns the resolved tracking URI."""
    uri = cfg.tracking_uri()

    # If logging to the local/Drive file store, make sure the folder exists.
    if uri.startswith("file://"):
        Path(uri.replace("file://", "")).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.mlflow["experiment_name"])
    print(f"MLflow tracking URI: {uri}")
    print(f"MLflow experiment:   {cfg.mlflow['experiment_name']}")
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
