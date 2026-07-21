"""Cellpose-SAM transfer learning for locus coeruleus segmentation.

Run end-to-end from the Colab notebook or the CLI:

    python -m bluespotter.train

It reads params.yaml, caches data from Drive, fine-tunes Cellpose-SAM, logs the
run to MLflow, and copies the trained weights back to Drive.
"""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

import mlflow
import numpy as np
from cellpose import io, models, train

from .config import Config, load_config
from .data import cache_from_drive, load_dataset
from .mlflow_utils import log_params_from_config, start_tracking


def _split(images, labels, test_split: float):
    """Simple deterministic train/val split when no test dir is provided."""
    n = len(images)
    k = max(1, int(round(n * test_split)))
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    val_idx, tr_idx = idx[:k], idx[k:]
    tr_x = [images[i] for i in tr_idx]
    tr_y = [labels[i] for i in tr_idx]
    va_x = [images[i] for i in val_idx]
    va_y = [labels[i] for i in val_idx]
    return tr_x, tr_y, va_x, va_y


def run(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    io.logger_setup()
    start_tracking(cfg)

    # 1. Data: Drive -> local scratch -> loaded arrays
    local = cache_from_drive(cfg.data_dir, cfg.local_cache)
    images, labels, test_images, test_labels = load_dataset(
        local,
        image_filter=cfg.data["image_filter"],
        mask_filter=cfg.data["mask_filter"],
    )
    if not test_images:
        images, labels, test_images, test_labels = _split(
            images, labels, cfg.data["test_split"]
        )

    # 2. Model: start from pretrained Cellpose-SAM weights
    model = models.CellposeModel(gpu=True, pretrained_model=cfg.model["pretrained"])

    run_name = f"{cfg.model['name']}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(run_name=run_name):
        log_params_from_config(cfg)
        mlflow.log_param("n_train", len(images))
        mlflow.log_param("n_test", len(test_images))

        # 3. Fine-tune
        model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=images,
            train_labels=labels,
            test_data=test_images,
            test_labels=test_labels,
            n_epochs=cfg.train["n_epochs"],
            learning_rate=cfg.train["learning_rate"],
            weight_decay=cfg.train["weight_decay"],
            batch_size=cfg.train["batch_size"],
            nchan=cfg.train["nchan"],
            save_path=str(cfg.local_cache),
            model_name=run_name,
        )

        # 4. Log metrics per epoch
        for epoch, tl in enumerate(np.atleast_1d(train_losses)):
            mlflow.log_metric("train_loss", float(tl), step=epoch)
        for epoch, vl in enumerate(np.atleast_1d(test_losses)):
            if vl is not None and not np.isnan(vl):
                mlflow.log_metric("test_loss", float(vl), step=epoch)

        # 5. Persist model: to Drive (source of truth) + MLflow artifact
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        drive_model = cfg.model_dir / Path(model_path).name
        shutil.copy2(model_path, drive_model)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_param("model_drive_path", str(drive_model))

        print(f"\nTrained model saved to Drive: {drive_model}")
        return drive_model


if __name__ == "__main__":
    run()
