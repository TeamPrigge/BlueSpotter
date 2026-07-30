"""Cellpose-SAM transfer learning for locus coeruleus segmentation.

Run end-to-end from the Colab notebook or the CLI:

    python -m bluespotter.train

The run() function is written as a sequence of clearly labelled steps so it is
easy to follow what is happening at each stage:

    1. Environment check   - GPU available? which cellpose/torch versions?
    2. Data                - copy from Drive to fast local disk, then load pairs
    3. Model               - load the pretrained Cellpose-SAM (cpsam) weights
    4. Train               - fine-tune, logging params + per-epoch losses to MLflow
    5. Save                - copy the trained model back to Drive + log as artifact
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
from .mlflow_utils import (
    log_data_provenance,
    log_params_from_config,
    start_tracking,
    write_dvc_metrics,
)


def _banner(step: str, text: str) -> None:
    """Print a clear section header so the notebook output is easy to read."""
    print("\n" + "=" * 70)
    print(f"[{step}] {text}")
    print("=" * 70)


def _split(images, labels, test_split: float):
    """Deterministic train/val split used when no separate test folder exists."""
    n = len(images)
    k = max(1, round(n * test_split))
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

    # ---- STEP 1: environment ------------------------------------------------
    _banner("1/5", "Environment check")
    try:
        import torch
        gpu = torch.cuda.is_available()
        print(f"  GPU available : {gpu}"
              + (f"  ({torch.cuda.get_device_name(0)})" if gpu
                 else "  <-- switch Colab to a GPU runtime!"))
    except Exception as e:
        gpu = False
        print(f"  Could not query torch/GPU: {e}")

    # ---- STEP 2: data -------------------------------------------------------
    if cfg.use_manifest:
        _banner("2/5", "Data: load image/mask pairs from manifest (train.csv / test.csv)")
        from .manifest import load_manifest
        # Prefer the DVC-tracked snapshot in the repo over the live Drive copy, so
        # the run is pinned to a manifest revision recorded in dvc.lock rather
        # than to whatever the CSV happens to say right now.
        train_mf, test_mf = cfg.manifest_for("train"), cfg.manifest_for("test")
        pinned = train_mf == cfg.repo_manifest("train")
        print(f"  Train manifest : {train_mf}")
        print(f"  Test  manifest : {test_mf}")
        print(f"  Image source   : {cfg.nmslices_root}")
        print(f"  Version pinned : {pinned}"
              + ("" if pinned else "   <-- run `dvc repro sync-manifests` to pin this run"))
        images, labels = load_manifest(train_mf, cfg.nmslices_root)
        test_images, test_labels = load_manifest(test_mf, cfg.nmslices_root)
    else:
        _banner("2/5", "Data: cache from Drive folder -> local disk, then load pairs")
        print(f"  Drive data dir : {cfg.data_dir}")
        print(f"  Local cache    : {cfg.local_cache}")
        local = cache_from_drive(cfg.data_dir, cfg.local_cache)
        images, labels, test_images, test_labels = load_dataset(
            local,
            image_filter=cfg.data["image_filter"],
            mask_filter=cfg.data["mask_filter"],
        )
        if not test_images:
            print("  No separate test folder -> splitting off "
                  f"{cfg.data['test_split']:.0%} for validation")
            images, labels, test_images, test_labels = _split(
                images, labels, cfg.data["test_split"])
    print(f"  Training images : {len(images)}")
    print(f"  Validation images: {len(test_images)}")
    if len(images) == 0:
        raise RuntimeError("No training images found. Check params.yaml paths "
                           "and the *_img/_masks filters.")

    # ---- STEP 3: MLflow + model --------------------------------------------
    _banner("3/5", "MLflow tracking + load pretrained Cellpose-SAM")
    start_tracking(cfg)
    print(f"  Loading pretrained model: {cfg.model['pretrained']}  (Cellpose-SAM)")
    model = models.CellposeModel(gpu=gpu, pretrained_model=cfg.model["pretrained"])

    run_name = f"{cfg.model['name']}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    with mlflow.start_run(run_name=run_name):
        log_params_from_config(cfg)
        mlflow.log_param("n_train", len(images))
        mlflow.log_param("n_test", len(test_images))
        mlflow.log_param("gpu", gpu)
        # Record git commit + DVC dataset hashes so this run can be traced back
        # to the exact bytes it trained on.
        log_data_provenance(cfg)

        # ---- STEP 4: fine-tune ---------------------------------------------
        _banner("4/5", f"Fine-tuning  (run: {run_name})")
        print(f"  epochs={cfg.train['n_epochs']}  lr={cfg.train['learning_rate']}  "
              f"weight_decay={cfg.train['weight_decay']}  batch_size={cfg.train['batch_size']}")
        print("  ...training (this can take a while; watch the loss go down)...")
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
            min_train_masks=cfg.train.get("min_train_masks", 1),
            save_path=str(cfg.local_cache),
            model_name=run_name,
        )

        # log per-epoch losses so you get curves in the MLflow UI
        for epoch, tl in enumerate(np.atleast_1d(train_losses)):
            mlflow.log_metric("train_loss", float(tl), step=epoch)
        for epoch, vl in enumerate(np.atleast_1d(test_losses)):
            if vl is not None and not np.isnan(vl):
                mlflow.log_metric("test_loss", float(vl), step=epoch)
        train_hist = [float(x) for x in np.atleast_1d(train_losses)]
        test_hist = [
            float(x) for x in np.atleast_1d(test_losses)
            if x is not None and not np.isnan(x)
        ]
        final_train = train_hist[-1]
        print(f"  Final train loss: {final_train:.4f}")

        # Mirror the headline numbers into reports/ so `dvc metrics diff` can show
        # the effect of a change on a pull request, not just in the MLflow UI.
        write_dvc_metrics(
            cfg,
            metrics={
                "final_train_loss": final_train,
                "best_train_loss": min(train_hist) if train_hist else None,
                "final_test_loss": test_hist[-1] if test_hist else None,
                "best_test_loss": min(test_hist) if test_hist else None,
                "n_epochs": cfg.train["n_epochs"],
                "n_train_images": len(images),
                "n_test_images": len(test_images),
                "run_name": run_name,
            },
            losses={"train": train_hist, "test": test_hist},
        )

        # ---- STEP 5: save --------------------------------------------------
        _banner("5/5", "Save model to Drive + log as MLflow artifact")
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        drive_model = cfg.model_dir / Path(model_path).name
        shutil.copy2(model_path, drive_model)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_param("model_drive_path", str(drive_model))
        print(f"  Model saved to Drive: {drive_model}")

    print("\nDONE. Trained model path returned to caller.\n")
    return drive_model


if __name__ == "__main__":
    run()
