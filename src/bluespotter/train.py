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

import contextlib
import datetime as _dt
import shutil
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from cellpose import io, models, train

from .config import Config, load_config
from .data import cache_from_drive, load_dataset
from .mlflow_utils import (
    log_data_provenance,
    log_params_from_config,
    register_model,
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


def run(cfg: Config | None = None, limit: int | None = None,
        n_epochs: int | None = None) -> Path:
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
    if n_epochs is not None:
        # Mutate the config rather than passing it down, so the banner, the
        # train_seg call and the MLflow params all report the same number. A run
        # logged with epochs=100 that actually did 2 is worse than no log.
        cfg.train["n_epochs"] = n_epochs

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
        lazy = bool(cfg.train.get("lazy", False))
        if lazy:
            # Cache image + flows files to local disk once, then let Cellpose
            # read one batch at a time. Peak RAM becomes batch_size images
            # instead of the whole corpus, which is the only way the full
            # 1,386-slice manifest fits on any Colab runtime.
            from .prepare import cache_dir_for, prepare_split
            _banner("2/5", "Data: prepare on-disk cache (lazy loading)")
            device = None
            if gpu:
                import torch
                device = torch.device("cuda")
            prep = {}
            for split, mf in (("train", train_mf), ("test", test_mf)):
                prep[split] = prepare_split(
                    mf, cfg.nmslices_root, cache_dir_for(cfg, split),
                    limit=limit if split == "train"
                    else (max(2, limit // 4) if limit else None),
                    device=device)
                r = prep[split]
                print(f"  {split}: {r['n_usable']}/{r['n_rows']} usable, "
                      f"{r['n_failed']} failed, {r['cache_gib']} GiB cached")
                for cause, n in sorted(r["causes"].items(), key=lambda kv: -kv[1]):
                    print(f"      {n:4d}  {cause}")
            images = labels = test_images = test_labels = None
            train_files = prep["train"]["image_files"]
            train_label_files = prep["train"]["label_files"]
            test_files = prep["test"]["image_files"]
            test_label_files = prep["test"]["label_files"]
            n_train, n_test = len(train_files), len(test_files)
        else:
            train_files = train_label_files = test_files = test_label_files = None
            images, labels = load_manifest(train_mf, cfg.nmslices_root, limit=limit)
            test_images, test_labels = load_manifest(
                test_mf, cfg.nmslices_root,
                limit=max(2, limit // 4) if limit else None)
            n_train, n_test = len(images), len(test_images)
    else:
        lazy = False
        train_files = train_label_files = test_files = test_label_files = None
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
    if not cfg.use_manifest or not lazy:
        n_train, n_test = len(images), len(test_images)
    print(f"  Training images : {n_train}")
    print(f"  Validation images: {n_test}")
    if n_train == 0:
        raise RuntimeError("No training images found. Check params.yaml paths "
                           "and the *_img/_masks filters.")

    # Augment the TRAIN set only. Augmenting the held-out set would mean scoring
    # the model on images it effectively trained on, and would make the test
    # loss incomparable to every previous run.
    from .augment import apply as _augment

    want_hflip = bool(cfg.train.get("augment_hflip", False))
    if lazy:
        # Mirroring works on arrays held in memory, which is exactly what lazy
        # mode does not have. Refuse rather than silently ignoring the setting:
        # a run logged as augmented that was not is a corrupted comparison.
        if want_hflip:
            raise RuntimeError(
                "train.augment_hflip and train.lazy cannot both be true. Lazy "
                "mode streams images from disk, so there is nothing in memory "
                "to mirror. Cellpose already applies random rotation and "
                "rescaling per batch; set augment_hflip: false.")
        aug_info = {"augment_hflip": False, "n_train_real": n_train,
                    "n_train_after_augment": n_train}
    else:
        images, labels, aug_info = _augment(images, labels, hflip=want_hflip)
    if aug_info["augment_hflip"]:
        print(f"  Augmentation    : horizontal flip  "
              f"({aug_info['n_train_real']} -> "
              f"{aug_info['n_train_after_augment']} training images)")

    # ---- STEP 3: MLflow + model --------------------------------------------
    _banner("3/5", "MLflow tracking + load pretrained Cellpose-SAM")
    start_tracking(cfg)
    print(f"  Loading pretrained model: {cfg.model['pretrained']}  (Cellpose-SAM)")
    model = models.CellposeModel(gpu=gpu, pretrained_model=cfg.model["pretrained"])

    # Keep publishing run metadata to Drive while training runs, so a Colab
    # session that dies mid-run does not take the whole history with it.
    if str(cfg.tracking_uri()).startswith("sqlite:"):
        from .mlflow_store import PeriodicCheckpointer
        db_checkpointer: Any = PeriodicCheckpointer(cfg)
    else:
        db_checkpointer = contextlib.nullcontext()

    run_name = f"{cfg.model['name']}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    with db_checkpointer, mlflow.start_run(run_name=run_name) as active_run:
        log_params_from_config(cfg)
        mlflow.log_param("n_train", n_train)
        mlflow.log_param("n_test", n_test)
        mlflow.log_param("lazy", lazy)
        mlflow.log_param("gpu", gpu)
        # Mark smoke runs in the UI. A limited run trains on a fraction of the
        # manifest while log_data_provenance() still stamps it with the full
        # dataset_hash, so without this tag it is indistinguishable from a real
        # one six months later — and it must never be the model that gets
        # registered or gates a release.
        mlflow.set_tag("smoke_run", limit is not None)
        if limit is not None:
            mlflow.log_param("limit", limit)
        # How much of what we trained on was real and how much was generated —
        # without this, two runs with very different effective dataset sizes
        # would be indistinguishable in the MLflow UI.
        for k, v in aug_info.items():
            mlflow.log_param(k, v)
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
            train_files=train_files,
            train_labels_files=train_label_files,
            test_data=test_images,
            test_labels=test_labels,
            test_files=test_files,
            test_labels_files=test_label_files,
            load_files=not lazy,
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
        _banner("5/5", "Save model to Drive, log artifact, register version")
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        drive_model = cfg.model_dir / Path(model_path).name
        shutil.copy2(model_path, drive_model)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_param("model_drive_path", str(drive_model))
        print(f"  Model saved to Drive: {drive_model}")

        # Version the weights in the Model Registry so deploy/ can refer to
        # models:/bluespotter-lc/<n> rather than a mutable path on Drive.
        version = register_model(cfg, active_run.info.run_id, weights_path=drive_model)
        if version:
            mlflow.set_tag("registered_version", version)

    # Publish the run metadata from the local SQLite working copy back to its
    # durable home on Drive. Outside the run context so it also happens if the
    # run itself ended badly.
    if str(cfg.tracking_uri()).startswith("sqlite:"):
        from .mlflow_store import checkpoint_quietly
        _banner("5/5", "Checkpoint MLflow database to Drive")
        checkpoint_quietly(cfg)

    print("\nDONE. Trained model path returned to caller.\n")
    return drive_model


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Fine-tune Cellpose-SAM on the LC dataset.")
    ap.add_argument("--limit", type=int, default=None,
                    help="train on at most N images (smoke run; not releasable)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.n_epochs from params.yaml")
    args = ap.parse_args(argv)
    run(limit=args.limit, n_epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
