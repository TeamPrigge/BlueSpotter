"""Visual QC for BlueSpotter: learning-rate schedule + predicted-vs-GT overlays."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .config import Config, load_config
from .manifest import load_manifest


# ---------------------------------------------------------------- LR schedule
def cellpose_lr_schedule(n_epochs: int, learning_rate: float) -> np.ndarray:
    """Reproduce Cellpose's internal LR schedule (10-epoch warmup + step decay)."""
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for _ in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))
    return LR


def plot_lr(cfg: Config | None = None):
    cfg = cfg or load_config()
    LR = cellpose_lr_schedule(cfg.train["n_epochs"], cfg.train["learning_rate"])
    plt.figure(figsize=(7, 3.2))
    plt.plot(LR, lw=2)
    plt.xlabel("epoch"); plt.ylabel("learning rate")
    plt.title(f"Cellpose LR schedule  (n_epochs={cfg.train['n_epochs']}, "
              f"base lr={cfg.train['learning_rate']})")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
    return LR


# ------------------------------------------------------- prediction overlays
def _disp(img) -> np.ndarray:
    """Turn any image array into a 1%-99% stretched 2D grayscale for display."""
    a = np.asarray(img).astype(float)
    if a.ndim == 3:
        a = np.moveaxis(a, int(np.argmin(a.shape)), 0)[0]  # channel-first, take ch0
    mn, mx = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - mn) / (mx - mn), 0, 1) if mx > mn else a


def _latest_model(cfg: Config) -> Path:
    md = cfg.model_dir
    cands = [p for p in md.glob("*") if p.is_file()] if md.exists() else []
    if not cands:
        raise FileNotFoundError(f"No model found in {md}; pass model= or model_path=.")
    return max(cands, key=lambda p: p.stat().st_mtime)


def plot_predictions(cfg: Config | None = None, model=None, model_path=None,
                     n: int = 4, seed: int = 0):
    """Overlay model masks (red, dashed) on ground truth (green) for n random
    crops of each split. Green = hand label, red = model prediction."""
    cfg = cfg or load_config()
    from cellpose import models
    from cellpose.utils import outlines_list

    if model is None:
        mp = model_path or _latest_model(cfg)
        print(f"Loading model: {mp}")
        model = models.CellposeModel(gpu=True, pretrained_model=str(mp))

    rng = random.Random(seed)
    for split, csv in [("train", cfg.train_manifest), ("test", cfg.test_manifest)]:
        imgs, lbls = load_manifest(csv, cfg.nmslices_root)
        idx = rng.sample(range(len(imgs)), min(n, len(imgs)))
        fig, axes = plt.subplots(1, len(idx), figsize=(4 * len(idx), 4.2))
        axes = np.atleast_1d(axes)
        for ax, i in zip(axes, idx):
            img, gt = imgs[i], np.asarray(lbls[i])
            pred = np.asarray(model.eval(img)[0])
            ax.imshow(_disp(img), cmap="gray")
            for o in outlines_list(gt):
                ax.plot(o[:, 0], o[:, 1], color="lime", lw=1.0)
            for o in outlines_list(pred):
                ax.plot(o[:, 0], o[:, 1], color="red", lw=1.0, ls="--")
            ax.set_title(f"GT={int(gt.max())}  pred={int(pred.max())}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{split.upper()}  —  green = ground truth,  red = model", fontsize=12)
        plt.tight_layout(); plt.show()
