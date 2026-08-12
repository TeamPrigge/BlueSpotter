"""Visual QC for BlueSpotter: learning-rate schedule + predicted-vs-GT overlays."""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Config, load_config
from .manifest import load_manifest  # noqa: F401  (re-exported for notebooks)


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
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title(f"Cellpose LR schedule  (n_epochs={cfg.train['n_epochs']}, "
              f"base lr={cfg.train['learning_rate']})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
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


def _sample_pairs(csv_path, nm_root, n, rng):
    """Load at most n random (image, mask) pairs from a manifest.

    Deliberately does not go through load_manifest: that returns the whole
    split, which for this dataset is far more than RAM. Unreadable rows are
    skipped quietly here — a broken slice should cost you one panel, not the
    figure, and load_manifest already reports them properly at training time.
    """
    import csv as _csv

    from .manifest import load_pair

    with open(csv_path, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    rng.shuffle(rows)

    pairs = []
    for row in rows:
        if len(pairs) >= n:
            break
        try:
            pairs.append(load_pair(nm_root, row))
        except Exception:
            continue
    return pairs


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
        # Choose the rows FIRST, then load only those. Loading the manifest and
        # then sampling pulled all 1,386 training slices into RAM to draw four
        # pictures — enough to OOM the VM and take the MLflow database with it.
        pairs = _sample_pairs(csv, cfg.nmslices_root, n, rng)
        if not pairs:
            print(f"  {split}: no readable pairs to plot")
            continue
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4.2))
        axes = np.atleast_1d(axes)
        for ax, (img, gt) in zip(axes, pairs, strict=False):
            gt = np.asarray(gt)
            pred = np.asarray(model.eval(img)[0])
            ax.imshow(_disp(img), cmap="gray")
            for o in outlines_list(gt):
                ax.plot(o[:, 0], o[:, 1], color="lime", lw=1.0)
            for o in outlines_list(pred):
                ax.plot(o[:, 0], o[:, 1], color="red", lw=1.0, ls="--")
            ax.set_title(f"GT={int(gt.max())}  pred={int(pred.max())}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{split.upper()}  —  green = ground truth,  red = model", fontsize=12)
        plt.tight_layout()
        plt.show()
