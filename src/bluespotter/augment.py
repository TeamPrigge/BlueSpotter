"""Training-time augmentation.

WHY THIS IS SEPARATE FROM THE MANIFESTS
---------------------------------------
The tempting shortcut is to add flipped copies as extra rows in `train.csv`.
Don't. Manifest rows are the unit the whole pipeline counts, hashes and splits
by: 1,634 pairs would become 6,536, every per-image metric would silently change
meaning, and `dataset_hash` would stop describing what is actually on Drive.
Worse, a flipped copy is not a new observation — scoring on it would be scoring
on training data wearing a hat.

Augmentation is a property of a *training run*, not of the dataset. So it lives
here, is switched on in `params.yaml`, and is logged to MLflow, which means two
runs that differ only in augmentation are directly comparable.

WHY HORIZONTAL ONLY
-------------------
The locus coeruleus is bilateral and close to mirror-symmetric about the
midline, so a mirrored left LC is a plausible right LC — the label stays true.

A *vertical* (dorsoventral) flip is not. It would put the fourth ventricle and
the dorsal surface of the brainstem in anatomically impossible places. For
segmenting cell bodies that does no harm, because a neuron looks like a neuron
either way. But LC-Seg exists to do anatomically aware analysis — dorsoventral
localisation and rostrocaudal position are stated objectives — and a model
trained to treat up and down as interchangeable cannot learn either. The cost of
the shortcut lands on the part of the platform that matters most, so this module
deliberately offers no vertical flip.

WHAT THIS COSTS
---------------
Cellpose's `train_seg` runs its own random rotate/rescale per batch but exposes
no hook for adding a reflection, so mirroring is applied once, up front, and the
arrays are held in memory at double size. On the current 1,386-pair training set
that is the trade to be aware of before switching it on.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def mirror_horizontal(images: list[Any], labels: list[Any]) -> tuple[list, list]:
    """Return (images + mirrored, labels + mirrored), preserving pairing order.

    The mask is flipped with exactly the same operation as its image, so a
    neuron and its label stay on top of each other. `np.fliplr` reorders pixels
    without touching their values, so instance IDs in the label image survive
    unchanged — which matters, because Cellpose treats each integer as one cell.
    """
    if len(images) != len(labels):
        raise ValueError(
            f"images and labels must be the same length, got {len(images)} and {len(labels)}"
        )

    flipped_images = [np.fliplr(np.asarray(im)) for im in images]
    flipped_labels = [np.fliplr(np.asarray(lb)) for lb in labels]
    return images + flipped_images, labels + flipped_labels


def apply(images: list[Any], labels: list[Any], hflip: bool = False) -> tuple[list, list, dict]:
    """Apply the configured augmentations. Returns (images, labels, info).

    `info` is what gets logged to MLflow, so a run always records how much of
    what it trained on was real and how much was generated.
    """
    n_real = len(images)
    if hflip:
        images, labels = mirror_horizontal(images, labels)
    return images, labels, {
        "augment_hflip": bool(hflip),
        "n_train_real": n_real,
        "n_train_after_augment": len(images),
    }
