"""Augmentation has to preserve the thing it is augmenting. The failure modes
worth pinning are silent ones: a mask that no longer sits on its image, instance
IDs mangled by the flip, or augmentation leaking into the held-out set."""
from __future__ import annotations

import numpy as np
import pytest

from bluespotter.augment import apply, mirror_horizontal


def _image_and_mask():
    # Deliberately asymmetric, so a flip is detectable and a no-op is not
    # mistaken for success.
    img = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=np.uint8)
    # Two instances with distinct IDs — Cellpose reads each integer as one cell.
    msk = np.array([[1, 1, 0],
                    [0, 2, 2]], dtype=np.int32)
    return img, msk


def test_mirror_appends_rather_than_replaces():
    img, msk = _image_and_mask()
    images, labels = mirror_horizontal([img], [msk])

    assert len(images) == len(labels) == 2
    np.testing.assert_array_equal(images[0], img)   # original still first
    np.testing.assert_array_equal(labels[0], msk)


def test_image_and_mask_are_flipped_the_same_way():
    # If these two ever diverge, every neuron is labelled as its mirror-image
    # neighbour and the loss quietly optimises for nonsense.
    img, msk = _image_and_mask()
    images, labels = mirror_horizontal([img], [msk])

    np.testing.assert_array_equal(images[1], np.fliplr(img))
    np.testing.assert_array_equal(labels[1], np.fliplr(msk))


def test_instance_ids_survive_the_flip():
    # A flip must reorder pixels, never renumber them: label 2 has to stay
    # label 2, or two cells become one.
    _, msk = _image_and_mask()
    _, labels = mirror_horizontal([msk], [msk])
    assert set(np.unique(labels[1])) == set(np.unique(msk))


def test_flip_is_horizontal_not_vertical():
    # Vertical flip is anatomically wrong for the LC — dorsal and ventral are
    # not interchangeable — so guard against the axis being swapped by mistake.
    img, msk = _image_and_mask()
    images, _ = mirror_horizontal([img], [msk])
    np.testing.assert_array_equal(images[1], np.array([[3, 2, 1], [6, 5, 4]]))
    assert not np.array_equal(images[1], np.flipud(img))


def test_disabled_by_default_is_a_no_op():
    img, msk = _image_and_mask()
    images, _labels, info = apply([img], [msk])
    assert len(images) == 1
    assert info == {"augment_hflip": False,
                    "n_train_real": 1,
                    "n_train_after_augment": 1}


def test_reports_real_versus_generated_counts():
    # These land in MLflow. Without them a run trained on 1386 real images looks
    # identical to one trained on 693 real plus 693 mirrored.
    img, msk = _image_and_mask()
    _, _, info = apply([img, img], [msk, msk], hflip=True)
    assert info["n_train_real"] == 2
    assert info["n_train_after_augment"] == 4


def test_mismatched_lengths_fail_loudly():
    img, msk = _image_and_mask()
    with pytest.raises(ValueError, match="same length"):
        mirror_horizontal([img, img], [msk])
