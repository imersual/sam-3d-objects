"""CPU-only tests for notebook/mask_io.py, the loader /infer reads masks with."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "notebook"))

from mask_io import load_mask, mask_threshold  # noqa: E402


def _disc(size=64, radius=20):
    yy, xx = np.mgrid[:size, :size]
    return (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius**2


def _grow(mask, steps):
    """The mask grown by `steps` pixels (8-connected), numpy only."""
    grown = mask.copy()
    for _ in range(steps):
        padded = np.pad(grown, 1)
        grown = np.zeros_like(grown)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                grown |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return grown


def test_a_jpeg_masks_compression_fringe_is_not_object(tmp_path):
    """Production masks used to be JPEGs. Their ringing puts near-zero values
    up to a block away from every edge; a nonzero test counted those as
    object and gave every mask a speckled halo."""
    truth = _disc()
    path = tmp_path / "object_1.jpg"
    Image.fromarray(truth.astype(np.uint8) * 255, "L").save(path, quality=75)

    raw = np.array(Image.open(path))
    far = ~_grow(truth, 2)
    assert (raw[far] > 0).any(), "the fixture must actually have a fringe"

    mask = load_mask(path)
    assert not mask[far].any()
    assert mask[~_grow(~truth, 2)].all(), "the object itself must survive"


def test_a_png_mask_loads_exactly(tmp_path):
    truth = _disc()
    path = tmp_path / "object_1.png"
    Image.fromarray(truth.astype(np.uint8) * 255, "L").save(path)
    assert np.array_equal(load_mask(path), truth)


def test_a_zero_one_mask_is_not_emptied(tmp_path):
    truth = _disc()
    path = tmp_path / "mask.png"
    Image.fromarray(truth.astype(np.uint8), "L").save(path)  # values 0 and 1
    assert np.array_equal(load_mask(path), truth)


def test_a_bilevel_mask_loads(tmp_path):
    truth = _disc()
    path = tmp_path / "mask.png"
    Image.fromarray(truth).convert("1").save(path)
    assert np.array_equal(load_mask(path), truth)


def test_an_rgba_cutout_uses_its_alpha(tmp_path):
    truth = _disc()
    rgba = np.zeros(truth.shape + (4,), np.uint8)
    rgba[..., :3] = 200  # colour everywhere; only alpha says where the object is
    rgba[..., 3] = truth.astype(np.uint8) * 255
    path = tmp_path / "cutout.png"
    Image.fromarray(rgba, "RGBA").save(path)
    assert np.array_equal(load_mask(path), truth)


def test_an_opaque_rgba_mask_uses_its_colour(tmp_path):
    truth = _disc()
    rgba = np.zeros(truth.shape + (4,), np.uint8)
    rgba[..., 0] = truth.astype(np.uint8) * 255
    rgba[..., 3] = 255
    path = tmp_path / "opaque.png"
    Image.fromarray(rgba, "RGBA").save(path)
    assert np.array_equal(load_mask(path), truth)


def test_an_rgb_mask_uses_its_brightest_channel(tmp_path):
    truth = _disc()
    rgb = np.zeros(truth.shape + (3,), np.uint8)
    rgb[..., 2] = truth.astype(np.uint8) * 255
    path = tmp_path / "rgb.png"
    Image.fromarray(rgb, "RGB").save(path)
    assert np.array_equal(load_mask(path), truth)


def test_mask_threshold_splits_each_range_at_its_midpoint():
    assert mask_threshold(np.array([0, 255])) == 127
    assert mask_threshold(np.array([0, 3, 255])) == 127
    assert mask_threshold(np.array([0, 1])) == 0
    assert mask_threshold(np.array([True, False])) == 0
