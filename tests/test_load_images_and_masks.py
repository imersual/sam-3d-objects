"""CPU-only tests for the multi-view image/mask loader."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "notebook"))

from load_images_and_masks import load_images_and_masks_from_path


def _write_image(path, size=(4, 4)):
    arr = np.full(size + (3,), 128, dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _write_rgba_mask(path, size=(4, 4)):
    arr = np.zeros(size + (4,), dtype=np.uint8)
    arr[1:3, 1:3, 3] = 255  # small object in the alpha channel
    Image.fromarray(arr, "RGBA").save(path)


def _write_grayscale_mask(path, size=(4, 4)):
    arr = np.zeros(size, dtype=np.uint8)
    arr[1:3, 1:3] = 255
    Image.fromarray(arr, "L").save(path)


def test_flat_layout(tmp_path):
    for stem in ["1", "2", "view_a"]:
        _write_image(tmp_path / f"{stem}.png")
        _write_rgba_mask(tmp_path / f"{stem}_mask.png")
    images, masks, names = load_images_and_masks_from_path(tmp_path)
    assert names == ["1", "2", "view_a"]
    assert len(images) == len(masks) == 3
    assert masks[0].dtype == bool and masks[0].shape == (4, 4)
    assert masks[0].sum() == 4  # 2x2 object


def test_split_layout_with_mask_prompt(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "toy").mkdir()
    _write_image(tmp_path / "images" / "1.png")
    _write_rgba_mask(tmp_path / "toy" / "1.png")  # plain stem name
    _write_image(tmp_path / "images" / "2.png")
    _write_rgba_mask(tmp_path / "toy" / "2_mask.png")  # _mask suffix name
    images, masks, names = load_images_and_masks_from_path(tmp_path, mask_prompt="toy")
    assert names == ["1", "2"]


def test_image_names_filter(tmp_path):
    for stem in ["1", "2", "3"]:
        _write_image(tmp_path / f"{stem}.png")
        _write_rgba_mask(tmp_path / f"{stem}_mask.png")
    images, masks, names = load_images_and_masks_from_path(
        tmp_path, image_names=["1", "3"]
    )
    assert names == ["1", "3"]


def test_missing_mask_is_skipped_with_warning(tmp_path):
    _write_image(tmp_path / "1.png")
    _write_rgba_mask(tmp_path / "1_mask.png")
    _write_image(tmp_path / "2.png")  # no mask for view 2
    images, masks, names = load_images_and_masks_from_path(tmp_path)
    assert names == ["1"]


def test_no_valid_pairs_raises(tmp_path):
    _write_image(tmp_path / "1.png")  # no masks at all
    with pytest.raises(ValueError):
        load_images_and_masks_from_path(tmp_path)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_images_and_masks_from_path(tmp_path / "nope")


def test_grayscale_mask_fallback(tmp_path):
    _write_image(tmp_path / "1.png")
    _write_grayscale_mask(tmp_path / "1_mask.png")
    _write_image(tmp_path / "2.png")
    _write_rgba_mask(tmp_path / "2_mask.png")
    images, masks, names = load_images_and_masks_from_path(tmp_path)
    assert names == ["1", "2"]
    assert masks[0].sum() == 4  # grayscale mask decoded via nonzero pixels
