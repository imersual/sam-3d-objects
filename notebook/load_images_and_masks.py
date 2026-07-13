"""Load multi-view images and masks (ported from upstream PR #37, cleaned up).

Two supported layouts:

1. ``mask_prompt=None`` — images and masks side by side in one directory::

       input_path/1.png, input_path/1_mask.png, input_path/2.png, ...

2. ``mask_prompt="stuffed_toy"`` — images in ``input_path/images/``, masks
   in ``input_path/stuffed_toy/`` named ``<stem>.png`` or ``<stem>_mask.png``.

Masks are RGBA with the mask in the alpha channel (alpha > 0 = object).
Grayscale masks use nonzero pixels; RGB masks fall back to "any non-black
pixel" with a warning (same semantics as ``notebook/inference.py``'s
``load_mask`` — NOT the PR's all-ones fallback, which would select the
whole image).

This module deliberately imports no torch/GPU dependencies so batch
scripts and tests can import it cheaply.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

IMAGE_EXTENSIONS = (".png", ".jpg")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    """Return a (H, W) bool mask, reading the alpha channel when present."""
    img = Image.open(path)
    arr = np.array(img)
    if img.mode == "RGBA" and arr.ndim == 3 and arr.shape[2] >= 4:
        return arr[..., 3] > 0
    if arr.ndim == 2:
        return arr > 0
    logger.warning(
        f"Mask {path} has no alpha channel (mode={img.mode}); "
        "using any non-black pixel as the mask."
    )
    return arr[..., :3].max(axis=-1) > 0


def _find_file(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_pairs(
    images_dir: Path,
    masks_dir: Path,
    image_names: Optional[List[str]],
    same_dir: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    if image_names is None:
        image_files = sorted(
            f
            for ext in IMAGE_EXTENSIONS
            for f in images_dir.glob(f"*{ext}")
            if "_mask" not in f.name
        )
        image_names = [f.stem for f in image_files]
        logger.info(f"Auto-detected {len(image_names)} images: {image_names}")

    images, masks, loaded_names = [], [], []
    for name in image_names:
        image_path = _find_file(
            [images_dir / f"{name}{ext}" for ext in IMAGE_EXTENSIONS]
        )
        if same_dir:
            mask_candidates = [
                masks_dir / f"{name}_mask{ext}" for ext in IMAGE_EXTENSIONS
            ]
        else:
            mask_candidates = [
                masks_dir / f"{name}{ext}" for ext in IMAGE_EXTENSIONS
            ] + [masks_dir / f"{name}_mask{ext}" for ext in IMAGE_EXTENSIONS]
        mask_path = _find_file(mask_candidates)

        if image_path is None:
            logger.warning(f"Image file not found for '{name}', skipping")
            continue
        if mask_path is None:
            logger.warning(f"Mask file not found for '{name}', skipping")
            continue

        images.append(load_image(image_path))
        masks.append(load_mask(mask_path))
        loaded_names.append(name)
        logger.info(
            f"Loaded '{name}': image={images[-1].shape}, mask={masks[-1].shape}"
        )

    return images, masks, loaded_names


def load_images_and_masks_from_path(
    input_path: Path,
    mask_prompt: Optional[str] = None,
    image_names: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Load view images and bool masks; returns (images, masks, names)."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_path}")

    if mask_prompt is None:
        logger.info(f"Loading images and masks from single directory: {input_path}")
        images, masks, names = _load_pairs(
            input_path, input_path, image_names, same_dir=True
        )
    else:
        images_dir = input_path / "images"
        masks_dir = input_path / mask_prompt
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {masks_dir}")
        logger.info(f"Loading images from {images_dir}, masks from {masks_dir}")
        images, masks, names = _load_pairs(
            images_dir, masks_dir, image_names, same_dir=False
        )

    if len(images) == 0:
        raise ValueError(f"No valid image/mask pairs found in {input_path}")
    logger.info(f"Successfully loaded {len(images)} views: {names}")
    return images, masks, names
