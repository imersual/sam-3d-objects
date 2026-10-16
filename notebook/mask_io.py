"""Loading the photo and mask files a reconstruction request points at.

Kept apart from inference.py, which imports the whole GPU stack at import
time, so the mask rules can be unit-tested anywhere. inference.py re-exports
both functions, so callers keep importing them from there.
"""
import numpy as np
from PIL import Image


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def mask_threshold(values):
    """The value a mask pixel must exceed to count as object.

    Masks arrive as 0/255, including JPEG ones whose compression leaves a
    fringe of near-zero values around every edge, and occasionally as 0/1
    (palette or bilevel files). Splitting at the midpoint of whichever range
    the mask uses keeps that fringe out without emptying a 0/1 mask.
    """
    return 0 if values.max() <= 1 else 127


def load_mask(path):
    # Return a (H, W) bool mask from any common encoding:
    #   - L / P (2D grayscale)          -> pixels above the midpoint
    #   - RGBA / LA with a VARYING alpha -> alpha is the cutout (e.g. notebook
    #                                       sample masks: object RGB on transparent bg)
    #   - RGBA / LA with a CONSTANT alpha (fully opaque) -> alpha carries no
    #     mask info, so the mask lives in the color channels instead. Blindly
    #     taking the last channel here would mark the WHOLE image as object and
    #     make SAM3D reconstruct the entire scene. Fall back to the color
    #     channels (any bright channel) in that case.
    #   - RGB (no alpha)                -> any bright channel
    #
    # "Above the midpoint", not "nonzero": see mask_threshold. A nonzero test
    # turned a JPEG mask's compression fringe into a speckled halo of extra
    # object pixels around every edge.
    mask = load_image(path)
    if mask.ndim == 2:
        return mask > mask_threshold(mask)

    n = mask.shape[-1]
    if n in (2, 4):  # has an alpha channel (LA or RGBA)
        alpha = mask[..., -1]
        if alpha.min() != alpha.max():
            return alpha > mask_threshold(alpha)
        color = mask[..., : n - 1]  # opaque alpha -> mask is in the color channels
    else:
        color = mask[..., :3]
    brightest = color.max(axis=-1)
    return brightest > mask_threshold(brightest)
