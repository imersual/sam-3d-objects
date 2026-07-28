"""Pure request/response helpers for the SAM3D inference server.

Kept free of torch/model imports so they can be unit-tested anywhere.
"""
import math

MAX_VIEWS = 4


def normalize_views(image_path, mask_paths, views):
    """Normalize an /infer request into a list of (image_path, mask_paths) specs.

    Accepts either the legacy single-image form (image_path + mask_paths) or
    the multiview form (views = [{"image_path": ..., "mask_paths": [...]}]).
    views takes precedence when provided. Raises ValueError on malformed input.
    """
    if views:
        if len(views) > MAX_VIEWS:
            raise ValueError(
                f"views supports at most {MAX_VIEWS} entries, got {len(views)}"
            )
        specs = []
        for i, view in enumerate(views):
            if not view.get("image_path"):
                raise ValueError(f"views[{i}] is missing image_path")
            if not view.get("mask_paths"):
                raise ValueError(f"views[{i}] needs at least one mask_path")
            specs.append((view["image_path"], list(view["mask_paths"])))
        return specs

    if not image_path:
        raise ValueError("either views or image_path must be provided")
    if not mask_paths:
        raise ValueError("mask_paths must not be empty")
    return [(image_path, list(mask_paths))]


def _flatten(value):
    """Flatten arbitrarily nested lists/tuples into a flat list of leaves."""
    if not isinstance(value, (list, tuple)):
        return [value]
    leaves = []
    for item in value:
        leaves.extend(_flatten(item))
    return leaves


def extract_metric_scale(output):
    """Pull the predicted real-world size out of a SAM3D result dict.

    SAM3D emits the mesh in a normalized [-0.5, 0.5] cube, so its longest side
    is always 1.0 regardless of the real object. The real-world size is
    returned separately as `scale` (metres per cube unit), decoded against
    MoGe-2's metric pointmap and then refined by the layout post-optimization.
    Multiplying the mesh by it makes the export metric.

    The three components are forced equal upstream (the decoder averages them
    and `refine_scale()` collapses them), so a single float is enough and no
    axis convention is involved.

    Returns a positive float, or None when the value is missing or unusable.
    """
    # The pipeline sets this to False when it decoded a scale it could not tie
    # to metric depth (SSI units, not metres). Baking that in would be worse
    # than leaving the mesh unit-cube sized, because it looks real.
    if output.get("scale_is_metric") is False:
        return None

    scale = output.get("scale")
    if scale is None:
        return None

    if hasattr(scale, "detach"):  # torch tensor
        scale = scale.detach().cpu()
    if hasattr(scale, "reshape") and hasattr(scale, "tolist"):  # tensor / ndarray
        scale = scale.reshape(-1).tolist()
    values = _flatten(scale)

    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if not values:
        return None

    mean = sum(values) / len(values)
    if not math.isfinite(mean) or mean <= 0:
        return None
    return mean
