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


def _to_flat_float_list(value):
    """Flatten a torch tensor / numpy array / (nested) list / bare number
    into a flat list of Python floats.

    Returns None when `value` is None, or when any element can't be
    converted to float (e.g. a stray string). SAM3D's pose/scale outputs are
    small tensors with a stray batch dim or two, so this accepts anything
    tensor-*like* (has .detach()/.reshape()/.tolist()) without this module
    depending on torch actually being importable.
    """
    if value is None:
        return None
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):  # tensor / ndarray
        value = value.reshape(-1).tolist()
    values = _flatten(value)

    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


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

    values = _to_flat_float_list(output.get("scale"))
    if not values:
        return None

    mean = sum(values) / len(values)
    if not math.isfinite(mean) or mean <= 0:
        return None
    return mean


def extract_pose(output):
    """Pull the camera-frame pose out of a SAM3D result dict, if any.

    With `with_layout_postprocess=True`, SAM3D solves the object's pose in
    the SOURCE PHOTO's metric camera frame:
    `InferencePipelinePointMap.run` (sam3d_objects/pipeline/
    inference_pipeline_pointmap.py) decodes an initial rotation/translation/
    scale via the pose decoder, then -- inside a try/except -- refines them
    with `run_post_optimization` / `run_post_optimization_GS`.

    Pass the raw pipeline output dict for a single-view request. Pass None
    (or {}) for a multiview result: `run_multi_view` never attempts layout
    postprocess (it would have to pick one view's frame arbitrarily), so
    every field below comes back None rather than inventing a pose from an
    arbitrary view.

    `iou` tells you how much to trust rotation/translation/scale:
      - a float (typically in [0, 1]): layout post-optimization ran to
        completion and this is its rendered-mask-vs-image-mask IoU.
      - exactly -1.0: post-optimization was SKIPPED -- its occlusion check,
        or its alignment step finding no target points, bailed out. rotation/
        translation/scale are still returned, but are the *unrefined*
        pose estimate straight out of the pose decoder.
      - None: post-optimization was never attempted for this result (no mesh
        or gaussian to optimize -- shouldn't happen for a successful
        single-view request), or it raised an exception (caught and logged
        by the pipeline, not re-raised). rotation/translation/scale may
        still be present in this case too (again the unrefined estimate)
        because the pose decoder runs before post-optimization is attempted.

    Returns a dict with keys:
      rotation        4 floats (quaternion, local -> camera), or None
      translation     3 floats (metres, camera frame), or None
      scale           3 floats (metres per grid unit -- see extract_metric_scale
                       for the single number baked into the exported mesh),
                       or None
      scale_is_metric bool, or None when absent/not a bool
      iou             float, or None -- see above

    Every value is independently None when missing or unusable; nothing here
    is fabricated. Plain JSON-serialisable types only (lists of float / bool
    / float / None), safe to return straight from the /infer response.
    """
    output = output or {}

    scale_is_metric = output.get("scale_is_metric")
    if not isinstance(scale_is_metric, bool):
        scale_is_metric = None

    iou = output.get("iou")
    if iou is not None:
        try:
            iou = float(iou)
        except (TypeError, ValueError):
            iou = None
        else:
            if not math.isfinite(iou):
                iou = None

    return {
        "rotation": _to_flat_float_list(output.get("rotation")),
        "translation": _to_flat_float_list(output.get("translation")),
        "scale": _to_flat_float_list(output.get("scale")),
        "scale_is_metric": scale_is_metric,
        "iou": iou,
    }
