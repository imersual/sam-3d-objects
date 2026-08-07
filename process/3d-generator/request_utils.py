"""Pure request/response helpers for the SAM3D inference server.

Kept free of torch/model imports so they can be unit-tested anywhere.
"""
import math
from pathlib import PurePath

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


def extract_intrinsics(output):
    """Pull the camera intrinsics matrix out of a SAM3D result dict, if any.

    MoGe-2 predicts this as part of ordinary per-image inference (see
    `InferencePipelinePointMap.compute_pointmap`, sam3d_objects/pipeline/
    inference_pipeline_pointmap.py); when the depth model doesn't supply one
    (e.g. an externally supplied pointmap), `infer_intrinsics_from_pointmap`
    fills in an equivalent one. Either way it is the standard OpenCV-style
    matrix

        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    and NORMALIZED, not pixel-space: fx/cx are fractions of image width, fy/cy
    are fractions of image height (cx = cy = 0.5 always -- MoGe assumes a
    centered principal point; see utils3d's `intrinsics_from_focal_center`,
    whose docstring calls this "OpenCV intrinsics" and whose `project_cv`
    consumer documents its output as "uv coordinates, value ranging in
    [0, 1]"). To convert to pixel units, multiply the first row by the image
    width and the second row by the image height:

        fx_px, cx_px = fx * width,  cx * width
        fy_px, cy_px = fy * height, cy * height

    This is MoGe's own camera-space convention (X right, Y down, Z forward --
    see batch/run_moge2.py's documented convention, and compute_pointmap's
    `normal`, which deliberately shares it). Whether that also matches the
    axis convention `extract_pose`'s rotation/translation are decoded in
    (SAM3D rotates points into a PyTorch3D convention for its own
    diffusion-model conditioning before the pose decoder ever sees them) was
    NOT verified here -- do not assume `pose` and `intrinsics` compose
    without checking.

    Pass the raw pipeline output dict for a single-view request. Pass None
    (or {}) for a multiview result, for the same reason `extract_pose` does:
    `run_multi_view` computes one intrinsics matrix per view internally and
    discards all of them (see InferencePipeline.run_multi_view), rather than
    picking one view's frame arbitrarily.

    Returns a 3x3 nested list of floats (row-major, matching the matrix
    above), or None when missing or unusable. Plain JSON-serialisable types
    only, safe to return straight from the /infer response.
    """
    output = output or {}
    flat = _to_flat_float_list(output.get("intrinsics"))
    if not flat or len(flat) != 9:
        return None
    if not all(math.isfinite(v) for v in flat):
        return None
    return [flat[0:3], flat[3:6], flat[6:9]]


def normal_map_sidecar_path(output_path):
    """Derive the on-disk path for an optional dumped normal map.

    Mirrors the caller-provided `output_path` (the exported GLB destination):
    same directory, same stem, `_normal.npy` suffix -- e.g.
    "out/mesh.glb" -> "out/mesh_normal.npy". Pure path-string manipulation
    (no filesystem access), so it's unit-testable without the torch/numpy
    imports the actual save (server.py) requires.

    Always returns a forward-slash ("/") path via `PurePath.as_posix()`: the
    server only ever runs on the same (Linux) machine as its caller, so this
    matches production exactly, and stays unambiguous when this module's
    tests run on a different OS.
    """
    p = PurePath(str(output_path))
    return p.with_name(p.stem + "_normal.npy").as_posix()
