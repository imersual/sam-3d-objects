"""Pure request-normalization helpers for the SAM3D inference server.

Kept free of torch/model imports so they can be unit-tested anywhere.
"""

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
