# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Prune Gaussian Splats to reduce file size by removing low-opacity Gaussians.
This is a simple and fast method that typically reduces file size by 80-90%.
"""

import torch


def prune_gaussians_by_opacity(gaussian, opacity_threshold=0.05, verbose=True):
    """
    Prune Gaussians with low opacity to reduce file size.

    Args:
        gaussian: Gaussian object from the model output
        opacity_threshold: Remove Gaussians with opacity below this value (default: 0.05)
        verbose: Print statistics about pruning

    Returns:
        Pruned Gaussian object (modified in-place)
    """
    # Get current opacity values
    opacity = gaussian.get_opacity.squeeze()

    # Count original Gaussians
    original_count = opacity.shape[0]

    # Create mask for Gaussians to keep (above threshold)
    mask = opacity > opacity_threshold
    mask_indices = torch.nonzero(mask).squeeze()

    # Prune all Gaussian parameters
    gaussian._xyz = gaussian._xyz[mask_indices]
    gaussian._rotation = gaussian._rotation[mask_indices]
    gaussian._scaling = gaussian._scaling[mask_indices]
    gaussian._opacity = gaussian._opacity[mask_indices]
    gaussian._features_dc = gaussian._features_dc[mask_indices]

    if gaussian._features_rest is not None:
        gaussian._features_rest = gaussian._features_rest[mask_indices]

    # Count remaining Gaussians
    remaining_count = mask_indices.shape[0] if mask_indices.dim() > 0 else 1
    removed_count = original_count - remaining_count
    reduction_percent = (removed_count / original_count) * 100

    if verbose:
        print(f"\n{'='*60}")
        print(f"Gaussian Pruning Results:")
        print(f"{'='*60}")
        print(f"  Original Gaussians:  {original_count:,}")
        print(f"  Remaining Gaussians: {remaining_count:,}")
        print(f"  Removed Gaussians:   {removed_count:,}")
        print(f"  Reduction:           {reduction_percent:.1f}%")
        print(f"  Opacity Threshold:   {opacity_threshold}")
        print(f"{'='*60}\n")

    return gaussian


def prune_gaussians_by_count(gaussian, target_count, verbose=True):
    """
    Prune Gaussians to a target count by removing lowest opacity Gaussians.

    Args:
        gaussian: Gaussian object from the model output
        target_count: Target number of Gaussians to keep
        verbose: Print statistics about pruning

    Returns:
        Pruned Gaussian object (modified in-place)
    """
    # Get current opacity values
    opacity = gaussian.get_opacity.squeeze()

    # Count original Gaussians
    original_count = opacity.shape[0]

    if target_count >= original_count:
        if verbose:
            print(
                f"Target count ({target_count}) >= current count ({original_count}). No pruning needed."
            )
        return gaussian

    # Get indices of top-k highest opacity Gaussians
    _, top_indices = torch.topk(opacity, k=target_count, largest=True, sorted=False)

    # Prune all Gaussian parameters
    gaussian._xyz = gaussian._xyz[top_indices]
    gaussian._rotation = gaussian._rotation[top_indices]
    gaussian._scaling = gaussian._scaling[top_indices]
    gaussian._opacity = gaussian._opacity[top_indices]
    gaussian._features_dc = gaussian._features_dc[top_indices]

    if gaussian._features_rest is not None:
        gaussian._features_rest = gaussian._features_rest[top_indices]

    # Calculate statistics
    removed_count = original_count - target_count
    reduction_percent = (removed_count / original_count) * 100
    min_opacity = opacity[top_indices].min().item()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Gaussian Pruning Results (Count-Based):")
        print(f"{'='*60}")
        print(f"  Original Gaussians:  {original_count:,}")
        print(f"  Target Gaussians:    {target_count:,}")
        print(f"  Removed Gaussians:   {removed_count:,}")
        print(f"  Reduction:           {reduction_percent:.1f}%")
        print(f"  Min Opacity Kept:    {min_opacity:.4f}")
        print(f"{'='*60}\n")

    return gaussian
