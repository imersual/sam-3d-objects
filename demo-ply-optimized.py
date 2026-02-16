# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import os

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask, load_single_mask
from prune_gaussians import prune_gaussians_by_opacity, prune_gaussians_by_count

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/sofa/sofa.jpeg")
mask = load_single_mask("notebook/images/sofa", index=1)

# run model
output = inference(
    image,
    mask,
    seed=42,
    rendering_engine="nvdiffrast",
)

# Get file size before pruning
output["gs"].save_ply(f"splat_original.ply")
original_size = os.path.getsize("splat_original.ply") / (1024 * 1024)  # MB
print(f"Original PLY file size: {original_size:.2f} MB")

# Method 1: Prune by opacity threshold (removes low-opacity Gaussians)
# This typically achieves 80-90% file size reduction
print("\n" + "=" * 60)
print("METHOD 1: Pruning by Opacity Threshold")
print("=" * 60)
gs_pruned_opacity = prune_gaussians_by_opacity(
    output["gs"],
    opacity_threshold=0.1,  # Adjust this: higher = more aggressive pruning
    verbose=True,
)
gs_pruned_opacity.save_ply(f"splat_pruned_opacity.ply")
pruned_size = os.path.getsize("splat_pruned_opacity.ply") / (1024 * 1024)  # MB
print(f"Pruned PLY file size: {pruned_size:.2f} MB")
print(f"Size reduction: {((original_size - pruned_size) / original_size * 100):.1f}%")

# Reload for Method 2 (since we modified in-place)
output2 = inference(image, mask, seed=42, rendering_engine="nvdiffrast")

# Method 2: Prune to specific target count (keeps highest opacity Gaussians)
print("\n" + "=" * 60)
print("METHOD 2: Pruning to Target Count")
print("=" * 60)
gs_pruned_count = prune_gaussians_by_count(
    output2["gs"],
    target_count=50000,  # Adjust this to control final file size
    verbose=True,
)
gs_pruned_count.save_ply(f"splat_pruned_count.ply")
pruned_size2 = os.path.getsize("splat_pruned_count.ply") / (1024 * 1024)  # MB
print(f"Pruned PLY file size: {pruned_size2:.2f} MB")
print(f"Size reduction: {((original_size - pruned_size2) / original_size * 100):.1f}%")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Original:              {original_size:.2f} MB  (splat_original.ply)")
print(f"Opacity Pruning:       {pruned_size:.2f} MB  (splat_pruned_opacity.ply)")
print(f"Count-Based Pruning:   {pruned_size2:.2f} MB  (splat_pruned_count.ply)")
print("=" * 60)
print("\nRecommendation:")
print("- For best quality/size balance: Use opacity threshold 0.05-0.1")
print(
    "- For matching Meta's playground (~4.5 MB): Use target_count around 50,000-100,000"
)
print("- Adjust parameters based on your specific needs")
