# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask, load_single_mask
from prune_gaussians import prune_gaussians_by_opacity

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

# OPTIMIZE: Prune low-opacity Gaussians to reduce file size (typically 80-90% reduction)
# Adjust opacity_threshold (0.05-0.2) to control quality vs file size
# Higher threshold = smaller file but potentially lower quality
prune_gaussians_by_opacity(output["gs"], opacity_threshold=0.2, verbose=True)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
