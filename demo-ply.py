# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask, load_single_mask

# Import the OFFICIAL optimization function from the repo
from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import simplify_gs

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

# OPTIMIZE using the official simplify_gs function from the repo
# This reduces Gaussians while maintaining quality through optimization
# simplify=0.95 means keep only 5% of Gaussians (remove 95%)
# NOTE: This takes ~5-10 minutes as it renders from 100 views and optimizes
print("\nOptimizing Gaussian splat (this may take several minutes)...")
output["gs"] = simplify_gs(output["gs"], simplify=0.95, verbose=True)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your optimized reconstruction has been saved to splat.ply")
