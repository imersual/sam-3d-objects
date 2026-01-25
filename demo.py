# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=True)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/sofa/sofa.jpeg")
mask = load_single_mask("notebook/images/sofa", index=1)

# run model
output = inference(
    image,
    mask,
    seed=1,
    with_mesh_postprocess=True,
    with_texture_baking=True,
    with_layout_postprocess=True,
    rendering_engine="nvdiffrast",
)

# export gaussian splat
mesh = output["glb"]
mesh.export(f"splat.glb")
print("Your reconstruction has been saved to splat.glb")
