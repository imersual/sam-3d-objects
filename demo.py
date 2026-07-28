# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
sys.path.append("process/3d-generator")
from inference import Inference, load_image, load_mask, load_single_mask
from request_utils import extract_metric_scale

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
    seed=1,
    with_mesh_postprocess=True,
    with_texture_baking=True,
    with_layout_postprocess=True,
    rendering_engine="nvdiffrast",
)

mesh = output["glb"]
print(f"Raw mesh size (normalized unit cube): {mesh.extents} m")

# SAM3D always emits the mesh in a [-0.5, 0.5] cube, so its longest side is 1.0
# no matter how big the real object is. The real-world size comes back
# separately as output["scale"] (metres per cube unit); apply it to get a mesh
# in metres.
metric_scale = extract_metric_scale(output)
if metric_scale is not None:
    mesh.apply_scale(metric_scale)
    print(f"Metric scale: {metric_scale:.4f} m per cube unit")
    print(f"Real-world size: {mesh.extents} m  (y is up)")
else:
    print("WARNING: no metric scale predicted; mesh stays unit-cube sized")

# export gaussian splat
mesh.export(f"splat.glb")
print("Your reconstruction has been saved to splat.glb")
