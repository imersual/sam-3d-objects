#!/usr/bin/env python
"""
SAM3D inference with video output script
Usage: python run_inference.py <task_dir> <image_path> <output_path>
"""
import sys

if len(sys.argv) != 4:
    print("Usage: python run_inference.py <image_path> <mask_path> <output_path>")
    sys.exit(1)

image_path = sys.argv[1]
mask_path = sys.argv[2]
output_path = sys.argv[3]

# import inference code
sys.path.append("notebook")
import imageio
from inference import (
    Inference,
    load_image,
    load_mask,
    ready_gaussian_for_video_rendering,
    render_video,
    make_scene,
)
import random
import os

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
print(f"Loading model from: {config_path}")
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
print(f"Loading image: {image_path}")
image = load_image(image_path)

# print(f"Loading mask from: {task_dir}")
# mask = load_single_mask(task_dir, index=1)

print(f"Loading mask: {mask_path}")
mask = load_mask(mask_path)

# run model
print("Running Video genration inference...")
output = inference(
    image,
    mask,
    seed=random.randint(0, 2**32 - 1),
    with_mesh_postprocess=False,
    with_texture_baking=False,
    with_layout_postprocess=False,
    rendering_engine="nvdiffrast",
)

scene_gs = make_scene(output)

scene_gs = ready_gaussian_for_video_rendering(scene_gs)

print("Generating video...")

video = render_video(
    scene_gs,
    pitch_deg=10,
    resolution=800,
)["color"]

print("Saving video as webm...")

imageio.mimsave(
    os.path.join(f"{output_path}"),
    video,
    fps=30,
    codec="libvpx-vp9",
)

print(f"Your rendering video has been saved to {output_path}")
