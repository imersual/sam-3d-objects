# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")

import os
import imageio
import uuid
from IPython.display import Image as ImageDisplay
from inference import (
    Inference,
    ready_gaussian_for_video_rendering,
    render_video,
    load_image,
    load_single_mask,
    display_image,
    make_scene,
    interactive_visualizer,
)

IMAGE_NAME = "sofa"

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image(f"notebook/images/{IMAGE_NAME}/{IMAGE_NAME}.jpeg")
mask = load_single_mask(f"notebook/images/{IMAGE_NAME}", index=1)

display_image(image, masks=[mask])

# run model
output = inference(
    image,
    mask,
    seed=1,
    with_mesh_postprocess=False,
    with_texture_baking=False,
    with_layout_postprocess=False,
    rendering_engine="pytorch3d",
)

# export gaussian splat
mesh = output["glb"]
mesh.export(f"{IMAGE_NAME}.glb")

print("Your reconstruction has been saved to splat.glb")

print("Preparing scene for video rendering...")

scene_gs = make_scene(output)

print("Rendering video...")

scene_gs = ready_gaussian_for_video_rendering(scene_gs)

print("Generating video...")

video = render_video(
    scene_gs,
    r=1.5,
    fov=60,
    pitch_deg=10,
    yaw_start_deg=0,
    resolution=750,
)["color"]

print("Saving video as gif...")

# save video as gif
imageio.mimsave(
    os.path.join(f"{IMAGE_NAME}.mp4"),
    video,
    fps=30,  # Use fps instead of duration for video formats
    codec="libx264",  # Common H.264 codec
)

print("Your rendering video has been saved to {IMAGE_NAME}.mp4")
