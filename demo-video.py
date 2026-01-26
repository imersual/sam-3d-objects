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
    with_mesh_postprocess=True,
    with_texture_baking=True,
    with_layout_postprocess=True,
    rendering_engine="nvdiffrast",
)

# export gaussian splat
mesh = output["glb"]
mesh.export(f"splat.glb")

print("Your reconstruction has been saved to splat.glb")

scene_gs = make_scene(output)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)

video = render_video(
    scene_gs,
    r=1,
    fov=60,
    pitch_deg=15,
    yaw_start_deg=-45,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    os.path.join(f"{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)

# notebook display
ImageDisplay(url=f"{IMAGE_NAME}.gif?cache_invalidator={uuid.uuid4()}")

# interactive_visualizer(f"{IMAGE_NAME}.glb")
