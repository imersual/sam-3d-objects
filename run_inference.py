#!/usr/bin/env python
"""
SAM3D inference script
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
from inference import Inference, load_image, load_mask

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
print("Running SAM3D inference...")
output = inference(
    image, mask, seed=42, with_mesh_postprocess=True, with_layout_postprocess=False
)

mesh = output["glb"]
print(f"Exporting 3D model to: {output_path}")
mesh.export(output_path)
print(f"✓ 3D model exported successfully to {output_path}")
