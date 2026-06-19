#!/usr/bin/env python
"""
run_da3.py
==========
Parameterized Depth Anything 3 inference (env: da3).

Runs DA3 on a single image and writes the raw outputs that
``depth_anything_to_pointmap.py`` expects:

    <out-dir>/depth.npy        (1, H, W) float32   metric depth (m)
    <out-dir>/intrinsics.npy   (1, 3, 3) float32   K
    <out-dir>/extrinsics.npy   (1, 3, 4) float32   [R | t]
    <out-dir>/confidence.npy   (1, H, W) float32   confidence
    <out-dir>/camera_params.json                   reference dump

Must run inside the `da3` mamba env (where `depth_anything_3` is installed).

Usage
-----
    python run_da3.py --image path/to/image.jpg --out-dir output/images/foo/da3
"""
import argparse
import json
import os

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out-dir", required=True, help="Directory for DA3 outputs")
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                    help="HF model id for DepthAnything3.from_pretrained")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[da3] loading {args.model} on {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device)

    print(f"[da3] inference on {args.image}")
    prediction = model.inference([args.image])

    np.save(os.path.join(args.out_dir, "depth.npy"), prediction.depth)
    np.save(os.path.join(args.out_dir, "intrinsics.npy"), prediction.intrinsics)
    np.save(os.path.join(args.out_dir, "extrinsics.npy"), prediction.extrinsics)
    np.save(os.path.join(args.out_dir, "confidence.npy"), prediction.conf)

    camera_params = {
        "intrinsics": prediction.intrinsics.tolist(),
        "extrinsics": prediction.extrinsics.tolist(),
        "depth_shape": list(prediction.depth.shape),
        "note": "Depth Anything 3 output",
    }
    with open(os.path.join(args.out_dir, "camera_params.json"), "w") as f:
        json.dump(camera_params, f, indent=4)

    print(f"[da3] saved outputs to {args.out_dir}")
    print(f"[da3] depth shape: {prediction.depth.shape}")


if __name__ == "__main__":
    main()
