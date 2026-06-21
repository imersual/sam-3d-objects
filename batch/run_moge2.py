#!/usr/bin/env python
"""
run_moge2.py
============
Parameterized MoGe-2 -> SAM3D pointmap (env: sam3d-objects).

MoGe-2 (microsoft/MoGe v2) predicts a metric, camera-space point map directly,
so no back-projection step is needed. The `MoGe` package is already a
dependency of the sam3d-objects env (SAM3D's built-in default uses MoGe v1),
so this runs in the same env -- no separate conda env required.

Writes the outputs SAM3D consumes:

    <out-dir>/pointmap.pt        torch.Tensor (H, W, 3) float32   <- SAM3D input
    <out-dir>/pointmap.npy       np.ndarray   (H, W, 3) float32
    <out-dir>/depth.npy          np.ndarray   (H, W)    float32   metric depth (m)
    <out-dir>/mask.npy           np.ndarray   (H, W)    bool       valid pixels
    <out-dir>/camera_params.json reference dump (intrinsics, shape)

Coordinate convention: X=right, Y=down, Z=forward (camera == world origin),
matching the other backends.

Usage
-----
    python run_moge2.py --image path/to/image.jpg --out-dir output/images/foo/moge2
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from moge.model.v2 import MoGeModel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out-dir", required=True, help="Directory for MoGe-2 outputs")
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl-normal",
                    help="HF model id for MoGeModel.from_pretrained")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[moge2] loading {args.model} on {device}")
    model = MoGeModel.from_pretrained(args.model).to(device).eval()

    print(f"[moge2] inference on {args.image}")
    rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(rgb).permute(2, 0, 1).to(device)  # (3, H, W) in [0,1]

    with torch.no_grad():
        output = model.infer(image)

    # MoGe returns a metric, camera-space point map -> use directly as pointmap.
    pointmap = output["points"].cpu().numpy().astype(np.float32)  # (H, W, 3)
    depth = output["depth"].cpu().numpy().astype(np.float32)      # (H, W)
    mask = output["mask"].cpu().numpy().astype(bool)              # (H, W)

    # Zero out invalid pixels (consistent with the other backends).
    pointmap[~mask] = 0.0

    intrinsics = output.get("intrinsics")
    intrinsics = (intrinsics.cpu().numpy().tolist()
                  if intrinsics is not None else None)

    H, W = depth.shape
    np.save(os.path.join(args.out_dir, "depth.npy"), depth)
    np.save(os.path.join(args.out_dir, "mask.npy"), mask)
    np.save(os.path.join(args.out_dir, "pointmap.npy"), pointmap)
    torch.save(torch.from_numpy(pointmap), os.path.join(args.out_dir, "pointmap.pt"))

    camera_params = {
        "intrinsics_normalized": intrinsics,
        "width": W, "height": H,
        "depth_unit": "meters",
        "note": "MoGe-2 metric point map (camera space)",
    }
    with open(os.path.join(args.out_dir, "camera_params.json"), "w") as f:
        json.dump(camera_params, f, indent=2)

    valid = depth[mask]
    print(f"[moge2] pointmap.pt saved: {pointmap.shape}")
    print(f"[moge2] valid {int(mask.sum())}/{mask.size} pixels")
    if valid.size:
        print(f"[moge2] Z range: [{valid.min():.3f}, {valid.max():.3f}] m")


if __name__ == "__main__":
    main()
