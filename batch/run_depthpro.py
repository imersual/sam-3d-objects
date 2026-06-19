#!/usr/bin/env python
"""
run_depthpro.py
===============
Parameterized Depth Pro -> SAM3D pointmap (env: depthpro).

Runs Depth Pro on a single image, then back-projects the metric depth
to a (H, W, 3) world-space pointmap and writes:

    <out-dir>/pointmap.pt        torch.Tensor (H, W, 3) float32   <- SAM3D input
    <out-dir>/pointmap.npy       np.ndarray   (H, W, 3) float32
    <out-dir>/depth.npy          np.ndarray   (H, W)    float32   metric depth (m)
    <out-dir>/camera_params.json reference dump

Must run inside the `depthpro` mamba env (where `depth_pro` is installed).
A single ``model.infer`` call yields both metric depth and focal length, so
no separate ``depth-pro-run`` CLI step is required.

Coordinate convention: X=right, Y=down, Z=forward (camera == world origin).

Usage
-----
    python run_depthpro.py --image path/to/image.jpg --out-dir output/images/foo/depthpro
"""
import argparse
import json
import os

import numpy as np
import torch

import depth_pro


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out-dir", required=True, help="Directory for Depth Pro outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[depthpro] loading model")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    print(f"[depthpro] inference on {args.image}")
    img, _, f_px = depth_pro.load_rgb(args.image)
    with torch.no_grad():
        pred = model.infer(transform(img), f_px=f_px)

    # Depth Pro returns metric depth at the native input resolution.
    depth = pred["depth"].squeeze().cpu().numpy().astype(np.float32)
    H, W = depth.shape

    # focallength_px is in pixels of the returned depth map. If the depth was
    # resized relative to the model's internal output, scale accordingly.
    f_internal = float(pred["focallength_px"].item())
    out_h, out_w = pred["depth"].shape[-2:]
    fx = f_internal * (W / out_w)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0
    print(f"[depthpro] fx={fx:.2f} fy={fy:.2f} cx={cx} cy={cy}  ({W}x{H})")

    # Back-project depth -> pointmap
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    Z = depth
    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z
    pointmap = np.stack([X, Y, Z], axis=-1).astype(np.float32)

    np.save(os.path.join(args.out_dir, "depth.npy"), depth)
    np.save(os.path.join(args.out_dir, "pointmap.npy"), pointmap)
    torch.save(torch.from_numpy(pointmap), os.path.join(args.out_dir, "pointmap.pt"))

    camera_params = {
        "fx": round(fx, 2), "fy": round(fy, 2),
        "cx": cx, "cy": cy,
        "width": W, "height": H,
        "depth_unit": "meters",
    }
    with open(os.path.join(args.out_dir, "camera_params.json"), "w") as f:
        json.dump(camera_params, f, indent=2)

    print(f"[depthpro] pointmap.pt saved: {pointmap.shape}")
    print(f"[depthpro] Z range: [{Z.min():.3f}, {Z.max():.3f}] m")


if __name__ == "__main__":
    main()
