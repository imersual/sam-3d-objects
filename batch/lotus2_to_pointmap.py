#!/usr/bin/env python
"""
lotus2_to_pointmap.py
=====================
Convert Lotus-2 monocular depth to a SAM3D-compatible pointmap (env: sam3d-objects).

Lotus-2 outputs normalised affine-invariant inverse depth (disparity) in [0, 1]:
    1.0 = closest, 0.0 = farthest.  No metric scale, no intrinsics, no extrinsics.

We map disparity -> metric depth linearly:
    depth_m = far - (far - near) * depth_norm
and supply intrinsics via --fov-h (or explicit --fx/--fy/--cx/--cy, or --cam json).

Outputs
    pointmap.pt   torch.Tensor (H, W, 3) float32  <- SAM3D input
    pointmap.npy  np.ndarray   (H, W, 3) float32
    depth_metric.npy  (H, W) float32  metric depth (m)

Usage
    python lotus2_to_pointmap.py --depth depth_npy/image.npy \
        --out-dir output/images/foo/lotus2 --fov-h 24 --near 0.2 --far 1.5
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def lotus2_depth_to_metric(depth_norm, near, far):
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    return (far - (far - near) * depth_norm).astype(np.float32)


def backproject_to_pointmap(depth_m, fx, fy, cx, cy):
    H, W = depth_m.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    Z = depth_m
    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def intrinsics_from_fov(fov_h_deg, W, H):
    fx = (W / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))
    return fx, fx, W / 2.0, H / 2.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", required=True, help="Lotus-2 depth .npy (H, W) in [0,1]")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cam", default=None, help="JSON with fx/fy/cx/cy")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--fov-h", type=float, default=None,
                   help="Horizontal FOV deg. 24=85mm FF, 55=35mm FF, 70=phone, 90=wide")
    p.add_argument("--near", type=float, default=0.3)
    p.add_argument("--far", type=float, default=3.0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    depth_norm = np.load(args.depth).astype(np.float32)
    if depth_norm.ndim == 3:
        depth_norm = depth_norm.squeeze(0)
    if depth_norm.ndim != 2:
        raise ValueError(f"Expected 2D depth, got {depth_norm.shape}")
    H, W = depth_norm.shape

    if args.cam:
        with open(args.cam) as f:
            cam = json.load(f)
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    elif all(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    elif args.fov_h is not None:
        fx, fy, cx, cy = intrinsics_from_fov(args.fov_h, W, H)
    else:
        fx, fy, cx, cy = intrinsics_from_fov(70.0, W, H)
        print("[lotus2->pt] no camera info -> defaulting to 70 deg FOV")

    print(f"[lotus2->pt] fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
          f"near={args.near} far={args.far} ({W}x{H})")

    depth_m = lotus2_depth_to_metric(depth_norm, args.near, args.far)
    pointmap = backproject_to_pointmap(depth_m, fx, fy, cx, cy)

    np.save(out / "depth_metric.npy", depth_m)
    np.save(out / "pointmap.npy", pointmap)

    import torch
    torch.save(torch.from_numpy(pointmap), out / "pointmap.pt")
    print(f"[lotus2->pt] saved pointmap.pt {pointmap.shape} -> {out}")
    print(f"[lotus2->pt] metric depth [{depth_m.min():.3f}, {depth_m.max():.3f}] m")


if __name__ == "__main__":
    main()
