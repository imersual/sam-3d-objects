#!/usr/bin/env python
"""
depth_anything_to_pointmap.py
=============================
Convert Depth Anything 3 outputs to a SAM3D-compatible pointmap (env: sam3d-objects).

Inputs (produced by run_da3.py)
    depth.npy       (1, H, W) float32  metric depth (m)
    confidence.npy  (1, H, W) float32  confidence (optional)
    intrinsics.npy  (1, 3, 3) float32  K
    extrinsics.npy  (1, 3, 4) float32  [R | t]  world-to-camera

Outputs
    pointmap.pt   torch.Tensor (H, W, 3) float32  <- SAM3D input
    pointmap.npy  np.ndarray   (H, W, 3) float32
    pointmap_confidence_mask.npy  (H, W) bool

Coordinate convention: pointmap[v, u] = [X, Y, Z] world space (m),
X=right, Y=down, Z=forward.

Usage
    python depth_anything_to_pointmap.py \
        --depth d.npy --confidence c.npy --intrinsics i.npy --extrinsics e.npy \
        --out-dir output/images/foo/da3 [--conf-percentile 10]
"""
import argparse
import os
from pathlib import Path

import numpy as np


def depth_anything_to_pointmap(depth, intrinsics, extrinsics,
                               confidence=None, conf_percentile=0.0):
    depth = np.squeeze(depth)
    K = np.squeeze(intrinsics).astype(np.float64)
    E = np.squeeze(extrinsics).astype(np.float64)

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    Z = depth.astype(np.float64)

    X_cam = (uu - cx) / fx * Z
    Y_cam = (vv - cy) / fy * Z
    pts_cam = np.stack([X_cam, Y_cam, Z], axis=-1)

    # world = R^T @ (cam - t)   (E is world-to-camera: cam = R @ world + t)
    R = E[:3, :3]
    t = E[:3, 3]
    pts_flat = pts_cam.reshape(-1, 3)
    pts_world = (R.T @ (pts_flat - t).T).T
    pointmap = pts_world.reshape(H, W, 3).astype(np.float32)

    valid = (
        np.isfinite(pointmap).all(axis=-1)
        & (depth > 0)
        & (np.linalg.norm(pointmap, axis=-1) > 1e-6)
    )

    conf_norm = np.zeros((H, W), dtype=np.float32)
    if confidence is not None:
        conf = np.squeeze(confidence).astype(np.float32)
        lo, hi = conf.min(), conf.max()
        conf_norm = (conf - lo) / (hi - lo + 1e-8)
        if conf_percentile > 0.0:
            threshold = np.percentile(conf_norm[valid], conf_percentile)
            valid &= conf_norm >= threshold

    pointmap[~valid] = 0.0
    return pointmap, valid, conf_norm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--confidence", default=None)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--extrinsics", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conf-percentile", type=float, default=0.0,
                    help="Discard pixels below this confidence percentile (0 = keep all)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    depth = np.load(args.depth)
    confidence = (np.load(args.confidence)
                  if args.confidence and os.path.exists(args.confidence) else None)
    intrinsics = np.load(args.intrinsics)
    extrinsics = np.load(args.extrinsics)

    pointmap, valid, _ = depth_anything_to_pointmap(
        depth, intrinsics, extrinsics, confidence, args.conf_percentile)

    n_valid = int(valid.sum())
    print(f"[da3->pt] valid {n_valid}/{valid.size} pixels")

    import torch
    torch.save(torch.from_numpy(pointmap), out / "pointmap.pt")
    np.save(out / "pointmap.npy", pointmap)
    np.save(out / "pointmap_confidence_mask.npy", valid)
    print(f"[da3->pt] saved pointmap.pt {pointmap.shape} -> {out}")


if __name__ == "__main__":
    main()
