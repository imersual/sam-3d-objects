#!/usr/bin/env python
"""
batch_sam3d.py
==============
Run SAM3D for one image once per available depth backend (env: sam3d-objects).

Loads the SAM3D model ONCE, then produces one GLB per pointmap that exists
under <out-dir>, plus the MoGe default (no pointmap):

    <out-dir>/splat_MoGe.glb               (always, pointmap=None; MoGe v1 default)
    <out-dir>/depthpro/pointmap.pt  -> <out-dir>/splat_with_pt_depthpro.glb
    <out-dir>/da3/pointmap.pt       -> <out-dir>/splat_with_pt_da3.glb
    <out-dir>/lotus2/pointmap.pt    -> <out-dir>/splat_with_pt_lotus2.glb
    <out-dir>/moge2/pointmap.pt     -> <out-dir>/splat_with_pt_moge2.glb

Usage
    python batch_sam3d.py --image input/images/foo/image.jpg \
        --mask input/images/foo/mask.png --out-dir output/images/foo --seed 1
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "notebook"))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from inference import Inference, load_image, load_mask


def mask_pointmap(pointmap, mask):
    """Set everything outside the object mask to NaN.

    The depth backends emit a FULL-SCENE pointmap (object + background). SAM3D
    infers the camera intrinsics / scene scale from every FINITE pixel of the
    pointmap (it filters only by torch.isfinite, with no object mask), so leaving
    the background finite makes the focal/shift solve fit the whole room instead
    of the object -> wrong scale/placement. Restricting the pointmap to the
    object (NaN elsewhere) is what ties the pointmap to mask.png.

    pointmap: torch.Tensor (H, W, 3); mask: bool ndarray (Hm, Wm).
    """
    H, W = pointmap.shape[:2]
    m = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m, size=(H, W), mode="nearest")
    keep = m[0, 0] > 0.5
    pointmap = pointmap.clone()
    pointmap[~keep] = float("nan")
    return pointmap

# backend key -> (pointmap path relative to out-dir, output glb name)
BACKENDS = [
    ("depthpro", "depthpro/pointmap.pt", "splat_with_pt_depthpro.glb"),
    ("da3", "da3/pointmap.pt", "splat_with_pt_da3.glb"),
    ("lotus2", "lotus2/pointmap.pt", "splat_with_pt_lotus2.glb"),
    ("moge2", "moge2/pointmap.pt", "splat_with_pt_moge2.glb"),
]


def check_mask_coverage(mask, label):
    """Warn when a mask covers ~0% or ~100% of the frame (misread mask)."""
    coverage = float(mask.mean())
    print(f"[sam3d] mask {label}: {coverage:.1%} of pixels are object")
    if coverage < 0.001 or coverage > 0.999:
        print(
            f"[sam3d] WARNING: mask coverage {coverage:.1%} looks wrong "
            f"(empty or whole-image). Check the mask format/polarity."
        )


def run_multiview(inference, views_dir, out_dir, seed, skip_existing):
    """Fuse all view pairs in views_dir into one splat_multiview.glb."""
    from load_images_and_masks import load_images_and_masks_from_path

    out_glb = os.path.join(out_dir, "splat_multiview.glb")
    if skip_existing and os.path.exists(out_glb):
        print(f"[sam3d] splat_multiview.glb exists -> skip")
        return

    images, masks, names = load_images_and_masks_from_path(views_dir)
    if len(images) < 2:
        print(
            f"[sam3d] ERROR: multiview needs >=2 valid image/mask pairs, "
            f"found {len(images)} ({names}) in {views_dir}"
        )
        sys.exit(1)

    for name, mask in zip(names, masks):
        check_mask_coverage(mask, f"view '{name}'")

    print(f"[sam3d] running multiview fusion over {len(images)} views: {names}")
    output = inference.multi_view(images, masks, seed=seed)
    output["glb"].export(out_glb)
    print(f"[sam3d] exported -> {out_glb}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", default=None, help="single-view input image")
    ap.add_argument("--mask", default=None, help="single-view input mask")
    ap.add_argument(
        "--views-dir",
        default=None,
        help="multiview: folder of <stem>.png + <stem>_mask.png pairs; "
        "overrides --image/--mask",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="hf", help="checkpoint subfolder")
    ap.add_argument("--seed", type=int, default=4096)
    ap.add_argument(
        "--no-moge",
        action="store_true",
        help="Skip the pointmap-free MoGe baseline run",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip backends whose output GLB already exists",
    )
    args = ap.parse_args()

    if args.views_dir is None and (args.image is None or args.mask is None):
        ap.error("either --views-dir or both --image and --mask are required")

    os.makedirs(args.out_dir, exist_ok=True)

    config_path = os.path.join(_ROOT, "checkpoints", args.tag, "pipeline.yaml")
    print(f"[sam3d] loading model from {config_path}")
    inference = Inference(config_path, compile=False)

    if args.views_dir:
        run_multiview(
            inference, args.views_dir, args.out_dir, args.seed, args.skip_existing
        )
        return

    image = load_image(args.image)
    mask = load_mask(args.mask)

    # A mask covering ~0% or ~100% of the frame means mask.png was misread
    # (wrong channel / inverted / empty) — surface it instead of silently
    # producing garbage.
    check_mask_coverage(mask, args.mask)

    # Build the run list: MoGe default + every backend whose pointmap exists.
    runs = []
    if not args.no_moge:
        runs.append(("MoGe", None, "splat_MoGe.glb"))
    for key, rel, glb in BACKENDS:
        pt_path = os.path.join(args.out_dir, rel)
        if os.path.exists(pt_path):
            runs.append((key, pt_path, glb))
        else:
            print(f"[sam3d] no pointmap for '{key}' ({pt_path}) -> skip")

    for key, pt_path, glb_name in runs:
        out_glb = os.path.join(args.out_dir, glb_name)
        if args.skip_existing and os.path.exists(out_glb):
            print(f"[sam3d] {glb_name} exists -> skip")
            continue

        pointmap = torch.load(pt_path) if pt_path else None
        if pointmap is not None:
            pointmap = mask_pointmap(pointmap, mask)
        print(
            f"[sam3d] running '{key}' (pointmap={'yes' if pointmap is not None else 'no'})"
        )
        try:
            output = inference(
                image,
                mask,
                seed=args.seed,
                pointmap=pointmap,
                with_mesh_postprocess=True,
                with_texture_baking=True,
                with_layout_postprocess=True,
                rendering_engine="nvdiffrast",
            )
            output["glb"].export(out_glb)
            print(f"[sam3d] exported -> {out_glb}")
        except Exception as exc:
            print(f"[sam3d] FAILED '{key}': {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
