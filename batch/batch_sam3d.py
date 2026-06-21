#!/usr/bin/env python
"""
batch_sam3d.py
==============
Run SAM3D for one image once per available depth backend (env: sam3d-objects).

Loads the SAM3D model ONCE, then produces one GLB per pointmap that exists
under <out-dir>, plus the MoGe default (no pointmap):

    <out-dir>/splat_MoGe.glb               (always, pointmap=None)
    <out-dir>/depthpro/pointmap.pt  -> <out-dir>/splat_with_pt_depthpro.glb
    <out-dir>/da3/pointmap.pt       -> <out-dir>/splat_with_pt_da3.glb
    <out-dir>/lotus2/pointmap.pt    -> <out-dir>/splat_with_pt_lotus2.glb

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

import torch
from inference import Inference, load_image, load_mask

# backend key -> (pointmap path relative to out-dir, output glb name)
BACKENDS = [
    ("depthpro", "depthpro/pointmap.pt", "splat_with_pt_depthpro.glb"),
    ("da3", "da3/pointmap.pt", "splat_with_pt_da3.glb"),
    ("lotus2", "lotus2/pointmap.pt", "splat_with_pt_lotus2.glb"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
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

    os.makedirs(args.out_dir, exist_ok=True)

    config_path = os.path.join(_ROOT, "checkpoints", args.tag, "pipeline.yaml")
    print(f"[sam3d] loading model from {config_path}")
    inference = Inference(config_path, compile=False)

    image = load_image(args.image)
    mask = load_mask(args.mask)

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
