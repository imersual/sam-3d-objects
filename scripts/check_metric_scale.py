#!/usr/bin/env python
"""Verify SAM3D's real-world size prediction against an object you can measure.

SAM3D emits the mesh normalized to a [-0.5, 0.5] cube, so the raw longest side
is always 1.0. The real-world size comes back separately as `scale` (metres per
cube unit), decoded against MoGe-2's metric pointmap and refined by the layout
post-optimization. The server bakes it in before exporting; this script prints
both the before and after so you can compare against a tape measure.

Uses the same `extract_metric_scale()` helper the server does, so a green run
here means the production path is right - not just this script.

Usage (from the repo root, inside the SAM3D conda env):

    python scripts/check_metric_scale.py IMAGE MASK [MASK ...]
        [--expected-cm 210] [--axis longest] [--seed 1] [--export out.glb]

Quick smoke test on the bundled sofa (a 3-seat sofa is ~2 m wide):

    python scripts/check_metric_scale.py notebook/images/sofa/sofa.jpeg
        notebook/images/sofa/1.png --expected-cm 200
"""
import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(_ROOT, "notebook"))
sys.path.insert(0, os.path.join(_ROOT, "process", "3d-generator"))
sys.path.insert(0, _ROOT)

from request_utils import extract_metric_scale  # noqa: E402

AXES = ("x", "y", "z")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("masks", nargs="+", help="one or more mask files, OR'd together")
    p.add_argument("--tag", default="hf", help="checkpoint tag under checkpoints/")
    p.add_argument("--seed", type=int, default=1, help="fixed seed for repeatability")
    p.add_argument(
        "--expected-cm",
        type=float,
        default=None,
        help="the real measurement in cm, to report the error against",
    )
    p.add_argument(
        "--axis",
        default="longest",
        choices=("longest", "x", "y", "z"),
        help="which axis --expected-cm refers to (default: longest). GLB is "
             "y-up, so height is y and the footprint is x/z.",
    )
    p.add_argument("--export", default=None, help="also write the scaled mesh here")
    return p.parse_args()


def main():
    args = parse_args()

    # Imported here, not at module scope, so --help works outside the conda env
    # (notebook/inference.py reads CONDA_PREFIX at import time).
    import numpy as np
    from inference import Inference, load_image, load_mask

    config_path = os.path.join(_ROOT, "checkpoints", args.tag, "pipeline.yaml")
    print(f"Loading model from: {config_path}")
    inference = Inference(config_path, compile=False)

    image = load_image(args.image)
    masks = [load_mask(m) for m in args.masks]
    mask = masks[0].copy()
    for m in masks[1:]:
        mask |= m

    # Identical settings to process/3d-generator/server.py's single-view path.
    output = inference(
        image,
        mask,
        seed=args.seed,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        rendering_engine="nvdiffrast",
    )

    mesh = output["glb"]
    raw_extents = np.asarray(mesh.extents, dtype=float).copy()

    print("\n" + "=" * 68)
    print(f"image  : {args.image}")
    print(f"masks  : {', '.join(args.masks)}")
    print(f"seed   : {args.seed}")
    print("-" * 68)
    print("raw mesh (normalized unit cube)")
    print(f"  extents      : {np.round(raw_extents, 4).tolist()}")
    print(f"  longest side : {raw_extents.max():.4f}  (always ~1.0 — this is the bug)")

    raw_scale = output.get("scale")
    if raw_scale is not None:
        print(f"  raw 'scale'  : {np.round(np.asarray(raw_scale.detach().cpu()), 6).tolist()}")
    translation = output.get("translation")
    if translation is not None:
        t = np.asarray(translation.detach().cpu(), dtype=float).reshape(-1)
        print(f"  camera dist. : {np.linalg.norm(t):.3f} m  (sanity check on the depth)")

    metric_scale = extract_metric_scale(output)
    print("-" * 68)
    if metric_scale is None:
        print("NO METRIC SCALE — the layout head returned nothing usable.")
        print("The export would stay unit-cube sized. Check the logs above for a")
        print("layout post-optimization error, and confirm the mask is not empty.")
        return 1

    mesh.apply_scale(metric_scale)
    extents = np.asarray(mesh.extents, dtype=float)
    print(f"metric scale   : {metric_scale:.6f} m per cube unit")
    print("scaled mesh (metres)")
    for name, value in zip(AXES, extents):
        label = " (up)" if name == "y" else ""
        print(f"  {name}{label:5s}       : {value:.4f} m   = {value * 100:7.1f} cm")
    print(f"  longest side : {extents.max():.4f} m   = {extents.max() * 100:7.1f} cm")

    if args.expected_cm is not None:
        if args.axis == "longest":
            measured_cm = extents.max() * 100
            which = f"longest side ({AXES[int(extents.argmax())]})"
        else:
            measured_cm = extents[AXES.index(args.axis)] * 100
            which = f"{args.axis} axis"
        error = measured_cm - args.expected_cm
        pct = 100 * error / args.expected_cm
        print("-" * 68)
        print(f"expected       : {args.expected_cm:7.1f} cm")
        print(f"measured       : {measured_cm:7.1f} cm  ({which})")
        print(f"error          : {error:+7.1f} cm  ({pct:+.1f}%)")

    if args.export:
        mesh.export(args.export)
        print(f"\nScaled mesh written to: {args.export}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
