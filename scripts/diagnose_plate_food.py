#!/usr/bin/env python
"""
diagnose_plate_food.py
======================
Why does a plate + food selection merged into ONE object come back as a bare
plate? And what do separately reconstructed objects look like before the
poller composes them? Run this on the GPU box to find out.

The evidence so far (2026-09-23): the same photo and the same two masks,
merged into one object, gave cake-on-plate on 2026-03-31 and a bare plate on
2026-08-05, 09-17, 09-19 and 09-23; a tray lost its spoon the same way on
08-02. What changed in between for stage 1 -- the step that decides which
voxels exist -- is the depth model: MoGe-1 was swapped for MoGe-2 on 07-22,
and its pointmap conditions stage 1. The seed is the other variable: until
2026-09-23 the server's default seed was stuck at 625644691 after the first
request (see request_utils.resolve_seed), so every failure may share it.

This separates the two. For each depth model and each seed it runs stage 1
on the merged mask and reports how tall the occupied voxels are relative to
their width: a bare plate is ~0.1, the cake on its plate was ~0.24.

    stage 1 (default)  depth models x seeds, merged mask, voxel height ratio
    --full             full reconstruction per depth model, merged mask,
                       first seed: GLB + pose, to look at
    --per-object       each mask on its own, production depth model, first
                       seed: GLB + /infer-style pose JSON, so the poller's
                       composition can be replayed offline

When exactly two masks are given, the first is taken to be the food and the
second its support (the plate), and each depth model's pointmap also reports
how far the food stands out toward the camera (depth_relief).

Usage, in the SAM3D conda env, on a GPU the server is not using:

    cd /workspace/sam-3d-objects
    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_plate_food.py \\
        --image https://.../outputs/<mask task>/upscaled.jpg \\
        --masks https://.../outputs/<mask task>/object_1.jpg \\
                https://.../outputs/<mask task>/object_2.jpg \\
        --out /workspace/diagnose/cheesecake --full --per-object

Everything lands in --out; summary.txt is the table to read first.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# The stuck seed first: every warm-server request before 2026-09-23 used it.
DEFAULT_SEEDS = (625644691, 0, 1, 2, 3)

# Voxel height over width. Measured on real outputs: bare plates 0.08-0.11,
# the cheesecake on its plate 0.24.
FOOD_HEIGHT_RATIO = 0.17


def voxel_height_ratio(coords):
    """Height of the occupied stage-1 voxels over their wider horizontal side.

    coords: (N, 4) [batch, x, y, z] as sample_sparse_structure returns them.
    The voxel grid is SAM3D's own z-up frame (to_glb turns it y-up later), so
    z is height.
    """
    xyz = np.asarray(coords)[:, 1:4]
    extent = xyz.max(axis=0) - xyz.min(axis=0) + 1
    return float(extent[2] / max(extent[0], extent[1]))


def depth_relief(pointmap, food_mask, support_mask):
    """How far the food stands out of its support toward the camera, as a
    fraction of the support's width: (median depth of the support - median
    depth of the food) / the support's horizontal span.

    pointmap: (H, W, 3) in the pipeline's camera convention (Z is distance
    from the camera). Invalid pixels (inf or nan) are ignored. Scale-free, so
    MoGe-1's relative depth and MoGe-2's metric depth compare directly: a
    depth model that sees the cake as part of the plate's surface gives ~0.
    """
    pointmap = np.asarray(pointmap, dtype=np.float64)
    valid = np.isfinite(pointmap).all(axis=-1)
    food = np.asarray(food_mask, bool) & valid
    support = np.asarray(support_mask, bool) & valid
    span = pointmap[support, 0].max() - pointmap[support, 0].min()
    return float((np.median(pointmap[support, 2]) - np.median(pointmap[food, 2])) / span)


def _resized(mask, shape):
    """A boolean mask at (H, W) `shape`, nearest-neighbour, unchanged if it
    already is."""
    if mask.shape == tuple(shape):
        return mask
    from PIL import Image

    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((shape[1], shape[0]), Image.NEAREST)) > 0


def _fetch(source, out_dir):
    """A local path as-is; a URL downloaded into out_dir (retried: the
    DigitalOcean CDN drops connections now and then)."""
    if not source.startswith(("http://", "https://")):
        return source
    dest = Path(out_dir) / "inputs" / source.rstrip("/").split("/")[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(source, dest)
            return str(dest)
        except Exception as exc:  # noqa: BLE001 -- retry anything, report the last
            error = exc
            time.sleep(2)
    raise RuntimeError(f"could not download {source}: {error}")


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", required=True, help="photo: path or URL")
    parser.add_argument("--masks", required=True, nargs="+", help="masks: paths or URLs")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--tag", default="hf", help="checkpoint tag under checkpoints/")
    parser.add_argument("--full", action="store_true", help="also run full merged reconstructions")
    parser.add_argument("--per-object", action="store_true", help="also reconstruct each mask alone")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[1]
    for extra in ("notebook", ".", "process/3d-generator", "scripts"):
        sys.path.insert(0, str(root / extra))
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from inference import Inference, load_image, load_mask
    from request_utils import extract_intrinsics, extract_metric_scale, extract_pose
    from sam3d_objects.pipeline.depth_models.moge2 import MoGe2
    from set_depth_model import DEPTH_MODELS

    class MoGe2WithoutProjection(MoGe2):
        """MoGe-2, called the way the pipeline calls MoGe-1: with the raw
        predicted points rather than points re-derived from depth and the
        recovered focal length."""

        def __call__(self, image):
            output = self.model.infer(image.to(self.device), force_projection=False)
            output["pointmaps"] = output["points"]
            return output

    image = load_image(_fetch(args.image, out_dir))
    masks = [load_mask(_fetch(m, out_dir)) for m in args.masks]
    merged = np.logical_or.reduce(masks)

    config = root / "checkpoints" / args.tag / "pipeline.yaml"
    print(f"Loading SAM3D from {config} ...", flush=True)
    inference = Inference(str(config), compile=False)
    pipeline = inference._pipeline
    production = pipeline.depth_model
    moge2 = production if isinstance(production, MoGe2) else instantiate(
        OmegaConf.create(DEPTH_MODELS["moge2"])
    )
    depth_models = {
        f"production ({type(production).__name__})": production,
        "moge1": instantiate(OmegaConf.create(DEPTH_MODELS["moge1"])),
        "moge2": moge2,
        "moge2-no-projection": MoGe2WithoutProjection(moge2.model),
    }

    rgba = inference.merge_mask_to_rgba(image, merged)
    rows, relief = [], {}
    for name, depth_model in depth_models.items():
        pipeline.depth_model = depth_model
        if len(masks) == 2:
            with pipeline.device:
                pointmap = pipeline.compute_pointmap(rgba)["pointmap"]
            pointmap = pointmap.permute(1, 2, 0).float().cpu().numpy()
            food, support = (_resized(m, pointmap.shape[:2]) for m in masks)
            relief[name] = depth_relief(pointmap, food, support)
        for seed in args.seeds:
            started = time.time()
            stage1 = pipeline.run(rgba, None, seed, stage1_only=True)
            coords = stage1["coords"].cpu().numpy()
            ratio = voxel_height_ratio(coords)
            rows.append({
                "depth_model": name, "seed": seed, "voxels": int(len(coords)),
                "height_ratio": round(ratio, 4), "food": ratio >= FOOD_HEIGHT_RATIO,
                "seconds": round(time.time() - started, 1),
            })
            print(json.dumps(rows[-1]), flush=True)
            torch.cuda.empty_cache()

    full = {}
    if args.full:
        for name, depth_model in depth_models.items():
            pipeline.depth_model = depth_model
            output = inference(
                image, merged, seed=args.seeds[0], with_mesh_postprocess=True,
                with_texture_baking=True, with_layout_postprocess=True,
            )
            path = out_dir / f"merged_{name.split(' ')[0]}.glb"
            output["glb"].export(str(path))
            extents = output["glb"].extents  # glTF, y-up
            full[name] = {
                "glb": path.name, "height_ratio": round(float(extents[1] / max(extents[0], extents[2])), 4),
                "pose": extract_pose(output), "metric_scale": extract_metric_scale(output),
            }
            print(json.dumps({name: full[name]}), flush=True)
            torch.cuda.empty_cache()

    per_object = []
    if args.per_object:
        pipeline.depth_model = production
        for index, mask in enumerate(masks):
            output = inference(
                image, mask, seed=args.seeds[0], with_mesh_postprocess=True,
                with_texture_baking=True, with_layout_postprocess=True,
            )
            metric_scale = extract_metric_scale(output)
            mesh = output["glb"]
            if metric_scale is not None:
                mesh.apply_scale(metric_scale)  # exactly as server.py exports it
            path = out_dir / f"object_{index}.glb"
            mesh.export(str(path))
            response = {
                "mask": args.masks[index], "seed": args.seeds[0], "metric_scale": metric_scale,
                "pose": extract_pose(output), "intrinsics": extract_intrinsics(output),
            }
            (out_dir / f"object_{index}.json").write_text(json.dumps(response, indent=1))
            per_object.append({"glb": path.name, **response})
            print(json.dumps(per_object[-1]), flush=True)
            torch.cuda.empty_cache()

    summary = {
        "image": args.image, "masks": args.masks, "food_height_ratio": FOOD_HEIGHT_RATIO,
        "stage1": rows, "depth_relief": relief, "full": full, "per_object": per_object,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"{'depth model':32s} {'seed':>10s} {'voxels':>7s} {'height':>7s}  food?"]
    for row in rows:
        lines.append(
            f"{row['depth_model']:32s} {row['seed']:>10d} {row['voxels']:>7d} "
            f"{row['height_ratio']:>7.3f}  {'YES' if row['food'] else 'no'}"
        )
    for name, value in relief.items():
        lines.append(f"depth relief of the food over its support, {name}: {value:.4f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
