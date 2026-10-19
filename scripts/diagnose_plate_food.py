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
how far the food stands out toward the camera (depth_relief). A --masks
entry may join several masks with commas, to use them as one.

    --context-ab       instead of all the above: does SAM3D build a dish under
                       food because it can SEE the bowl? Food in a deep bowl,
                       reconstructed on its own, comes back sitting on a
                       complete dish of its own (the bowl photo, 2026-10-19:
                       19-41% of the food mesh coloured like the bowl, 1.8%
                       of the food in the photo). SAM3D's input crop keeps
                       the pixels around the mask, so it sees the bowl whose
                       front wall hides the food's bottom. This reconstructs
                       the food three ways, --ab-seeds seeds each:
                         as_is           the photo as the server sends it
                         support_hidden  the support's pixels painted over
                         cutout          everything but the food painted over
                       Every variant gets the depth model's output for the
                       UNTOUCHED photo, so only what the model sees changes,
                       not where the food sits. Each writes a GLB, a preview
                       PNG (front, side, top, underneath) and how much of the
                       mesh is coloured like the support, against the same
                       share of the food in the photo. A variant that works
                       shows an excess near 0 and no dish in its preview.

Usage, in the SAM3D conda env, on a GPU the server is not using:

    cd /workspace/sam-3d-objects
    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_plate_food.py \\
        --image https://.../outputs/<mask task>/upscaled.jpg \\
        --masks https://.../outputs/<mask task>/object_1.jpg \\
                https://.../outputs/<mask task>/object_2.jpg \\
        --out /workspace/diagnose/cheesecake --full --per-object

    # The bowl photo: the food (three masks used as one), then the bowl
    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_plate_food.py \\
        --image https://.../outputs/<mask task>/upscaled.jpg \\
        --masks https://.../object_2.png,https://.../object_3.png,https://.../object_8.png \\
                https://.../outputs/<mask task>/object_6.png \\
        --out /workspace/diagnose/bowl --context-ab

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

# --context-ab: what SAM3D is shown in place of what it should not see.
FILL_COLOURS = {"grey": (128, 128, 128), "black": (0, 0, 0)}

# "Coloured like the support": Lab chroma within this percentile of the
# support's own spread around its median, and not darker than this
# percentile of its lightness. Chroma rather than full colour, because a
# bowl's shading spans a lightness range that bread crumb falls into.
# Calibrated on production meshes: bowl-photo food 1.8% in the photo and
# 19-41% in its meshes, the bowl's own mesh 99%, cheesecake 0.0-0.2%, and a
# blue shrimp plate 80% against 12% for its food.
SUPPORT_CHROMA_PERCENTILE = 90
SUPPORT_LIGHTNESS_PERCENTILE = 10


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


def mask_groups(entries):
    """--masks entries as lists of sources: an entry may join several masks
    with commas, which are then used as one mask."""
    return [[part for part in entry.split(",") if part] for entry in entries]


def background_colour(image, masks):
    """Median colour of the photo outside every mask: the table or backdrop,
    the least surprising thing to paint over what SAM3D should not see."""
    rgb = np.asarray(image)[..., :3]
    outside = ~np.logical_or.reduce([np.asarray(mask, bool) for mask in masks])
    if not outside.any():
        return np.array(FILL_COLOURS["grey"], np.uint8)
    return np.median(rgb[outside], axis=0).astype(np.uint8)


def hide(image, mask, colour):
    """A copy of the photo with `mask`'s pixels painted `colour`."""
    painted = np.array(image, copy=True)
    painted[np.asarray(mask, bool), :3] = colour
    return painted


def context_variants(image, food_mask, support_mask, colour):
    """The three photos --context-ab reconstructs the food from, by name."""
    return {
        "as_is": np.array(image, copy=True),
        "support_hidden": hide(image, support_mask, colour),
        "cutout": hide(image, ~np.asarray(food_mask, bool), colour),
    }


def _srgb_to_lab(rgb):
    """(N, 3) 8-bit sRGB to CIE Lab (D65): L in 0-100, a and b signed."""
    c = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    linear = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = linear @ np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ]).T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def support_colour_model(support_rgb):
    """What "coloured like the support" means for one photo, from the
    support's own pixels: (median chroma, chroma radius, lightness floor)."""
    lab = _srgb_to_lab(support_rgb)
    centre = np.median(lab[:, 1:], axis=0)
    radius = float(np.percentile(np.linalg.norm(lab[:, 1:] - centre, axis=1), SUPPORT_CHROMA_PERCENTILE))
    floor = float(np.percentile(lab[:, 0], SUPPORT_LIGHTNESS_PERCENTILE))
    return centre, radius, floor


def share_like_support(rgb, model):
    """Fraction of the (N, 3) colours that are coloured like the support."""
    lab = _srgb_to_lab(rgb)
    if not len(lab):
        return 0.0
    centre, radius, floor = model
    like = (np.linalg.norm(lab[:, 1:] - centre, axis=1) <= radius) & (lab[:, 0] >= floor)
    return float(like.mean())


def surface_samples(mesh, count=100_000, seed=0):
    """(points, colours): `count` points spread evenly over a mesh's surface,
    and the colour of each, from its texture or vertex colours (8-bit RGB).
    Area-weighted, so a big flat dish counts for its area, not its vertices."""
    import trimesh

    rng = np.random.default_rng(seed)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    faces = rng.choice(len(area), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    weights = np.stack([1 - u - v, u, v], axis=1)
    points = (triangles[faces] * weights[:, :, None]).sum(axis=1)

    colours = np.full((count, 3), 200, np.uint8)
    visual = mesh.visual
    corners = np.asarray(mesh.faces)[faces]
    if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.uv is not None:
        material = visual.material
        texture = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        if texture is not None:
            pixels = np.asarray(texture.convert("RGB"))
            uv = (np.asarray(visual.uv)[corners] * weights[:, :, None]).sum(axis=1)
            height, width = pixels.shape[:2]
            x = np.clip(((uv[:, 0] % 1.0) * (width - 1)).round().astype(int), 0, width - 1)
            y = np.clip(((1.0 - uv[:, 1] % 1.0) * (height - 1)).round().astype(int), 0, height - 1)
            colours = pixels[y, x]
    elif getattr(visual, "kind", None) == "vertex":
        vertex_colours = np.asarray(visual.vertex_colors)[:, :3].astype(np.float64)
        colours = (vertex_colours[corners] * weights[:, :, None]).sum(axis=1).round().astype(np.uint8)
    return points, colours


# (name, azimuth, elevation) in degrees, glTF frame: Y up, front toward +Z.
PREVIEW_VIEWS = (("front", 0, 30), ("side", 90, 10), ("top", 0, 85), ("underneath", 0, -45))


def preview_sheet(mesh, path, size=320, samples=150_000):
    """Write a PNG of a reconstruction from the front, the side, the top and
    underneath, side by side: a dish under the food shows from underneath."""
    from PIL import Image

    points, colours = surface_samples(mesh, samples)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    half = max(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) / 2, 1e-9) * 1.05
    tiles = []
    for _, azimuth, elevation in PREVIEW_VIEWS:
        a, e = np.radians(azimuth), np.radians(elevation)
        toward_eye = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
        right = np.cross([0.0, 1.0, 0.0], toward_eye)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(toward_eye, right)
        rel = points - centre
        px = ((rel @ right / half + 1) / 2 * (size - 1)).round().astype(int)
        py = ((1 - (rel @ up / half + 1) / 2) * (size - 1)).round().astype(int)
        nearness = rel @ toward_eye
        # Each point covers 2x2 pixels; every pixel shows its nearest point.
        offsets = [(dx, dy) for dx in (0, 1) for dy in (0, 1)]
        px = np.concatenate([px + dx for dx, _ in offsets])
        py = np.concatenate([py + dy for _, dy in offsets])
        near = np.tile(nearness, len(offsets))
        rgb = np.tile(colours, (len(offsets), 1))
        inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        index, near, rgb = (py * size + px)[inside], near[inside], rgb[inside]
        order = np.lexsort((-near, index))
        _, first = np.unique(index[order], return_index=True)
        tile = np.full((size * size, 3), 235, np.uint8)
        tile[index[order][first]] = rgb[order][first]
        tiles.append(tile.reshape(size, size, 3))
    Image.fromarray(np.hstack(tiles)).save(path)


def ab_summary_lines(rows, photo_share):
    """The --context-ab table: one line per reconstruction."""
    lines = [
        f"food pixels in the photo coloured like the support: {photo_share * 100:.1f}%",
        "",
        f"{'variant':16s} {'seed':>10s} {'like support':>13s} {'excess':>8s} {'height/width':>13s}  glb",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:16s} {row['seed']:>10d} {row['like_support'] * 100:>12.1f}% "
            f"{(row['like_support'] - photo_share) * 100:>+8.1f} {row['height_ratio']:>13.3f}  {row['glb']}"
        )
    return lines


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", required=True, help="photo: path or URL")
    parser.add_argument(
        "--masks", required=True, nargs="+",
        help="masks: paths or URLs; join several with commas to use them as one",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--tag", default="hf", help="checkpoint tag under checkpoints/")
    parser.add_argument("--full", action="store_true", help="also run full merged reconstructions")
    parser.add_argument("--per-object", action="store_true", help="also reconstruct each mask alone")
    parser.add_argument(
        "--context-ab", action="store_true",
        help="only the A/B: the food (first mask) as is, with its support (second mask) "
        "painted over, and as a cutout",
    )
    parser.add_argument("--ab-seeds", type=int, default=2, help="seeds per --context-ab variant, from --seeds")
    parser.add_argument(
        "--fill", choices=("background", "grey", "black"), default="background",
        help="what --context-ab paints with: the photo's median background colour, grey or black",
    )
    return parser.parse_args(argv)


def _run_context_ab(args, inference, pipeline, image, food, support, out_dir):
    """--context-ab: the food three ways, see the module docstring."""
    import torch
    from PIL import Image
    from request_utils import extract_metric_scale, resolve_postprocess_flags

    flags, warnings = resolve_postprocess_flags(os.environ)  # the server's own switches
    for warning in warnings:
        print(warning, flush=True)
    food, support = (_resized(mask, np.asarray(image).shape[:2]) for mask in (food, support))
    colour = (
        background_colour(image, [food, support]) if args.fill == "background"
        else np.array(FILL_COLOURS[args.fill], np.uint8)
    )
    rgb = np.asarray(image)[..., :3]
    model = support_colour_model(rgb[support])
    photo_share = share_like_support(rgb[food], model)

    # The depth model's output for the untouched photo, handed to every
    # variant: painting pixels must change what SAM3D sees, not where it
    # thinks the food is. Only the point colours follow the variant's photo.
    with pipeline.device:
        untouched = pipeline.compute_pointmap(inference.merge_mask_to_rgba(image, food))
    compute_pointmap = pipeline.compute_pointmap

    def untouched_depth(rgba, pointmap=None):
        depth = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in untouched.items()}
        depth["pts_color"] = torch.from_numpy(pipeline.image_to_float(rgba)).permute(2, 0, 1).contiguous()[:3]
        return depth

    rows = []
    pipeline.compute_pointmap = untouched_depth
    try:
        for name, picture in context_variants(image, food, support, colour).items():
            Image.fromarray(np.asarray(picture)[..., :3]).save(out_dir / f"ab_input_{name}.png")
            for seed in args.seeds[: args.ab_seeds]:
                started = time.time()
                output = inference(
                    picture, food, seed=seed,
                    with_mesh_postprocess=flags["with_mesh_postprocess"],
                    with_texture_baking=True,
                    with_layout_postprocess=flags["with_layout_postprocess"],
                    rendering_engine="nvdiffrast",
                )
                mesh = output["glb"]
                metric_scale = extract_metric_scale(output)
                if metric_scale is not None:
                    mesh.apply_scale(metric_scale)  # exactly as server.py exports it
                stem = f"ab_{name}_seed{seed}"
                mesh.export(str(out_dir / f"{stem}.glb"))
                preview_sheet(mesh, out_dir / f"{stem}.png")
                _, colours = surface_samples(mesh)
                extents = mesh.extents  # glTF, y-up
                rows.append({
                    "variant": name, "seed": seed, "glb": f"{stem}.glb", "preview": f"{stem}.png",
                    "like_support": round(share_like_support(colours, model), 4),
                    "height_ratio": round(float(extents[1] / max(extents[0], extents[2])), 4),
                    "seconds": round(time.time() - started, 1),
                })
                print(json.dumps(rows[-1]), flush=True)
                torch.cuda.empty_cache()
    finally:
        pipeline.compute_pointmap = compute_pointmap

    summary = {
        "image": args.image, "masks": args.masks, "fill": args.fill, "fill_colour": colour.tolist(),
        "postprocess": flags, "photo_like_support": round(photo_share, 4), "context_ab": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ab_summary_lines(rows, photo_share)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_dir}")


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
    masks = [
        np.logical_or.reduce([load_mask(_fetch(m, out_dir)) for m in group])
        for group in mask_groups(args.masks)
    ]
    merged = np.logical_or.reduce(masks)

    config = root / "checkpoints" / args.tag / "pipeline.yaml"
    print(f"Loading SAM3D from {config} ...", flush=True)
    inference = Inference(str(config), compile=False)
    pipeline = inference._pipeline
    if args.context_ab:
        if len(masks) != 2:
            raise SystemExit("--context-ab needs two --masks: the food, then what it sits on")
        _run_context_ab(args, inference, pipeline, image, masks[0], masks[1], out_dir)
        return
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
