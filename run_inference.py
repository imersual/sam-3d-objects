"""SAM 3D Objects inference CLI — single-view and multi-view reconstruction.

Ported from upstream PR #37 and adapted to this fork (nvdiffrast + texture
baking by default, outputs under output/multiview/).

Examples:
    # Multi-view: images and masks side by side (1.png + 1_mask.png, ...)
    python run_inference.py --input_path ./data/images_and_masks

    # Single view: pick one image by stem
    python run_inference.py --input_path ./data/images_and_masks --image_names 1

    # Multi-view, split layout (images/ + stuffed_toy/)
    python run_inference.py --input_path ./data --mask_prompt stuffed_toy

    # Subset of views, custom output dir
    python run_inference.py --input_path ./data --mask_prompt stuffed_toy \
        --image_names 1,2,5 --out_dir output/my_test
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "notebook"))

from loguru import logger


def parse_image_names(image_names_str: Optional[str]) -> Optional[List[str]]:
    """'a, b,c' -> ['a', 'b', 'c']; None/empty -> None (= use all images)."""
    if not image_names_str:
        return None
    names = [x.strip() for x in image_names_str.split(",") if x.strip()]
    return names or None


def parse_decode_formats(formats_str: str) -> List[str]:
    formats = [f.strip() for f in formats_str.split(",") if f.strip()]
    return formats or ["gaussian", "mesh"]


def default_out_dir(
    input_path: Path,
    mask_prompt: Optional[str],
    image_names: Optional[List[str]],
    is_single_view: bool,
) -> Path:
    """output/multiview/<prompt-or-dirname>[_<names>|_multiview]"""
    base = mask_prompt if mask_prompt else input_path.name
    if image_names:
        safe = [n.replace("/", "_").replace("\\", "_") for n in image_names]
        suffix = "_".join(safe[:3])
        if len(safe) > 3:
            suffix += f"_and_{len(safe) - 3}_more"
        name = f"{base}_{suffix}"
    elif is_single_view:
        name = f"{base}_single"
    else:
        name = f"{base}_multiview"
    return Path("output") / "multiview" / name


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects inference - single-view and multi-view",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Input directory. With --mask_prompt: images in <input_path>/images/, "
        "masks in <input_path>/<mask_prompt>/. Without: <stem>.png + "
        "<stem>_mask.png side by side in <input_path>.",
    )
    parser.add_argument(
        "--mask_prompt",
        default=None,
        help="Mask folder name for the split layout (default: flat layout)",
    )
    parser.add_argument(
        "--image_names",
        default=None,
        help="Comma-separated image stems, e.g. '1,2,5'. Default: all images. "
        "A single name runs single-view inference.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_steps", type=int, default=50)
    parser.add_argument("--stage2_steps", type=int, default=25)
    parser.add_argument("--decode_formats", default="gaussian,mesh")
    parser.add_argument("--model_tag", default="hf")
    parser.add_argument(
        "--rendering_engine",
        default="nvdiffrast",
        choices=["nvdiffrast", "pytorch3d"],
    )
    parser.add_argument("--no_texture_baking", action="store_true")
    parser.add_argument("--no_mesh_postprocess", action="store_true")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: output/multiview/<derived name>)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    image_names = parse_image_names(args.image_names)
    decode_formats = parse_decode_formats(args.decode_formats)

    # Heavy imports deferred so the module stays importable for helper tests.
    from inference import Inference
    from load_images_and_masks import load_images_and_masks_from_path

    images, masks, names = load_images_and_masks_from_path(
        input_path, mask_prompt=args.mask_prompt, image_names=image_names
    )
    is_single_view = len(images) == 1

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else default_out_dir(input_path, args.mask_prompt, image_names, is_single_view)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    config_path = _HERE / "checkpoints" / args.model_tag / "pipeline.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    logger.info(f"Loading model: {config_path}")
    inference = Inference(str(config_path), compile=False)

    if is_single_view:
        logger.info(f"Single-view inference (view '{names[0]}')")
        result = inference(
            images[0],
            masks[0],
            seed=args.seed,
            with_mesh_postprocess=not args.no_mesh_postprocess,
            with_texture_baking=not args.no_texture_baking,
            rendering_engine=args.rendering_engine,
        )
    else:
        logger.info(f"Multi-view inference with {len(images)} views: {names}")
        result = inference.multi_view(
            images,
            masks,
            seed=args.seed,
            with_mesh_postprocess=not args.no_mesh_postprocess,
            with_texture_baking=not args.no_texture_baking,
            rendering_engine=args.rendering_engine,
            stage1_inference_steps=args.stage1_steps,
            stage2_inference_steps=args.stage2_steps,
            decode_formats=decode_formats,
        )

    saved = []
    if result.get("glb") is not None:
        glb_path = out_dir / "result.glb"
        result["glb"].export(str(glb_path))
        saved.append(str(glb_path))
    if result.get("gs") is not None:
        ply_path = out_dir / "result.ply"
        result["gs"].save_ply(str(ply_path))
        saved.append(str(ply_path))

    if not saved:
        logger.error("No exportable outputs in result (no 'glb' or 'gs' key)")
        sys.exit(1)
    logger.info("Saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
