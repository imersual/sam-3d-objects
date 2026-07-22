#!/usr/bin/env python
"""
set_depth_model.py
==================
Rewrite the ``depth_model`` block of a checkpoint's ``pipeline.yaml``.

``checkpoints/<tag>/pipeline.yaml`` is downloaded from ``facebook/sam-3d-objects``
and overwritten every time the checkpoint is (re)downloaded, so the depth model
choice cannot simply live in a committed file. ``setup-gpu-server.sh`` calls this
script right after the download to re-apply it.

Both variants are written explicitly, so this flips the config in either
direction without re-downloading the checkpoint. It is idempotent.

Usage
-----
    python scripts/set_depth_model.py --tag hf --variant moge2   # default
    python scripts/set_depth_model.py --tag hf --variant moge1   # rollback
"""
import argparse
import os
import sys

from omegaconf import OmegaConf

DEPTH_MODELS = {
    "moge2": {
        "_target_": "sam3d_objects.pipeline.depth_models.moge2.MoGe2",
        "model": {
            "_target_": "moge.model.v2.MoGeModel.from_pretrained",
            "pretrained_model_name_or_path": "Ruicheng/moge-2-vitl-normal",
        },
    },
    "moge1": {
        "_target_": "sam3d_objects.pipeline.depth_models.moge.MoGe",
        "model": {
            "_target_": "moge.model.v1.MoGeModel.from_pretrained",
            "pretrained_model_name_or_path": "Ruicheng/moge-vitl",
        },
    },
}


def set_depth_model(config, variant):
    """Replace ``config.depth_model`` with ``variant``; mutate and return config.

    Raises ValueError on an unknown variant rather than silently leaving the
    config on the previous model.
    """
    if variant not in DEPTH_MODELS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(DEPTH_MODELS)}"
        )
    config["depth_model"] = OmegaConf.create(DEPTH_MODELS[variant])
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", default="hf", help="Checkpoint tag (subfolder under checkpoints/)"
    )
    parser.add_argument(
        "--variant",
        default=os.environ.get("SAM3D_DEPTH_MODEL", "moge2"),
        choices=sorted(DEPTH_MODELS),
        help="Depth model to write (default: $SAM3D_DEPTH_MODEL, else moge2)",
    )
    parser.add_argument(
        "--checkpoints-root",
        default="checkpoints",
        help="Directory containing <tag>/pipeline.yaml",
    )
    args = parser.parse_args(argv)

    path = os.path.join(args.checkpoints_root, args.tag, "pipeline.yaml")
    if not os.path.exists(path):
        sys.exit(f"[set_depth_model] not found: {path}")

    config = OmegaConf.load(path)
    set_depth_model(config, args.variant)
    OmegaConf.save(config, path)

    model_id = DEPTH_MODELS[args.variant]["model"]["pretrained_model_name_or_path"]
    print(f"[set_depth_model] {path} -> {args.variant} ({model_id})")


if __name__ == "__main__":
    main()
