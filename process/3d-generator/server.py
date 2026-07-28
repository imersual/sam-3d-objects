#!/usr/bin/env python
"""
SAM3D persistent inference server.

Loads the model ONCE on startup, then handles inference requests via HTTP.
This eliminates per-task startup overhead (conda activation, torch import,
model loading) that previously happened on every subprocess call.

Usage:
    python server.py [--port 8000] [--host 0.0.0.0] [--tag hf]

The poller should POST to /infer instead of calling run.sh.
"""
import argparse
import os
import sys
import random
import logging
from pathlib import Path

# Reduce memory fragmentation. Must be set before torch is imported.
# Can also be set via PYTORCH_CUDA_ALLOC_CONF env var in start_server.sh.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── path setup (mirror run_inference.py) ────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
_NOTEBOOK = os.path.join(_ROOT, "notebook")
sys.path.insert(0, _NOTEBOOK)
sys.path.insert(0, _ROOT)

# ── third-party ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ── SAM3D ────────────────────────────────────────────────────────────────────
import torch
import numpy as np
from inference import Inference, load_image, load_mask
from request_utils import normalize_views, extract_metric_scale

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sam3d-server")

# ── argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SAM3D inference server")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument(
    "--tag", default="hf", help="Checkpoint tag (subfolder under checkpoints/)"
)
args, _unknown = parser.parse_known_args()

# ── model loading (happens ONCE at startup) ───────────────────────────────────
config_path = os.path.join(_ROOT, "checkpoints", args.tag, "pipeline.yaml")
log.info(f"Loading model from: {config_path}")
log.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
if torch.cuda.is_available():
    log.info(f"Using GPU: {torch.cuda.get_device_name(0)} | "
             f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
_inference = Inference(config_path, compile=False)
log.info("Model loaded and ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SAM3D Inference Server")


class ViewInput(BaseModel):
    image_path: str
    mask_paths: list[str]


class InferRequest(BaseModel):
    image_path: str | None = None
    mask_paths: list[str] | None = None
    output_path: str
    seed: int | None = None  # omit to use a random seed
    # Multiview: one entry per photo of the same object.
    # Takes precedence over image_path/mask_paths when provided.
    views: list[ViewInput] | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
def infer(req: InferRequest):
    # ── validate inputs ───────────────────────────────────────────────────────
    try:
        view_specs = normalize_views(
            req.image_path,
            req.mask_paths,
            [v.model_dump() for v in req.views] if req.views else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    for image_path, mask_paths in view_specs:
        if not Path(image_path).is_file():
            raise HTTPException(
                status_code=400, detail=f"image_path not found: {image_path}"
            )
        for mp in mask_paths:
            if not Path(mp).is_file():
                raise HTTPException(
                    status_code=400, detail=f"mask_path not found: {mp}"
                )

    output_path = Path(req.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    log.info(f"Inference request | views={len(view_specs)} seed={seed}")

    try:
        images, masks = [], []
        for image_path, mask_paths in view_specs:
            image = load_image(image_path)
            view_masks = [load_mask(mp) for mp in mask_paths]
            mask = view_masks[0].copy()
            for m in view_masks[1:]:
                mask |= m
            images.append(image)
            masks.append(mask)

        if len(view_specs) == 1:
            output = _inference(
                images[0],
                masks[0],
                seed=seed,
                with_mesh_postprocess=True,
                with_texture_baking=True,
                with_layout_postprocess=True,
                rendering_engine="nvdiffrast",
            )
        else:
            # Layout postprocess is not supported in multi-view mode: it aligns
            # the object into one view's scene frame, which is ambiguous with
            # several views. The metric scale still comes through — it is
            # decoded per view against that view's pointmap and combined.
            output = _inference.multi_view(
                images,
                masks,
                seed=seed,
                with_mesh_postprocess=True,
                with_texture_baking=True,
                rendering_engine="nvdiffrast",
            )

        metric_scale = extract_metric_scale(output)

        mesh = output["glb"]
        if metric_scale is not None:
            # SAM3D always emits the mesh normalized to a [-0.5, 0.5] cube
            # (longest side = 1.0). Bake in the predicted size so the exported
            # file is in metres.
            mesh.apply_scale(metric_scale)
            log.info(
                f"Applied metric scale {metric_scale:.4f} | "
                f"bbox (m) = {np.round(mesh.extents, 4).tolist()}"
            )
        else:
            log.warning(
                "No metric scale available; exporting unit-cube mesh "
                "(longest side = 1.0)"
            )
        mesh.export(str(output_path))
        log.info(f"Exported mesh to: {output_path}")

    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Release fragmented reserved-but-unallocated memory back to CUDA.
        torch.cuda.empty_cache()

    return {
        "output_path": str(output_path),
        "seed": seed,
        "views": len(view_specs),
        # Metres per unit-cube unit, already baked into the exported mesh.
        # None means the mesh is still unit-cube sized (the layout head
        # produced nothing usable).
        "metric_scale": metric_scale,
    }


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
