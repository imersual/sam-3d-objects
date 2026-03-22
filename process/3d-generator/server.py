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
from inference import Inference, load_image, load_mask

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
_inference = Inference(config_path, compile=False)
log.info("Model loaded and ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SAM3D Inference Server")


class InferRequest(BaseModel):
    image_path: str
    mask_path: str
    output_path: str
    seed: int | None = None  # omit to use a random seed


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
def infer(req: InferRequest):
    # ── validate inputs ───────────────────────────────────────────────────────
    if not Path(req.image_path).is_file():
        raise HTTPException(
            status_code=400, detail=f"image_path not found: {req.image_path}"
        )
    if not Path(req.mask_path).is_file():
        raise HTTPException(
            status_code=400, detail=f"mask_path not found: {req.mask_path}"
        )

    output_path = Path(req.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    log.info(
        f"Inference request | image={req.image_path} mask={req.mask_path} seed={seed}"
    )

    try:
        image = load_image(req.image_path)
        mask = load_mask(req.mask_path)

        output = _inference(
            image,
            mask,
            seed=seed,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=True,
            rendering_engine="nvdiffrast",
        )

        mesh = output["glb"]
        mesh.export(str(output_path))
        log.info(f"Exported mesh to: {output_path}")

    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"output_path": str(output_path), "seed": seed}


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
