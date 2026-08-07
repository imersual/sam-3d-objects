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
from request_utils import (
    normalize_views,
    extract_metric_scale,
    extract_pose,
    extract_intrinsics,
    normal_map_sidecar_path,
)

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
    # Opt-in: dump MoGe-2's full-resolution per-pixel normal map (HxWx3
    # float32, likely several MB) to a .npy file beside output_path and
    # return its path as normal_map_path. Ignored (stays null) for
    # multiview requests and when the depth model has no normal head.
    # Default False: callers that don't ask for it see no new file and no
    # extra work, exactly as before this field existed.
    return_normal_map: bool = False


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
            # with_layout_postprocess=True solves the object's pose in the
            # source photo's metric camera frame; surface it (see extract_pose)
            # instead of discarding it as before.
            pose = extract_pose(output)
            # MoGe-2's normalized camera intrinsics for this same photo (see
            # extract_intrinsics). Additive/informational, like pose.
            intrinsics = extract_intrinsics(output)
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
            # No layout postprocess in multi-view -> no camera-frame pose.
            # Nulls, not an invented pose from an arbitrary view.
            pose = extract_pose(None)
            # Same reasoning as pose: run_multi_view computes one intrinsics
            # matrix per view internally and keeps none of them, rather than
            # picking one view's frame arbitrarily. Null, not invented.
            intrinsics = extract_intrinsics(None)

        metric_scale = extract_metric_scale(output)

        mesh = output["glb"]
        if metric_scale is not None:
            # SAM3D always emits the mesh normalized to a [-0.5, 0.5] cube
            # (longest side = 1.0). Bake in the predicted size so the exported
            # file is in metres.
            #
            # TRAP for a future caller that also consumes `pose` below: do
            # NOT apply both `metric_scale` (mesh.apply_scale, right here)
            # and `pose["scale"]` to the same mesh. pose["scale"] already
            # contains this same metric scale (it's the unrounded per-axis
            # source `metric_scale` is averaged from) -- applying both
            # squares it. Placing a mesh with `pose` must start from the
            # mesh AS EXPORTED (already metric-scaled here) and apply only
            # pose["rotation"] + pose["translation"] on top.
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

        # Opt-in (req.return_normal_map): dump MoGe-2's per-pixel normal map
        # next to the GLB and return its path, instead of inlining a
        # full-resolution HxWx3 float array as JSON (easily several MB per
        # request -- see request_utils.normal_map_sidecar_path / the /infer
        # docstring in this module's design notes). Not meaningful for
        # multiview (no single camera frame), so it stays null there
        # regardless of the flag, same as pose/intrinsics above.
        normal_map_path = None
        if req.return_normal_map and len(view_specs) == 1:
            normal_tensor = output.get("normal")
            if normal_tensor is None:
                log.info(
                    "return_normal_map=True but the depth model produced no "
                    "normal map (checkpoint has no normal head); returning null."
                )
            else:
                try:
                    candidate_path = normal_map_sidecar_path(str(output_path))
                    np.save(
                        candidate_path,
                        normal_tensor.detach().cpu().numpy().astype(np.float32),
                    )
                    normal_map_path = candidate_path
                    log.info(f"Wrote normal map to: {normal_map_path}")
                except Exception as exc:
                    # Non-fatal: the mesh above already reconstructed
                    # successfully and is the primary deliverable. Losing this
                    # optional sidecar must not turn a good reconstruction
                    # into a 500.
                    log.warning(f"Failed to write normal map (non-fatal): {exc}")
        elif req.return_normal_map:
            log.info(
                "return_normal_map=True has no effect for multiview requests "
                "(no single camera frame to express a normal map in)."
            )

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
        # The camera-frame pose the pipeline solves for (with_layout_postprocess
        # single-view requests only; see extract_pose). Additive / informational
        # only -- the exported mesh above is unaffected and stays in its own
        # object-local frame, exactly as before this field existed. All entries
        # are None for multiview requests (no layout postprocess is run).
        "pose": pose,
        # MoGe-2's normalized camera intrinsics for the source photo (see
        # extract_intrinsics for the exact convention and how to convert to
        # pixels). Additive / informational, same caveats as pose: None for
        # multiview requests. NOTE: not confirmed to share pose's axis
        # convention -- see extract_intrinsics's docstring before combining
        # the two.
        "intrinsics": intrinsics,
        # Path to MoGe-2's full-resolution per-pixel normal map (float32
        # .npy, HxWx3, MoGe's own camera-space convention -- same one
        # `intrinsics` above uses), written only when the request set
        # return_normal_map=True. None otherwise: not requested, multiview,
        # the checkpoint has no normal head, or the write failed (logged,
        # non-fatal).
        "normal_map_path": normal_map_path,
    }


# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
