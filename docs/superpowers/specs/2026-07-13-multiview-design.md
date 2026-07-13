# Multi-View 3D Reconstruction — Design

**Date:** 2026-07-13
**Source:** Port of [facebookresearch/sam-3d-objects PR #37](https://github.com/facebookresearch/sam-3d-objects/pull/37) ("Support multi-view inference"), adapted to this fork.

## Goal

Uploading more than one photo of the same object produces a better final 3D
model than any single photo. The upstream PR achieves this training-free via
**multidiffusion**: one shared latent is denoised while, at every diffusion
step, the denoiser runs once per view (each conditioned on that view's image)
and the predictions are averaged. No new model weights are needed.

## Decisions made during brainstorming

| Question | Decision |
|---|---|
| Where does multiview live? | Both: a standalone CLI (`run_inference.py`) and batch-pipeline integration. |
| Pointmaps for multiview | Internal (pipeline computes a MoGe pointmap per view). External per-view pointmaps from the depth backends are a **follow-up**, not in scope. |
| Masks for extra views | User provides a mask per view (`<stem>.png` + `<stem>_mask.png`). No auto-segmentation. |
| Batch behavior for multiview folders | Multiview only: one fused GLB; the per-backend single-view comparison runs are skipped. |
| Porting style | Cleaned-up port: English comments, trimmed debug logging, reuse of this fork's existing helpers. Fusion math stays identical to the PR. |
| Output settings | Full settings immediately: nvdiffrast + texture baking + mesh postprocess (the PR author only validated pytorch3d + vertex colors; if texture baking misbehaves, fall back to vertex colors). |

## Components

### 1. `sam3d_objects/pipeline/multi_view_utils.py` (new)

Context manager `inject_generator_multi_view(generator, num_views, num_steps,
mode)`. Temporarily replaces `generator._generate_dynamics`:

- `multidiffusion` (default): each step runs the original dynamics once per
  view condition and averages the predictions (handles tensor / tuple / dict
  prediction types, same as the PR).
- `stochastic`: each step uses one view's condition, rotating round-robin.
  Cheaper, lower quality; kept as an option.

Restores the original dynamics in a `finally` block. English docstrings; the
PR's one-time shape-dump logging becomes `logger.debug`. **The fusion math is
byte-identical to the PR.**

### 2. `sam3d_objects/pipeline/inference_pipeline.py` (extend)

Three methods added to `InferencePipeline` (inherited by
`InferencePipelinePointMap`):

- `get_multi_view_condition_input(condition_embedder, view_input_dicts,
  input_mapping)` — embeds each view's condition and stacks tokens to
  `(num_views, B, tokens, dim)`.
- `sample_sparse_structure_multi_view(...)` / `sample_slat_multi_view(...)` —
  Stage 1 / Stage 2 sampling wrapped in the injection context manager.
  Logic identical to the PR.
- `run_multi_view(view_images, view_masks, ...)` — orchestrator. Cleaned-up
  differences vs the PR:
  - Reuses existing `self.merge_image_and_mask()` per view instead of the
    PR's ad-hoc RGBA merging block.
  - New param `rendering_engine: str = "nvdiffrast"`, passed through to this
    fork's 5-arg `postprocess_slat_output()` (the PR passes only 4 args).
  - Defaults `with_mesh_postprocess=True, with_texture_baking=True,
    use_vertex_color=False` to match this fork's batch settings.
  - Keeps the `hasattr(self, "compute_pointmap")` branch: on the pointmap
    pipeline, each view gets an internally computed MoGe pointmap for Stage 1
    preprocessing.
  - **No layout postprocess in multiview** (deliberate): layout postprocess
    aligns the object into one view's scene frame, which is ambiguous with N
    views; batch export only uses the object-space GLB anyway.

### 3. `sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py` (patch)

Port the PR's guard: skip `fill_holes` when `rendering_engine ==
"pytorch3d"` (fill_holes requires nvdiffrast), with a warning. Inert for
nvdiffrast runs; prevents a crash if the engine is ever switched.

### 4. `notebook/inference.py` (extend)

New public method `Inference.multi_view(images, masks, seed=None, **kwargs)`
alongside `__call__`. Both entry points (CLI and batch) call this method;
nothing outside `notebook/inference.py` touches `_pipeline` directly.

### 5. `run_inference.py` (new, repo root) + `notebook/load_images_and_masks.py` (new)

Cleaned port of the PR's CLI and loader.

- Loader supports the PR's two folder layouts:
  1. Flat: `<dir>/<stem>.png` + `<dir>/<stem>_mask.png` (also `.jpg`).
  2. Split: `<dir>/images/<stem>.png` + `<dir>/<mask_prompt>/<stem>[_mask].png`.
  Masks are RGBA with the mask in the alpha channel (alpha>0 = object), with
  the same RGB fallback warnings as the PR.
- CLI auto-selects single-view (1 image → `Inference.__call__`) vs multiview
  (2+ → `Inference.multi_view`).
- Flags: `--input_path` (required), `--mask_prompt`, `--image_names`,
  `--seed` (default 42), `--stage1_steps` (50), `--stage2_steps` (25),
  `--decode_formats` (gaussian,mesh), `--model_tag` (hf),
  `--rendering_engine` (nvdiffrast), `--no_texture_baking`,
  `--no_mesh_postprocess`, `--out_dir` (all flags use the PR's underscore
  style for consistency).
- Output defaults to `output/multiview/<name>/` (this repo's `output/`
  convention, not the PR's `visualization/`): `result.glb` + `result.ply`.

### 6. Batch integration (`batch/`)

- **Input convention:** a folder `input/images/<object>/` containing **2+
  view pairs** (`<stem>.png|.jpg` + `<stem>_mask.png`) is a multiview folder.
  The legacy single-view layout (`image.jpg` + `mask.png`) is untouched and
  detected first.
- **`run_batch.sh`:** new `detect_multiview` helper. Multiview folders skip
  depth-backend stages 1–4 (pointmaps are computed internally per view) and
  go straight to the SAM3D stage, invoked with the new flag.
- **`batch_sam3d.py`:** new `--views-dir <dir>` mode: load all view pairs via
  the shared loader, call `inference.multi_view(...)` with full settings
  (nvdiffrast, texture baking, mesh postprocess, no layout postprocess),
  export `output/images/<object>/splat_multiview.glb`. Existing
  `--image/--mask` single-view mode unchanged.
- **`config.sh`:** new knob `SAM3D_MULTIVIEW=1` (set `0` to disable
  detection and treat every folder as single-view).

## Data flow (multiview batch)

```
input/images/<object>/{1.png,1_mask.png,2.png,2_mask.png,...}
  → run_batch.sh: detect_multiview → skip depth backends
  → batch_sam3d.py --views-dir
      → load_images_and_masks (N images, N bool masks)
      → Inference.multi_view
          → run_multi_view: per view merge_image_and_mask → compute_pointmap
            → preprocess (ss + slat input dicts)
          → Stage 1: sample_sparse_structure_multi_view (multidiffusion) → coords
          → Stage 2: sample_slat_multi_view (multidiffusion) → slat
          → decode_slat → postprocess (nvdiffrast, texture baking)
  → output/images/<object>/splat_multiview.glb
```

## Error handling

- Multiview mode with fewer than 2 valid image/mask pairs → clear error
  listing which stems were found and which masks are missing (the likely user
  mistake), non-zero exit; `run_batch.sh` logs and continues to the next
  sample, consistent with existing stages.
- Mask/image size mismatch → resized during merge (existing
  `merge_image_and_mask` behavior).
- Mask sanity check (coverage ~0% or ~100%) applied per view, mirroring the
  existing single-view warning in `batch_sam3d.py`.
- Runtime scales ~linearly with view count (N sequential denoiser calls per
  step; no meaningful extra VRAM). Log a note when N > 8.

## Out of scope (agreed follow-ups)

1. External per-view pointmaps from the depth backends (depthpro/da3/lotus2/
   moge2) passed into `run_multi_view` — enables backend comparison in
   multiview mode.
2. Auto-generated masks (rembg/SAM) for views without a mask.
3. Layout postprocess for multiview.

## Verification plan

Executed on the remote Linux `/workspace` environment (`sam3d-objects` env):

1. Fetch the PR branch's 8-view stuffed-toy example data as a known-good
   fixture; run the standalone CLI; confirm a sane `result.glb` (red collar
   present — the PR's own success criterion vs single view).
2. Run the same views as a batch folder; confirm `splat_multiview.glb`.
3. Regression: run one existing single-image folder through the batch and
   confirm outputs are unchanged.
4. Full-settings experiment: if texture baking fails or produces artifacts in
   multiview, rerun with `use_vertex_color=True` / no baking and record the
   result — that decides whether baking needs the follow-up treatment.
