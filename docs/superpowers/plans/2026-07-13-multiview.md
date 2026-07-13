# Multi-View Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port upstream PR #37's training-free multi-view (multidiffusion) reconstruction into this fork, exposed via `Inference.multi_view()`, a standalone `run_inference.py` CLI, and the batch pipeline.

**Architecture:** A context manager (`multi_view_utils.py`) hot-swaps the diffusion generator's `_generate_dynamics` so every denoising step runs once per view condition and averages the predictions. Four methods added to `InferencePipeline` orchestrate per-view preprocessing and the two sampling stages. A lightweight loader + CLI and a batch-pipeline branch sit on top.

**Tech Stack:** Python 3.10+ (remote conda env `sam3d-objects`), PyTorch, loguru, PIL, numpy, bash (batch orchestration), pytest for CPU-testable units.

**Spec:** `docs/superpowers/specs/2026-07-13-multiview-design.md` — read it before starting.

## Global Constraints

- **Execution environment:** code is edited on Windows but GPU inference runs on a remote Linux box under `/workspace` (env `sam3d-objects`). Locally you can run: `python -m py_compile`, `pytest` for CPU-only tests (local Python 3.11 has numpy/PIL/torch-cpu; Task 1 installs pytest+loguru), and `bash -n`. Anything needing the model/GPU is a **remote verification step** (Task 10).
- **Fusion math must stay identical to PR #37** — cleanups may rename/restructure but not change tensor operations.
- Layout postprocess is NOT supported in multiview (deliberate; see spec).
- Multiview defaults at the `Inference` API level match this fork's batch settings: `with_mesh_postprocess=True, with_texture_baking=True, use_vertex_color=True, rendering_engine="nvdiffrast"`.
- `run_inference.py` CLI flags use underscore style (`--no_texture_baking`); `batch_sam3d.py` keeps its existing dash style (`--views-dir`).
- Work on branch `multiview`. Commit after every task.
- All Python indentation: 4 spaces. Methods added to `InferencePipeline` are indented one level (4 spaces) as class members.

---

### Task 1: `multi_view_utils.py` — the fusion context manager (TDD)

**Files:**
- Create: `sam3d_objects/pipeline/multi_view_utils.py`
- Test: `tests/test_multi_view_utils.py`

**Interfaces:**
- Produces: `inject_generator_multi_view(generator, num_views: int, num_steps: int, mode: Literal["stochastic", "multidiffusion"] = "multidiffusion")` — a `@contextmanager`. Inside the `with` block, `generator._generate_dynamics(x_t, t, *args, **kwargs)` fuses per-view predictions; on exit (even on exception) the original function is restored. Task 2 imports it as `from sam3d_objects.pipeline.multi_view_utils import inject_generator_multi_view`.

- [ ] **Step 1: Install local test dependencies**

Run: `python -m pip install --user pytest loguru`
Expected: both install (or "already satisfied"). Verify: `python -m pytest --version` prints a version.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_multi_view_utils.py`:

```python
"""CPU-only tests for the multi-view fusion context manager.

The module is loaded directly from its file to avoid importing the heavy
sam3d_objects package (spconv/CUDA deps not present locally).
"""
import importlib.util
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "multi_view_utils",
    _ROOT / "sam3d_objects" / "pipeline" / "multi_view_utils.py",
)
multi_view_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(multi_view_utils)
inject_generator_multi_view = multi_view_utils.inject_generator_multi_view


class FakeGenerator:
    """Mimics a generator: prediction = x_t * cond, records each cond used."""

    def __init__(self):
        self.calls = []

    def _generate_dynamics(self, x_t, t, cond):
        self.calls.append(cond)
        return x_t * cond


class FakeGeneratorWithFlag:
    """Generator whose first conditional arg is a scalar flag (cond_idx=1 case)."""

    def __init__(self):
        self.calls = []

    def _generate_dynamics(self, x_t, t, flag, cond):
        self.calls.append((flag, cond))
        return x_t * cond


class DictPredGenerator:
    """Generator returning dict predictions (MM-DiT style)."""

    def _generate_dynamics(self, x_t, t, cond):
        return {"shape": x_t * cond}


def test_multidiffusion_averages_predictions():
    gen = FakeGenerator()
    conds = torch.tensor([1.0, 2.0, 3.0])  # shape[0] == num_views -> split per view
    with inject_generator_multi_view(gen, num_views=3, num_steps=10):
        out = gen._generate_dynamics(torch.tensor(2.0), 0.5, conds)
    # mean(2*1, 2*2, 2*3) = 4.0
    assert torch.isclose(out, torch.tensor(4.0))
    assert len(gen.calls) == 3


def test_multidiffusion_skips_scalar_first_arg():
    gen = FakeGeneratorWithFlag()
    conds = torch.tensor([1.0, 3.0])
    with inject_generator_multi_view(gen, num_views=2, num_steps=4):
        out = gen._generate_dynamics(torch.tensor(1.0), 0.5, 7, conds)
    assert torch.isclose(out, torch.tensor(2.0))  # mean(1*1, 1*3)
    assert all(flag == 7 for flag, _ in gen.calls)


def test_multidiffusion_dict_predictions():
    gen = DictPredGenerator()
    conds = torch.tensor([2.0, 4.0])
    with inject_generator_multi_view(gen, num_views=2, num_steps=4):
        out = gen._generate_dynamics(torch.tensor(1.0), 0.5, conds)
    assert torch.isclose(out["shape"], torch.tensor(3.0))  # mean(2, 4)


def test_multidiffusion_same_condition_for_all_views_fallback():
    # cond tensor whose shape[0] != num_views -> reused for every view
    gen = FakeGenerator()
    conds = torch.tensor([5.0, 5.0, 5.0])  # 3 entries, but num_views=2
    with inject_generator_multi_view(gen, num_views=2, num_steps=4):
        out = gen._generate_dynamics(torch.tensor(1.0), 0.5, conds)
    assert out.shape == conds.shape
    assert len(gen.calls) == 2


def test_stochastic_rotates_views_round_robin():
    gen = FakeGenerator()
    conds = torch.tensor([[1.0], [2.0]])  # 2 views
    with inject_generator_multi_view(gen, num_views=2, num_steps=4, mode="stochastic"):
        for _ in range(4):
            gen._generate_dynamics(torch.tensor(1.0), 0.5, conds)
    assert [c.item() for c in gen.calls] == [1.0, 2.0, 1.0, 2.0]


def test_restores_original_dynamics():
    gen = FakeGenerator()
    original = gen._generate_dynamics
    with inject_generator_multi_view(gen, num_views=2, num_steps=4):
        assert gen._generate_dynamics is not original
    assert gen._generate_dynamics is original


def test_restores_original_dynamics_on_exception():
    gen = FakeGenerator()
    original = gen._generate_dynamics
    with pytest.raises(RuntimeError):
        with inject_generator_multi_view(gen, num_views=2, num_steps=4):
            raise RuntimeError("boom")
    assert gen._generate_dynamics is original


def test_unsupported_mode_raises():
    gen = FakeGenerator()
    with pytest.raises(ValueError):
        with inject_generator_multi_view(gen, num_views=2, num_steps=4, mode="bogus"):
            pass
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_multi_view_utils.py -v`
Expected: FAIL at module load — `FileNotFoundError` (multi_view_utils.py does not exist yet).

- [ ] **Step 4: Write the implementation**

Create `sam3d_objects/pipeline/multi_view_utils.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Training-free multi-view fusion utilities for SAM 3D Objects.

Ported from facebookresearch/sam-3d-objects PR #37 (a multidiffusion
approach adapted from TRELLIS). The fusion math is intentionally identical
to the PR; only naming, comments, and logging were cleaned up.
"""
from contextlib import contextmanager
from typing import Literal

import torch
from loguru import logger


@contextmanager
def inject_generator_multi_view(
    generator,
    num_views: int,
    num_steps: int,
    mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
):
    """Temporarily patch ``generator._generate_dynamics`` for multi-view sampling.

    In ``multidiffusion`` mode every diffusion step runs the original
    dynamics once per view condition and averages the predictions, so one
    shared latent is guided by all views simultaneously. In ``stochastic``
    mode each step uses a single view's condition, rotating round-robin
    (cheaper, lower quality).

    The condition tokens are expected either stacked in a tensor of shape
    ``(num_views, ...)`` or as a list/tuple with one entry per view (see
    ``InferencePipeline.get_multi_view_condition_input``). The original
    dynamics function is restored on exit, including on exceptions.
    """
    original_dynamics = generator._generate_dynamics

    if mode == "stochastic":
        if num_views > num_steps:
            logger.warning(
                f"Number of views ({num_views}) exceeds number of steps "
                f"({num_steps}); some views will never be sampled."
            )

        cond_indices = (torch.arange(num_steps) % num_views).tolist()
        step_counter = [0]

        def _dynamics_stochastic(x_t, t, *args_conditionals, **kwargs_conditionals):
            cond_idx = cond_indices[step_counter[0] % len(cond_indices)]
            step_counter[0] += 1

            if len(args_conditionals) > 0:
                cond_tokens = args_conditionals[0]
                if isinstance(cond_tokens, (list, tuple)):
                    cond_i = (
                        cond_tokens[cond_idx : cond_idx + 1]
                        if isinstance(cond_tokens[0], torch.Tensor)
                        else [cond_tokens[cond_idx]]
                    )
                    new_args = (cond_i,) + args_conditionals[1:]
                elif (
                    isinstance(cond_tokens, torch.Tensor)
                    and cond_tokens.shape[0] == num_views
                ):
                    cond_i = cond_tokens[cond_idx : cond_idx + 1]
                    new_args = (cond_i,) + args_conditionals[1:]
                else:
                    new_args = args_conditionals
            else:
                new_args = args_conditionals

            return original_dynamics(x_t, t, *new_args, **kwargs_conditionals)

        generator._generate_dynamics = _dynamics_stochastic

    elif mode == "multidiffusion":

        def _dynamics_multidiffusion(x_t, t, *args_conditionals, **kwargs_conditionals):
            # Locate the condition tokens among the positional conditionals:
            # some generators prepend a scalar (e.g. a flag) before the tokens.
            cond_idx = 0
            if len(args_conditionals) > 0:
                first = args_conditionals[0]
                if isinstance(first, (int, float)) or (
                    isinstance(first, torch.Tensor) and first.numel() == 1
                ):
                    cond_idx = 1

            if len(args_conditionals) <= cond_idx:
                return original_dynamics(
                    x_t, t, *args_conditionals, **kwargs_conditionals
                )

            cond_tokens = args_conditionals[cond_idx]

            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif (
                isinstance(cond_tokens, torch.Tensor)
                and cond_tokens.shape[0] == num_views
            ):
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                logger.warning(
                    "Condition tokens are not organized per view; using the "
                    "same condition for every view."
                )
                view_conditions = [cond_tokens] * num_views

            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                new_args = (
                    args_conditionals[:cond_idx]
                    + (view_cond,)
                    + args_conditionals[cond_idx + 1 :]
                )
                preds.append(
                    original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                )

            if isinstance(preds[0], dict):
                return {
                    key: torch.stack([p[key] for p in preds]).mean(dim=0)
                    for key in preds[0].keys()
                }
            if isinstance(preds[0], (list, tuple)):
                return tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
            return torch.stack(preds).mean(dim=0)

        generator._generate_dynamics = _dynamics_multidiffusion

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_view_utils.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add sam3d_objects/pipeline/multi_view_utils.py tests/test_multi_view_utils.py
git commit -m "feat: add multi-view multidiffusion fusion context manager (from upstream PR #37)"
```

---

### Task 2: Multi-view methods on `InferencePipeline`

**Files:**
- Modify: `sam3d_objects/pipeline/inference_pipeline.py` (import at line 26; append methods after `_get_dtype`, currently the last method ending at line 864)

**Interfaces:**
- Consumes: `inject_generator_multi_view` from Task 1; existing `InferencePipeline` members: `self.models`, `self.condition_embedders`, `self.ss_condition_input_mapping`, `self.slat_condition_input_mapping`, `self.map_input_keys`, `self.embed_condition`, `self.is_mm_dit()`, `self.merge_image_and_mask`, `self.preprocess_image`, `self.pose_decoder`, `self.decode_slat`, `self.postprocess_slat_output(outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color, rendering_engine)`, `self.downsample_ss_dist`, `self.ss_cfg_strength`, `self.ss_cfg_strength_pm`, `self.slat_cfg_strength`, `self.slat_mean`, `self.slat_std`, `self.shape_model_dtype`, `self.dtype`, module-level `prune_sparse_structure`, `downsample_sparse_structure`, `sp`, `np`, `Image`, `torch`, `logger`.
- Produces: `InferencePipeline.run_multi_view(view_images: List[np.ndarray|PIL.Image], view_masks: Optional[List]=None, seed: Optional[int]=None, stage1_inference_steps=None, stage2_inference_steps=None, use_stage1_distillation=False, use_stage2_distillation=False, decode_formats: Optional[List[str]]=None, with_mesh_postprocess=True, with_texture_baking=True, use_vertex_color=False, stage1_only=False, mode="multidiffusion", rendering_engine="nvdiffrast") -> dict` (result dict has the same keys as `run()`: `glb`, `gs`, `gaussian`, `coords`, ...). Inherited by `InferencePipelinePointMap`, which is what `notebook/inference.py` instantiates. Tasks 5–7 call it via `Inference.multi_view`.

- [ ] **Step 1: Extend the typing import**

In `sam3d_objects/pipeline/inference_pipeline.py` line 26, change:

```python
from typing import List, Union
```

to:

```python
from typing import List, Literal, Optional, Union
```

- [ ] **Step 2: Append the four multi-view methods**

At the end of the file (after the `_get_dtype` static method, which is the last member of class `InferencePipeline`), append — note the 4-space indentation, these are class methods:

```python

    # ------------------------------------------------------------------
    # Multi-view inference (ported from upstream PR #37): training-free
    # multidiffusion fusion — one shared latent, denoiser predictions
    # averaged across all view conditions at every step. See
    # sam3d_objects/pipeline/multi_view_utils.py for the fusion itself.
    # ------------------------------------------------------------------

    def get_multi_view_condition_input(
        self, condition_embedder, view_input_dicts: List[dict], input_mapping
    ):
        """Embed each view's condition and stack the tokens.

        Returns ``((stacked_conditions,), {})`` where ``stacked_conditions``
        has shape (num_views, batch, tokens, dim) when the embedder returns
        tensors, or is a list with one entry per view otherwise.
        """
        view_conditions = []
        for view_input_dict in view_input_dicts:
            condition_args = self.map_input_keys(view_input_dict, input_mapping)
            condition_kwargs = {
                k: v for k, v in view_input_dict.items() if k not in input_mapping
            }
            embedded_cond, _, _ = self.embed_condition(
                condition_embedder, *condition_args, **condition_kwargs
            )
            if embedded_cond is not None:
                view_conditions.append(embedded_cond)
            else:
                view_conditions.append(condition_args)

        if isinstance(view_conditions[0], torch.Tensor):
            all_conditions = torch.stack(view_conditions, dim=0)
        else:
            all_conditions = view_conditions

        return (all_conditions,), {}

    def sample_sparse_structure_multi_view(
        self,
        view_ss_input_dicts: List[dict],
        inference_steps=None,
        use_distillation=False,
        mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
    ):
        """Stage 1 (sparse structure) sampling fused across views."""
        from sam3d_objects.pipeline.multi_view_utils import (
            inject_generator_multi_view,
        )

        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        num_views = len(view_ss_input_dicts)

        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = view_ss_input_dicts[0]["image"]
        bs = image.shape[0]
        logger.info(
            f"Sampling sparse structure with {num_views} views: "
            f"inference_steps={ss_generator.inference_steps}, mode={mode}"
        )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                if self.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                condition_args, condition_kwargs = self.get_multi_view_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    view_ss_input_dicts,
                    self.ss_condition_input_mapping,
                )

                with inject_generator_multi_view(
                    ss_generator,
                    num_views=num_views,
                    num_steps=ss_generator.inference_steps,
                    mode=mode,
                ):
                    return_dict = ss_generator(
                        latent_shape_dict,
                        image.device,
                        *condition_args,
                        **condition_kwargs,
                    )

                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}

                shape_latent = return_dict["shape"]
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent.shape[0], 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

                # downsample output
                return_dict["coords_original"] = coords
                original_shape = coords.shape
                if self.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=self.downsample_ss_dist,
                    )
                coords, downsample_factor = downsample_sparse_structure(coords)
                logger.info(
                    f"Downsampled coords from {original_shape[0]} to {coords.shape[0]}"
                )
                return_dict["coords"] = coords
                return_dict["downsample_factor"] = downsample_factor

        ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def sample_slat_multi_view(
        self,
        view_slat_input_dicts: List[dict],
        coords: torch.Tensor,
        inference_steps=25,
        use_distillation=False,
        mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
    ) -> sp.SparseTensor:
        """Stage 2 (structured latent) sampling fused across views."""
        from sam3d_objects.pipeline.multi_view_utils import (
            inject_generator_multi_view,
        )

        image = view_slat_input_dicts[0]["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        num_views = len(view_slat_input_dicts)
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        logger.info(
            f"Sampling sparse latent with {num_views} views: "
            f"inference_steps={slat_generator.inference_steps}, mode={mode}"
        )

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                condition_args, condition_kwargs = self.get_multi_view_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    view_slat_input_dicts,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)

                with inject_generator_multi_view(
                    slat_generator,
                    num_views=num_views,
                    num_steps=slat_generator.inference_steps,
                    mode=mode,
                ):
                    slat = slat_generator(
                        latent_shape, DEVICE, *condition_args, **condition_kwargs
                    )

                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat[0],
                ).to(DEVICE)
                slat = slat * self.slat_std.to(DEVICE) + self.slat_mean.to(DEVICE)

        slat_generator.inference_steps = prev_inference_steps
        return slat

    def run_multi_view(
        self,
        view_images: List[Union[np.ndarray, Image.Image]],
        view_masks: Optional[List[Union[None, np.ndarray, Image.Image]]] = None,
        seed: Optional[int] = None,
        stage1_inference_steps: Optional[int] = None,
        stage2_inference_steps: Optional[int] = None,
        use_stage1_distillation: bool = False,
        use_stage2_distillation: bool = False,
        decode_formats: Optional[List[str]] = None,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        stage1_only: bool = False,
        mode: Literal["stochastic", "multidiffusion"] = "multidiffusion",
        rendering_engine: str = "nvdiffrast",  # nvdiffrast OR pytorch3d
    ) -> dict:
        """Training-free multi-view reconstruction (multidiffusion fusion).

        Each view is preprocessed independently; a single shared latent is
        then denoised while every diffusion step averages the denoiser
        predictions across all view conditions. Runtime scales roughly
        linearly with the number of views.

        Layout postprocess is not supported here: it aligns the object into
        one view's scene frame, which is ambiguous with several views.

        Args:
            view_images: one image per view (RGB + mask, or RGBA if the
                matching entry in view_masks is None).
            view_masks: one bool/uint8 mask per view, or None per entry if
                the image already carries the mask in its alpha channel.
        """
        num_views = len(view_images)
        if view_masks is None:
            view_masks = [None] * num_views
        assert (
            len(view_masks) == num_views
        ), "Number of masks must match number of images"

        if seed is not None:
            torch.manual_seed(seed)

        logger.info(
            f"Running multi-view inference with {num_views} views, mode={mode}"
        )
        if num_views > 8:
            logger.info(
                f"Note: runtime scales roughly linearly with view count "
                f"({num_views} views)."
            )

        view_ss_input_dicts = []
        view_slat_input_dicts = []
        for i, (image, mask) in enumerate(zip(view_images, view_masks)):
            logger.info(f"Preprocessing view {i + 1}/{num_views}")

            mask_uint8 = None
            if mask is not None:
                mask_uint8 = np.array(mask)
                if mask_uint8.dtype == bool:
                    mask_uint8 = mask_uint8.astype(np.uint8) * 255
                elif mask_uint8.dtype != np.uint8:
                    if mask_uint8.max() <= 1.0:
                        mask_uint8 = (mask_uint8 * 255).astype(np.uint8)
                    else:
                        mask_uint8 = mask_uint8.astype(np.uint8)
            # embeds the mask into the alpha channel; with mask=None the
            # image must already be RGBA
            rgba_image = self.merge_image_and_mask(image, mask_uint8)

            if hasattr(self, "compute_pointmap"):
                # Pointmap pipeline: each view gets an internally computed
                # (MoGe) pointmap for stage-1 preprocessing. External
                # per-view pointmaps are a planned follow-up.
                pointmap_dict = self.compute_pointmap(rgba_image)
                ss_input_dict = self.preprocess_image(
                    rgba_image,
                    self.ss_preprocessor,
                    pointmap=pointmap_dict["pointmap"],
                )
                slat_input_dict = self.preprocess_image(
                    rgba_image, self.slat_preprocessor
                )
            else:
                ss_input_dict = self.preprocess_image(
                    rgba_image, self.ss_preprocessor
                )
                slat_input_dict = self.preprocess_image(
                    rgba_image, self.slat_preprocessor
                )

            view_ss_input_dicts.append(ss_input_dict)
            view_slat_input_dicts.append(slat_input_dict)

        logger.info("Stage 1: sampling sparse structure...")
        ss_return_dict = self.sample_sparse_structure_multi_view(
            view_ss_input_dicts,
            inference_steps=stage1_inference_steps,
            use_distillation=use_stage1_distillation,
            mode=mode,
        )

        ss_return_dict.update(self.pose_decoder(ss_return_dict))

        if "scale" in ss_return_dict:
            logger.info(
                f"Rescaling scale by {ss_return_dict['downsample_factor']}"
            )
            ss_return_dict["scale"] = (
                ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
            )

        if stage1_only:
            logger.info("Finished!")
            ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
            return ss_return_dict

        coords = ss_return_dict["coords"]
        logger.info("Stage 2: sampling structured latent...")
        slat = self.sample_slat_multi_view(
            view_slat_input_dicts,
            coords,
            inference_steps=stage2_inference_steps,
            use_distillation=use_stage2_distillation,
            mode=mode,
        )

        outputs = self.decode_slat(
            slat, self.decode_formats if decode_formats is None else decode_formats
        )
        outputs = self.postprocess_slat_output(
            outputs,
            with_mesh_postprocess,
            with_texture_baking,
            use_vertex_color,
            rendering_engine,
        )
        logger.info("Finished!")

        return {
            **ss_return_dict,
            **outputs,
        }
```

- [ ] **Step 3: Verify syntax and method placement**

Run (Git Bash):

```bash
python -m py_compile sam3d_objects/pipeline/inference_pipeline.py && python - <<'PY'
import ast
src = open("sam3d_objects/pipeline/inference_pipeline.py", encoding="utf-8").read()
tree = ast.parse(src)
cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "InferencePipeline")
names = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
required = {"get_multi_view_condition_input", "sample_sparse_structure_multi_view",
            "sample_slat_multi_view", "run_multi_view"}
missing = required - names
assert not missing, f"missing methods on InferencePipeline: {missing}"
print("OK: all four methods are members of InferencePipeline")
PY
```

Expected: `OK: all four methods are members of InferencePipeline`. If the assert fires, the appended code is indented at module level instead of inside the class — fix indentation.

- [ ] **Step 4: Commit**

```bash
git add sam3d_objects/pipeline/inference_pipeline.py
git commit -m "feat: add run_multi_view + multi-view sampling to InferencePipeline (from upstream PR #37)"
```

---

### Task 3: `fill_holes` guard for pytorch3d in `postprocessing_utils.py`

**Files:**
- Modify: `sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py:629-643` (inside `to_glb`)

**Interfaces:**
- Consumes: existing `to_glb(..., fill_holes: bool = True, ..., rendering_engine: str = "nvdiffrast")` and the module's existing `logger` and `np` imports.
- Produces: no API change — `fill_holes` is silently forced off (with a warning) when `rendering_engine == "pytorch3d"`, because `postprocess_mesh`'s hole filling requires nvdiffrast.

- [ ] **Step 1: Apply the guard**

In `to_glb`, the current code at lines 629-643 reads:

```python
    if with_mesh_postprocess:
        # mesh postprocess
        vertices, faces = postprocess_mesh(
            vertices,
            faces,
            simplify=simplify > 0,
            simplify_ratio=simplify,
            fill_holes=fill_holes,
            fill_holes_max_hole_size=fill_holes_max_size,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            debug=debug,
            verbose=verbose,
        )
```

Replace with:

```python
    if with_mesh_postprocess:
        # mesh postprocess
        # fill_holes requires nvdiffrast; disable it under pytorch3d
        effective_fill_holes = fill_holes and rendering_engine == "nvdiffrast"
        if fill_holes and rendering_engine == "pytorch3d":
            logger.warning(
                "fill_holes is disabled because rendering_engine is "
                "'pytorch3d' (requires nvdiffrast)"
            )
        vertices, faces = postprocess_mesh(
            vertices,
            faces,
            simplify=simplify > 0,
            simplify_ratio=simplify,
            fill_holes=effective_fill_holes,
            fill_holes_max_hole_size=fill_holes_max_size,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
            fill_holes_resolution=1024,
            fill_holes_num_views=1000,
            debug=debug,
            verbose=verbose,
        )
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py`
Expected: exit 0, no output. Also confirm `logger` is imported in that module: `grep -n "logger" sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py | head -3` — if there is no logger import, use `from loguru import logger` added to the module imports.

- [ ] **Step 3: Commit**

```bash
git add sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py
git commit -m "fix: skip fill_holes when rendering_engine is pytorch3d (needs nvdiffrast)"
```

---

### Task 4: Multi-view loader `notebook/load_images_and_masks.py` (TDD)

**Files:**
- Create: `notebook/load_images_and_masks.py`
- Test: `tests/test_load_images_and_masks.py`

**Interfaces:**
- Produces: `load_images_and_masks_from_path(input_path: Path, mask_prompt: Optional[str] = None, image_names: Optional[List[str]] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]` — returns (uint8 RGB/RGBA images, bool (H, W) masks, loaded stem names). Raises `FileNotFoundError` for missing dirs, `ValueError` when zero valid pairs load. **Note the 3-tuple return** — Tasks 6 and 7 rely on the names for logging/output naming. Module imports only pathlib/typing/numpy/PIL/loguru (no torch/GPU deps) so the batch scripts and tests can import it cheaply.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_load_images_and_masks.py`:

```python
"""CPU-only tests for the multi-view image/mask loader."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "notebook"))

from load_images_and_masks import load_images_and_masks_from_path


def _write_image(path, size=(4, 4)):
    arr = np.full(size + (3,), 128, dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _write_rgba_mask(path, size=(4, 4)):
    arr = np.zeros(size + (4,), dtype=np.uint8)
    arr[1:3, 1:3, 3] = 255  # small object in the alpha channel
    Image.fromarray(arr, "RGBA").save(path)


def _write_grayscale_mask(path, size=(4, 4)):
    arr = np.zeros(size, dtype=np.uint8)
    arr[1:3, 1:3] = 255
    Image.fromarray(arr, "L").save(path)


def test_flat_layout(tmp_path):
    for stem in ["1", "2", "view_a"]:
        _write_image(tmp_path / f"{stem}.png")
        _write_rgba_mask(tmp_path / f"{stem}_mask.png")
    images, masks, names = load_images_and_masks_from_path(tmp_path)
    assert names == ["1", "2", "view_a"]
    assert len(images) == len(masks) == 3
    assert masks[0].dtype == bool and masks[0].shape == (4, 4)
    assert masks[0].sum() == 4  # 2x2 object


def test_split_layout_with_mask_prompt(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "toy").mkdir()
    _write_image(tmp_path / "images" / "1.png")
    _write_rgba_mask(tmp_path / "toy" / "1.png")  # plain stem name
    _write_image(tmp_path / "images" / "2.png")
    _write_rgba_mask(tmp_path / "toy" / "2_mask.png")  # _mask suffix name
    images, masks, names = load_images_and_masks_from_path(tmp_path, mask_prompt="toy")
    assert names == ["1", "2"]


def test_image_names_filter(tmp_path):
    for stem in ["1", "2", "3"]:
        _write_image(tmp_path / f"{stem}.png")
        _write_rgba_mask(tmp_path / f"{stem}_mask.png")
    images, masks, names = load_images_and_masks_from_path(
        tmp_path, image_names=["1", "3"]
    )
    assert names == ["1", "3"]


def test_missing_mask_is_skipped_with_warning(tmp_path):
    _write_image(tmp_path / "1.png")
    _write_rgba_mask(tmp_path / "1_mask.png")
    _write_image(tmp_path / "2.png")  # no mask for view 2
    images, masks, names = load_images_and_masks_from_path(tmp_path)
    assert names == ["1"]


def test_no_valid_pairs_raises(tmp_path):
    _write_image(tmp_path / "1.png")  # no masks at all
    with pytest.raises(ValueError):
        load_images_and_masks_from_path(tmp_path)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_images_and_masks_from_path(tmp_path / "nope")


def test_grayscale_mask_fallback(tmp_path):
    _write_image(tmp_path / "1.png")
    _write_grayscale_mask(tmp_path / "1_mask.png")
    _write_image(tmp_path / "2.png")
    _write_rgba_mask(tmp_path / "2_mask.png")
    images, masks, names = load_images_and_masks_from_path(tmp_path)
    assert names == ["1", "2"]
    assert masks[0].sum() == 4  # grayscale mask decoded via nonzero pixels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_load_images_and_masks.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'load_images_and_masks'`.

- [ ] **Step 3: Write the implementation**

Create `notebook/load_images_and_masks.py`:

```python
"""Load multi-view images and masks (ported from upstream PR #37, cleaned up).

Two supported layouts:

1. ``mask_prompt=None`` — images and masks side by side in one directory::

       input_path/1.png, input_path/1_mask.png, input_path/2.png, ...

2. ``mask_prompt="stuffed_toy"`` — images in ``input_path/images/``, masks
   in ``input_path/stuffed_toy/`` named ``<stem>.png`` or ``<stem>_mask.png``.

Masks are RGBA with the mask in the alpha channel (alpha > 0 = object).
Grayscale masks use nonzero pixels; RGB masks fall back to "any non-black
pixel" with a warning (same semantics as ``notebook/inference.py``'s
``load_mask`` — NOT the PR's all-ones fallback, which would select the
whole image).

This module deliberately imports no torch/GPU dependencies so batch
scripts and tests can import it cheaply.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

IMAGE_EXTENSIONS = (".png", ".jpg")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    """Return a (H, W) bool mask, reading the alpha channel when present."""
    img = Image.open(path)
    arr = np.array(img)
    if img.mode == "RGBA" and arr.ndim == 3 and arr.shape[2] >= 4:
        return arr[..., 3] > 0
    if arr.ndim == 2:
        return arr > 0
    logger.warning(
        f"Mask {path} has no alpha channel (mode={img.mode}); "
        "using any non-black pixel as the mask."
    )
    return arr[..., :3].max(axis=-1) > 0


def _find_file(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_pairs(
    images_dir: Path,
    masks_dir: Path,
    image_names: Optional[List[str]],
    same_dir: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    if image_names is None:
        image_files = sorted(
            f
            for ext in IMAGE_EXTENSIONS
            for f in images_dir.glob(f"*{ext}")
            if "_mask" not in f.name
        )
        image_names = [f.stem for f in image_files]
        logger.info(f"Auto-detected {len(image_names)} images: {image_names}")

    images, masks, loaded_names = [], [], []
    for name in image_names:
        image_path = _find_file(
            [images_dir / f"{name}{ext}" for ext in IMAGE_EXTENSIONS]
        )
        if same_dir:
            mask_candidates = [
                masks_dir / f"{name}_mask{ext}" for ext in IMAGE_EXTENSIONS
            ]
        else:
            mask_candidates = [
                masks_dir / f"{name}{ext}" for ext in IMAGE_EXTENSIONS
            ] + [masks_dir / f"{name}_mask{ext}" for ext in IMAGE_EXTENSIONS]
        mask_path = _find_file(mask_candidates)

        if image_path is None:
            logger.warning(f"Image file not found for '{name}', skipping")
            continue
        if mask_path is None:
            logger.warning(f"Mask file not found for '{name}', skipping")
            continue

        images.append(load_image(image_path))
        masks.append(load_mask(mask_path))
        loaded_names.append(name)
        logger.info(
            f"Loaded '{name}': image={images[-1].shape}, mask={masks[-1].shape}"
        )

    return images, masks, loaded_names


def load_images_and_masks_from_path(
    input_path: Path,
    mask_prompt: Optional[str] = None,
    image_names: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Load view images and bool masks; returns (images, masks, names)."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_path}")

    if mask_prompt is None:
        logger.info(f"Loading images and masks from single directory: {input_path}")
        images, masks, names = _load_pairs(
            input_path, input_path, image_names, same_dir=True
        )
    else:
        images_dir = input_path / "images"
        masks_dir = input_path / mask_prompt
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {masks_dir}")
        logger.info(f"Loading images from {images_dir}, masks from {masks_dir}")
        images, masks, names = _load_pairs(
            images_dir, masks_dir, image_names, same_dir=False
        )

    if len(images) == 0:
        raise ValueError(f"No valid image/mask pairs found in {input_path}")
    logger.info(f"Successfully loaded {len(images)} views: {names}")
    return images, masks, names
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_load_images_and_masks.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the full local test suite**

Run: `python -m pytest tests/ -v`
Expected: 15 passed (8 from Task 1 + 7 from this task).

- [ ] **Step 6: Commit**

```bash
git add notebook/load_images_and_masks.py tests/test_load_images_and_masks.py
git commit -m "feat: add multi-view image/mask loader"
```

---

### Task 5: `Inference.multi_view()` public API

**Files:**
- Modify: `notebook/inference.py` (add method to class `Inference`, after `__call__` which ends at line 132)

**Interfaces:**
- Consumes: `InferencePipeline.run_multi_view` (Task 2) via `self._pipeline`; existing `self.merge_mask_to_rgba(image, mask)` (resizes the mask to the image and embeds it in the alpha channel).
- Produces: `Inference.multi_view(images: List[np.ndarray|PIL.Image], masks: List[np.ndarray|PIL.Image], seed: Optional[int] = None, with_mesh_postprocess: bool = True, with_texture_baking: bool = True, use_vertex_color: bool = True, rendering_engine: str = "nvdiffrast", stage1_inference_steps: Optional[int] = None, stage2_inference_steps: Optional[int] = None, decode_formats: Optional[List[str]] = None, mode: str = "multidiffusion") -> dict`. Result dict contains `glb` (trimesh, may be None if mesh not decoded) and `gs` (gaussians with `.save_ply()`). Defaults mirror `Inference.__call__`'s effective batch settings (note `use_vertex_color=True` like `__call__`, unlike the pipeline-level default). Tasks 6 and 7 call this.

- [ ] **Step 1: Add the method**

In `notebook/inference.py`, immediately after the `__call__` method (line 132), add inside class `Inference`:

```python
    def multi_view(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        masks: List[Union[Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = True,
        rendering_engine: str = "nvdiffrast",  # nvdiffrast OR pytorch3d
        stage1_inference_steps: Optional[int] = None,
        stage2_inference_steps: Optional[int] = None,
        decode_formats: Optional[List[str]] = None,
        mode: str = "multidiffusion",
    ) -> dict:
        """Multi-view reconstruction: fuse several photos of one object.

        Training-free multidiffusion (upstream PR #37): every diffusion step
        averages the denoiser predictions across all view conditions.
        Runtime scales roughly linearly with the number of views. Layout
        postprocess is not supported in multi-view mode.
        """
        assert len(images) == len(masks), "one mask per image required"
        assert len(images) >= 2, "multi_view needs at least 2 views"
        rgba_images = [
            self.merge_mask_to_rgba(np.array(image), np.array(mask) > 0)
            for image, mask in zip(images, masks)
        ]
        return self._pipeline.run_multi_view(
            view_images=rgba_images,
            view_masks=None,
            seed=seed,
            stage1_inference_steps=stage1_inference_steps,
            stage2_inference_steps=stage2_inference_steps,
            decode_formats=decode_formats,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            use_vertex_color=use_vertex_color,
            mode=mode,
            rendering_engine=rendering_engine,
        )
```

(`List`, `Optional`, `Union`, `np`, `Image` are already imported at the top of `notebook/inference.py`.)

- [ ] **Step 2: Verify syntax and placement**

Run (Git Bash):

```bash
python -m py_compile notebook/inference.py && python - <<'PY'
import ast
src = open("notebook/inference.py", encoding="utf-8").read()
tree = ast.parse(src)
cls = next(n for n in ast.walk(tree)
           if isinstance(n, ast.ClassDef) and n.name == "Inference")
names = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
assert "multi_view" in names, names
print("OK: Inference.multi_view exists")
PY
```

Expected: `OK: Inference.multi_view exists`.

- [ ] **Step 3: Commit**

```bash
git add notebook/inference.py
git commit -m "feat: add Inference.multi_view public API"
```

---

### Task 6: Standalone CLI `run_inference.py` (TDD for pure helpers)

**Files:**
- Create: `run_inference.py` (repo root)
- Test: `tests/test_run_inference_helpers.py`

**Interfaces:**
- Consumes: `Inference` and `Inference.multi_view` (Task 5), `load_images_and_masks_from_path` (Task 4). Heavy imports happen inside `main()` so the module can be imported for helper tests without GPU deps.
- Produces: CLI `python run_inference.py --input_path <dir> [--mask_prompt NAME] [--image_names a,b] [--seed 42] [--stage1_steps 50] [--stage2_steps 25] [--decode_formats gaussian,mesh] [--model_tag hf] [--rendering_engine nvdiffrast|pytorch3d] [--no_texture_baking] [--no_mesh_postprocess] [--out_dir DIR]`. Writes `result.glb` and `result.ply` under `--out_dir` (default `output/multiview/<derived name>`). Module-level pure helpers used by tests: `parse_image_names(s: Optional[str]) -> Optional[List[str]]`, `parse_decode_formats(s: str) -> List[str]`, `default_out_dir(input_path: Path, mask_prompt: Optional[str], image_names: Optional[List[str]], is_single_view: bool) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_inference_helpers.py`:

```python
"""Tests for run_inference.py pure helpers (no GPU imports at module level)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from run_inference import default_out_dir, parse_decode_formats, parse_image_names


def test_parse_image_names_none_and_empty():
    assert parse_image_names(None) is None
    assert parse_image_names("") is None
    assert parse_image_names(" , ,") is None


def test_parse_image_names_list():
    assert parse_image_names("image1, view_a,2") == ["image1", "view_a", "2"]
    assert parse_image_names("solo") == ["solo"]


def test_parse_decode_formats():
    assert parse_decode_formats("gaussian,mesh") == ["gaussian", "mesh"]
    assert parse_decode_formats(" gaussian ") == ["gaussian"]
    assert parse_decode_formats("") == ["gaussian", "mesh"]  # default fallback


def test_default_out_dir_multiview_all_images():
    out = default_out_dir(Path("data/images_and_masks"), None, None, False)
    assert out == Path("output/multiview/images_and_masks_multiview")


def test_default_out_dir_mask_prompt():
    out = default_out_dir(Path("data"), "stuffed_toy", None, False)
    assert out == Path("output/multiview/stuffed_toy_multiview")


def test_default_out_dir_single_view():
    out = default_out_dir(Path("data"), "toy", ["3"], True)
    assert out == Path("output/multiview/toy_3")


def test_default_out_dir_named_views():
    out = default_out_dir(Path("data"), None, ["1", "2", "3", "4"], False)
    assert out == Path("output/multiview/data_1_2_3_and_1_more")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_inference_helpers.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'run_inference'`.

- [ ] **Step 3: Write the implementation**

Create `run_inference.py` at the repo root:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_inference_helpers.py -v`
Expected: 7 passed. (`from run_inference import ...` must not pull in torch/kaolin — if it fails with an import error of a heavy dep, a heavy import leaked to module level; move it back inside `main()`.)

- [ ] **Step 5: Commit**

```bash
git add run_inference.py tests/test_run_inference_helpers.py
git commit -m "feat: add run_inference.py CLI for single- and multi-view reconstruction"
```

---

### Task 7: Batch multiview mode in `batch_sam3d.py`

**Files:**
- Modify: `batch/batch_sam3d.py` (argparse in `main()` at lines 66-85; add `run_multiview()` function; branch at the top of the run logic)

**Interfaces:**
- Consumes: `Inference.multi_view` (Task 5), `load_images_and_masks_from_path` (Task 4 — `notebook/` is already on `sys.path` via the existing lines 25-28).
- Produces: CLI mode `python batch_sam3d.py --views-dir <dir> --out-dir <dir> [--tag hf] [--seed N] [--skip-existing]` which exports `<out-dir>/splat_multiview.glb`. The existing `--image/--mask` mode is unchanged (they become optional args, validated when `--views-dir` is absent). Task 8's `run_batch.sh` invokes this.

- [ ] **Step 1: Make `--image`/`--mask` optional and add `--views-dir`**

In `batch/batch_sam3d.py` `main()`, replace:

```python
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
```

with:

```python
    ap.add_argument("--image", default=None, help="single-view input image")
    ap.add_argument("--mask", default=None, help="single-view input mask")
    ap.add_argument(
        "--views-dir",
        default=None,
        help="multiview: folder of <stem>.png + <stem>_mask.png pairs; "
        "overrides --image/--mask",
    )
```

And directly after `args = ap.parse_args()`, add:

```python
    if args.views_dir is None and (args.image is None or args.mask is None):
        ap.error("either --views-dir or both --image and --mask are required")
```

- [ ] **Step 2: Add the multiview runner and mask-coverage helper**

Add after the `BACKENDS` list (line 63), before `main()`:

```python
def check_mask_coverage(mask, label):
    """Warn when a mask covers ~0% or ~100% of the frame (misread mask)."""
    coverage = float(mask.mean())
    print(f"[sam3d] mask {label}: {coverage:.1%} of pixels are object")
    if coverage < 0.001 or coverage > 0.999:
        print(
            f"[sam3d] WARNING: mask coverage {coverage:.1%} looks wrong "
            f"(empty or whole-image). Check the mask format/polarity."
        )


def run_multiview(inference, views_dir, out_dir, seed, skip_existing):
    """Fuse all view pairs in views_dir into one splat_multiview.glb."""
    from load_images_and_masks import load_images_and_masks_from_path

    out_glb = os.path.join(out_dir, "splat_multiview.glb")
    if skip_existing and os.path.exists(out_glb):
        print(f"[sam3d] splat_multiview.glb exists -> skip")
        return

    images, masks, names = load_images_and_masks_from_path(views_dir)
    if len(images) < 2:
        print(
            f"[sam3d] ERROR: multiview needs >=2 valid image/mask pairs, "
            f"found {len(images)} ({names}) in {views_dir}"
        )
        sys.exit(1)

    for name, mask in zip(names, masks):
        check_mask_coverage(mask, f"view '{name}'")

    print(f"[sam3d] running multiview fusion over {len(images)} views: {names}")
    output = inference.multi_view(images, masks, seed=seed)
    output["glb"].export(out_glb)
    print(f"[sam3d] exported -> {out_glb}")
```

- [ ] **Step 3: Branch in `main()` and reuse the coverage helper**

In `main()`, right after the model is loaded (`inference = Inference(config_path, compile=False)`, line 91), add:

```python
    if args.views_dir:
        run_multiview(
            inference, args.views_dir, args.out_dir, args.seed, args.skip_existing
        )
        return
```

Then replace the existing inline single-view coverage check (currently lines 96-106):

```python
    # Sanity-check the mask: a mask that covers ~0% or ~100% of the frame means
    # mask.png was misread (wrong channel / inverted / empty) and SAM3D will
    # reconstruct nothing or the whole scene. Surface it instead of silently
    # producing garbage.
    coverage = float(mask.mean())
    print(f"[sam3d] mask {args.mask}: {coverage:.1%} of pixels are object")
    if coverage < 0.001 or coverage > 0.999:
        print(
            f"[sam3d] WARNING: mask coverage {coverage:.1%} looks wrong "
            f"(empty or whole-image). Check mask.png format/polarity."
        )
```

with:

```python
    # A mask covering ~0% or ~100% of the frame means mask.png was misread
    # (wrong channel / inverted / empty) — surface it instead of silently
    # producing garbage.
    check_mask_coverage(mask, args.mask)
```

- [ ] **Step 4: Verify syntax**

Run: `python -m py_compile batch/batch_sam3d.py`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add batch/batch_sam3d.py
git commit -m "feat: add --views-dir multiview mode to batch_sam3d.py"
```

---

### Task 8: Multiview detection in `run_batch.sh` + `config.sh`

**Files:**
- Modify: `batch/run_batch.sh` (add `count_view_pairs` after `find_image`, line 64; add multiview branch after the `IN_DIR` existence check, line 91)
- Modify: `batch/config.sh` (add `SAM3D_MULTIVIEW` near `SAM3D_SEED`, line 45)

**Interfaces:**
- Consumes: `batch_sam3d.py --views-dir` (Task 7); existing config vars `ENV_SAM3D`, `SAM3D_DIR`, `SAM3D_SEED`, `SAM3D_SKIP_EXISTING`, `RUN_SAM3D`; helpers `log`, loop vars `IN_DIR`, `OUT`, `NAME`, counters `OK`/`FAIL`.
- Produces: folders with ≥2 `<stem>_mask.png` + matching `<stem>.<img ext>` pairs are processed as multiview (depth backends skipped, output `splat_multiview.glb`); everything else follows the existing single-view path. `SAM3D_MULTIVIEW=0` disables detection.

- [ ] **Step 1: Add the config knob**

In `batch/config.sh`, after the `SAM3D_SKIP_EXISTING` line (line 46), add:

```bash
: "${SAM3D_MULTIVIEW:=1}"  # 1 = folders with >=2 <stem>+<stem>_mask.png pairs run multiview fusion
```

- [ ] **Step 2: Add the pair counter to `run_batch.sh`**

After the `find_image()` function (ends line 64), add:

```bash
# Count multiview pairs: <stem>_mask.png files with a matching <stem>.<img ext>.
count_view_pairs() {
    local dir="$1" count=0 m stem ext
    for m in "${dir}"/*_mask.png; do
        [ -f "$m" ] || continue
        stem="$(basename "$m" _mask.png)"
        for ext in png jpg jpeg PNG JPG JPEG; do
            if [ -f "${dir}/${stem}.${ext}" ]; then count=$((count+1)); break; fi
        done
    done
    echo "$count"
}
```

- [ ] **Step 3: Add the multiview branch to the sample loop**

In the sample loop, directly after the input-dir existence check (line 91: `if [ ! -d "${IN_DIR}" ]; ...`) and BEFORE the `IMG="$(find_image ...)"` line, add:

```bash
    # ---- multiview: >=2 view pairs -> fuse all views, skip depth backends --
    if [ "${SAM3D_MULTIVIEW}" = "1" ]; then
        PAIRS="$(count_view_pairs "${IN_DIR}")"
        if [ "${PAIRS}" -ge 2 ]; then
            log "  multiview folder (${PAIRS} view pairs) -> skipping depth backends"
            mkdir -p "${OUT}"
            if [ "${RUN_SAM3D}" = "1" ]; then
                (
                    set -e
                    conda activate "${ENV_SAM3D}"
                    cd "${SAM3D_DIR}"
                    SKIP_FLAG=""
                    [ "${SAM3D_SKIP_EXISTING}" = "1" ] && SKIP_FLAG="--skip-existing"
                    python "${SCRIPT_DIR}/batch_sam3d.py" \
                        --views-dir "${IN_DIR}" --out-dir "${OUT}" \
                        --seed "${SAM3D_SEED}" ${SKIP_FLAG}
                ) && { OK=$((OK+1)); log "  DONE ${NAME} (multiview)"; } \
                  || { FAIL=$((FAIL+1)); log "  [sam3d] FAILED ${NAME} (multiview)"; }
            else
                OK=$((OK+1))
            fi
            continue
        fi
    fi
```

- [ ] **Step 4: Syntax-check both scripts**

Run: `bash -n batch/run_batch.sh && bash -n batch/config.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`.

- [ ] **Step 5: Functionally test the pair counter locally**

Run in Git Bash (copies the function into an isolated script with fixtures — run_batch.sh itself can't be sourced without executing):

```bash
T=$(mktemp -d)
mkdir -p "$T/mv" "$T/sv" "$T/half"
touch "$T/mv/1.png" "$T/mv/1_mask.png" "$T/mv/2.jpg" "$T/mv/2_mask.png"
touch "$T/sv/image.jpg" "$T/sv/mask.png"
touch "$T/half/1.png" "$T/half/1_mask.png" "$T/half/2_mask.png"  # 2_mask has no image
sed -n '/^count_view_pairs()/,/^}/p' batch/run_batch.sh > "$T/fn.sh"
source "$T/fn.sh"
[ "$(count_view_pairs "$T/mv")" = "2" ] && \
[ "$(count_view_pairs "$T/sv")" = "0" ] && \
[ "$(count_view_pairs "$T/half")" = "1" ] && echo PAIRS_OK || echo PAIRS_FAIL
rm -rf "$T"
```

Expected: `PAIRS_OK`. (Note: legacy `mask.png` does not match `*_mask.png`, so single-view folders count 0 — that's the test's `sv` case.)

- [ ] **Step 6: Commit**

```bash
git add batch/run_batch.sh batch/config.sh
git commit -m "feat: detect multiview folders in batch pipeline, skip depth backends"
```

---

### Task 9: Document multiview in `batch/README.md`

**Files:**
- Modify: `batch/README.md` (append a section)

**Interfaces:**
- Consumes: behavior established in Tasks 6-8.
- Produces: user documentation only.

- [ ] **Step 1: Append the docs section**

Read `batch/README.md` first to match its existing tone/format, then append:

```markdown
## Multiview (multiple photos of one object)

Put **2 or more** view pairs in a sample folder instead of `image.jpg` + `mask.png`:

```
input/images/<object>/
    1.png          # photo from viewpoint 1 (any stem works)
    1_mask.png     # its mask (RGBA, mask in the alpha channel; alpha>0 = object)
    2.png
    2_mask.png
    ...
```

`run_batch.sh` detects these folders automatically, **skips the depth
backends** (each view gets an internal MoGe pointmap), and produces a single
fused reconstruction:

```
output/images/<object>/splat_multiview.glb
```

Fusion is training-free multidiffusion (upstream PR #37): every diffusion
step averages the model's predictions across all views, so occluded parts in
one photo are filled in by the others. Runtime scales roughly linearly with
the view count.

Knobs (see `config.sh`): `SAM3D_MULTIVIEW=0` disables detection;
`SAM3D_SEED` and `SAM3D_SKIP_EXISTING` work as in single-view mode.

One-off runs without the batch harness:

```bash
conda activate sam3d-objects
python run_inference.py --input_path input/images/<object>
```
```

- [ ] **Step 2: Commit**

```bash
git add batch/README.md
git commit -m "docs: document batch multiview mode"
```

---

### Task 10: Remote verification (GPU) — run on the Linux box

**Files:** none created (verification only; fixture data downloaded on the remote box).

**Interfaces:**
- Consumes: everything above, synced/pulled to `/workspace/sam-3d-objects` on the remote box.

This task cannot run on the local Windows machine. Execute each step on the remote box (env `sam3d-objects`) and record the outcome. If the person executing this plan has no remote access, present this task to the user as a ready-to-run checklist and stop.

- [ ] **Step 1: Sync the branch to the remote box**

On the remote box:

```bash
cd /workspace/sam-3d-objects && git fetch && git checkout multiview && git pull
```

- [ ] **Step 2: Run the CPU test suite in the real env**

```bash
conda activate sam3d-objects
pip install pytest  # if missing
python -m pytest tests/ -v
```

Expected: 22 passed (8 + 7 + 7).

- [ ] **Step 3: Fetch the PR's known-good example data**

The PR head is `devinli123:pr-for-official` (fork repo `devinli123/multi-view-sam-3d-objects`; verify on the PR page if this 404s):

```bash
cd /workspace/sam-3d-objects
mkdir -p input/images/stuffed_toy_mv
BASE=https://raw.githubusercontent.com/devinli123/multi-view-sam-3d-objects/pr-for-official/data/example
for i in 1 2 3 4 5 6 7 8; do
    curl -fsSL "$BASE/images/$i.png"            -o "input/images/stuffed_toy_mv/$i.png"
    curl -fsSL "$BASE/stuffed_toy/${i}_mask.png" -o "input/images/stuffed_toy_mv/${i}_mask.png"
done
ls -la input/images/stuffed_toy_mv   # expect 16 files
```

- [ ] **Step 4: Standalone CLI, multiview**

```bash
python run_inference.py --input_path input/images/stuffed_toy_mv
```

Expected: logs show "Multi-view inference with 8 views", Stage 1 + Stage 2 complete, and `output/multiview/stuffed_toy_mv_multiview/result.glb` + `result.ply` exist. Open the GLB (Blender / online viewer): the dog should show the red collar (visible in only some views) and a coherent front — the PR's own success criterion vs single-view.

- [ ] **Step 5: Full-settings check (texture baking)**

Step 4 already ran with nvdiffrast + texture baking + mesh postprocess (the defaults). If it crashed or the texture looks broken, rerun with the author's validated settings and record the difference:

```bash
python run_inference.py --input_path input/images/stuffed_toy_mv \
    --no_texture_baking --out_dir output/multiview/stuffed_toy_vertexcolor
```

If only the fallback works, file the texture-baking incompatibility as a known issue in `batch/README.md` and keep `--no_texture_baking` documented as the workaround.

- [ ] **Step 6: Batch pipeline, multiview folder**

```bash
cd /workspace/sam-3d-objects/batch
./run_batch.sh stuffed_toy_mv
```

Expected: log line "multiview folder (8 view pairs) -> skipping depth backends"; no lotus2/da3/depthpro/moge2 stages run; `output/images/stuffed_toy_mv/splat_multiview.glb` exists.

- [ ] **Step 7: Single-view regression**

Pick any existing single-view sample folder (`image.jpg` + `mask.png`), e.g. one already under `input/images/`:

```bash
./run_batch.sh <existing_single_view_sample>
```

Expected: behaves exactly as before this branch — depth backends run, `splat_*.glb` outputs produced, no multiview log lines.

- [ ] **Step 8: Commit any fixes and report**

Commit fixes made during verification. Report per step: pass/fail + the artifact paths, and (subjectively) whether the multiview GLB beats a single-view GLB of the same object.

---

## Self-review notes (already applied)

- **Spec coverage:** core fusion (T1), pipeline methods incl. `rendering_engine` pass-through + no-layout-postprocess (T2), fill_holes guard (T3), loader with both layouts (T4), public API (T5), CLI with underscore flags + `output/multiview/` (T6), batch `--views-dir` + `splat_multiview.glb` + <2-pairs error + per-view coverage warnings (T7), detection + skip-backends + `SAM3D_MULTIVIEW` knob (T8), docs (T9), full verification plan incl. PR fixture + texture-baking experiment + regression (T10). N>8 runtime note lives in `run_multi_view` (T2).
- **Deviation from PR, on purpose:** loader's RGB-mask fallback uses "non-black pixel" (fork's `load_mask` semantics) instead of the PR's all-ones mask; loader returns a 3-tuple (adds names); `load_from_segmentation_structure` dropped (YAGNI). Fusion math untouched.
- **Type consistency check:** `load_images_and_masks_from_path` 3-tuple used consistently in T6/T7; `Inference.multi_view(images, masks, ...)` matches T5 signature at both call sites; `--views-dir` spelling consistent in T7/T8.
