# MoGe-2 as the default depth model

**Date:** 2026-07-22
**Status:** Approved

## Goal

Make MoGe-2 (`Ruicheng/moge-2-vitl-normal`) the depth model the production SAM3D
inference server uses, replacing MoGe v1 (`Ruicheng/moge-vitl`). MoGe-2 runs
in-process in the `sam3d-objects` env, exactly where v1 runs today — not as a
separate pre-computation stage.

Ambiguous settings resolve toward final 3D output quality. Where MoGe-2's own
defaults already represent the best-quality choice, they are used unchanged.

## Background

Production flows `gpu-server-scripts/beestoon.py` → `POST /infer` →
`process/3d-generator/server.py`, which loads `checkpoints/<tag>/pipeline.yaml`
(tag defaults to `hf`). That config wires the depth model:

```yaml
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge.MoGe
  model:
    _target_: moge.model.v1.MoGeModel.from_pretrained
    pretrained_model_name_or_path: Ruicheng/moge-vitl
```

Two facts shape the whole design:

1. **The pinned MoGe commit has no v2.** `a8c37341` ships only `moge/model/v1.py`,
   so the dependency pin must move.
2. **`checkpoints/hf/pipeline.yaml` is a downloaded artifact**, pulled from
   `facebook/sam-3d-objects` and overwritten on every setup run
   (`setup-gpu-server.sh:327-330`). Editing the repo's `checkpoints/pipeline.yaml`
   does not affect production.

A MoGe-2 integration already exists at `batch/run_moge2.py`, but only as a
batch backend in a separate `moge2` conda env that writes a `pointmap.pt` for
SAM3D to consume. That path is unaffected by this change.

### The utils3d question

`batch/config.sh:17-18` records that MoGe-2 needs a newer utils3d than SAM3D
pins, because SAM3D uses the old `.numpy`/`.torch` API and MoGe-2 needs the new
`.pt` API. That is why the batch backend got its own env.

This is only half true. The utils3d commit MoGe-2 pins (`3fab839f`) still
exposes both naming schemes:

```python
lazy_import(globals(), '.numpy', 'numpy')
lazy_import(globals(), '.numpy', 'np')
lazy_import(globals(), '.torch', 'torch')
lazy_import(globals(), '.torch', 'pt')
```

The namespaces survive, so a single shared env is viable. The residual risk is
function-level drift, not namespace naming — SAM3D calls roughly fifteen utils3d
functions (`RastContext`, `rasterize_triangle_faces`, `compute_dual_graph`,
`compute_connected_components`, `depth_edge`, and others) across its rendering
and post-processing code. This risk is accepted and covered by rollback rather
than by a compatibility shim.

## Design

### 1. Dependencies

In `requirements.txt`, move the MoGe pin to a commit containing `moge/model/v2.py`,
and add an **explicit** utils3d pin:

```
MoGe @ git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030
utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
```

`925b8ed` is `main` as of 2026-07-21. It is pinned rather than tracked, matching
the existing convention of pinning MoGe to an exact commit. The utils3d SHA is
the one MoGe's own `requirements.txt` specifies at that commit.

Note: MoGe-3 is referenced in the upstream README as of this commit, but no
`moge/model/v3.py` module exists in the repository. MoGe-2 is the newest model
actually importable here, so it is the correct target.

utils3d is pinned explicitly rather than left transitive. MoGe declares utils3d
as a bare git URL, and pip does not install a git dependency's own
`requirements.txt` unless it appears in `install_requires`. Leaving it implicit
risks a working local machine and a broken server.

### 2. Depth model wrapper

New file `sam3d_objects/pipeline/depth_models/moge2.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
from .base import DepthModel


class MoGe2(DepthModel):
    def __call__(self, image):
        # MoGe-2 defaults are the best-quality settings: resolution_level=9
        # already maps to max_tokens, and apply_mask=True marks invalid pixels
        # inf, which is SAM3D's own convention.
        output = self.model.infer(image.to(self.device))
        output["pointmaps"] = output["points"]
        return output
```

`moge.py` (v1) is left untouched; it is what rollback selects.

Three properties make this wrapper as thin as the v1 one:

- **No mask patching.** With `apply_mask=True` (default), MoGe-2 writes `torch.inf`
  at invalid pixels. That is already SAM3D's convention — `pointmap.py:98` performs
  the identical `torch.where(mask_binary[..., None], points, torch.inf)`, and every
  consumer filters with `torch.isfinite`. The `pointmap[~mask] = np.nan` line in
  `batch/run_moge2.py:73` is belt-and-braces, not a requirement.
- **No kwargs.** `force_projection=True`, `apply_mask=True`, `use_fp16=True` and
  `resolution_level=9` are all v2 defaults, and `resolution_level=9` is already the
  maximum-detail setting (it maps to the top of the `[1200, 3600]` token range).
  Passing nothing matches `batch/run_moge2.py:61`.
- **`normal` rides along unused.** The `-normal` variant returns a `normal` key that
  SAM3D does not consume. Harmless, and available if later work wants it.

### 3. Config patch

New script `scripts/set_depth_model.py`, rewriting three fields in a given tag's
`pipeline.yaml`:

| Field | `moge2` (default) | `moge1` (rollback) |
|---|---|---|
| `depth_model._target_` | `sam3d_objects.pipeline.depth_models.moge2.MoGe2` | `sam3d_objects.pipeline.depth_models.moge.MoGe` |
| `depth_model.model._target_` | `moge.model.v2.MoGeModel.from_pretrained` | `moge.model.v1.MoGeModel.from_pretrained` |
| `depth_model.model.pretrained_model_name_or_path` | `Ruicheng/moge-2-vitl-normal` | `Ruicheng/moge-vitl` |

Usage: `python scripts/set_depth_model.py --tag hf --variant moge2`

The script uses OmegaConf, which is explicitly listed in `requirements.txt` and is
already how `Inference.__init__` reads this very file. PyYAML is only a transitive
dependency of OmegaConf, so depending on it directly would be relying on something
the project never declares. OmegaConf is also preferred over `yq`, which is not
installed on the GPU box, and over `sed`, which is brittle against a
vendor-controlled file. `pipeline.yaml` contains no `${...}` interpolations, so a
load/save round-trip is lossless in every way that matters to the consumer.

The script is idempotent and writes explicit values for both variants, so it can
flip the config in either direction without re-downloading the checkpoint.

`setup-gpu-server.sh` calls it immediately after the
`mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}` line, honouring a
`SAM3D_DEPTH_MODEL` variable that defaults to `moge2`.

Nothing loads `checkpoints/pipeline.yaml` directly — both `server.py` and
`run_inference.py` read the tag subdirectory. (Design-time note: an earlier draft
proposed committing an updated reference copy of this file. That was dropped during
implementation — `checkpoints/.gitignore` is `*` / `!.gitignore`, so the whole
directory's contents are deliberately untracked; the file ships from the
`facebook/sam-3d-objects` HuggingFace repo and is downloaded, then overridden by the
script above. The tracked source of truth for the v2 default is
`scripts/set_depth_model.py`'s `DEPTH_MODELS["moge2"]`.)

### 4. Weight pre-fetch

`setup-gpu-server.sh` adds `Ruicheng/moge-2-vitl-normal` to its `hf download`
block. Without this, the first `/infer` after deploy triggers a multi-hundred-MB
download inside a live request, or fails outright on a network-restricted box.

### 5. Hydra safety

No change required. `check_hydra_safety` whitelists targets by top-level module
and `moge` is already permitted (`notebook/inference.py:40`), so
`moge.model.v2.MoGeModel.from_pretrained` passes unchanged.

## Risks

**utils3d function drift.** The newer utils3d may have changed signatures of
functions SAM3D's rendering and post-processing rely on. Mitigation: verify by
import/attribute audit before the smoke test, and roll back if inference breaks.

**Metric vs scale-invariant scale.** MoGe v1 emits scale-invariant point maps;
MoGe-2 emits metric metres. Occlusion detection uses absolute thresholds —
`is_occluded_by_others(..., z_thresh=0.05)` and `z_thresh=0.3` at
`layout_post_optimization_utils.py:50,99`. These thresholds therefore change
meaning. They are deliberately left untouched: the working range for this project
is 0.2–1.5 m (`batch/config.sh:41-42`), so metric magnitudes land near v1's
normalised ~1.0. If output quality regresses, re-tuning these thresholds is the
first thing to try.

**fp16 nesting.** The pipeline already calls the depth model inside
`torch.autocast(device_type="cuda", dtype=self.dtype)`
(`inference_pipeline_pointmap.py:290`), and MoGe-2's `infer` defaults to
`use_fp16=True`. Nested half-precision paths are a common source of dtype
mismatch. Watch for it during the smoke test.

## Verification

1. **Import/attribute audit** over the utils3d functions SAM3D uses, plus
   `from moge.model.v2 import MoGeModel`.
2. **End-to-end smoke test**: one `/infer` request producing a valid GLB, checked
   for finite geometry and sane object scale.

No v1-versus-v2 comparison run. The rollback path exists as a safety net, not as
an evaluation workflow.

## Rollback

```bash
python scripts/set_depth_model.py --tag hf --variant moge1
# restart the SAM3D server
```

The v1 wrapper and its dependency path remain intact, so this needs no
re-download and no dependency change.

## Out of scope

- `batch/` keeps its separate `moge2` env and `run_moge2.py` pointmap backend.
- `notebook/mesh_alignment.py:17` continues to use `moge.model.v1`.
