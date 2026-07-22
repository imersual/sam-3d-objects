# MoGe-2 Default Depth Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MoGe-2 (`Ruicheng/moge-2-vitl-normal`) the depth model the production SAM3D inference server loads, replacing MoGe v1.

**Architecture:** A new `MoGe2` wrapper class sits beside the existing `MoGe` one and calls MoGe-2's `infer()` with no keyword arguments, because every v2 default is already the best-quality setting. Because production reads `checkpoints/hf/pipeline.yaml` — a file downloaded from HuggingFace and overwritten on every setup run — the config change is applied by an idempotent script invoked from `setup-gpu-server.sh` rather than committed to a yaml.

**Tech Stack:** Python 3.11, PyTorch 2.8 / CUDA 12.8, Hydra + OmegaConf, MoGe, utils3d, pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-moge2-default-design.md`

## Global Constraints

- Two separate git repositories are touched. Tasks 1, 2, 3 and 5 commit in `sam-3d-objects`; Task 4 commits in `gpu-server-scripts`. Never mix them in one commit.
- MoGe pin is exactly `925b8ed835a7a9cdb7578ba15c658a0afc969030`.
- utils3d pin is exactly `3fab839f0be9931dac7c8488eb0e1600c236e183`.
- MoGe-2 HF model id is exactly `Ruicheng/moge-2-vitl-normal`.
- MoGe v1 code (`sam3d_objects/pipeline/depth_models/moge.py`) must remain intact and working — it is the rollback path.
- Tests must not import CUDA or load real model weights at module level. Follow the existing convention in `tests/`: guard heavy imports with `pytest.importorskip`.
- `sam3d_objects/pipeline/depth_models/` has no `__init__.py` and does not need one; it resolves as a PEP 420 namespace package. Do not add one.
- No MoGe v1 vs v2 comparison runs. Rollback exists as a safety net, not an evaluation workflow.

## Execution Amendment (2026-07-22)

The development machine is Windows with Python 3.11 and **pytest only** — `torch`,
`omegaconf`, `moge` and `utils3d` are all absent, and installing them here is out
of scope. This changes how the TDD steps execute:

- **Tests are written but not run during implementation.** Every "Run test to
  verify it fails / passes" step is deferred. An implementer must report its
  tests as `UNRUN (deps unavailable locally)` and must **never** fabricate or
  guess pytest output. A task is complete when the code and tests are written and
  committed, not when tests are green.
- **All GPU-box commands live in `setup-gpu-server.sh`.** The operator runs that
  one script rather than copying commands by hand. Task 3's install-and-verify
  step and the test-suite run therefore move into Task 4's script edit, and
  Task 6 reduces to running the script plus an optional smoke test.

---

### Task 1: MoGe2 depth model wrapper

**Files:**
- Create: `sam3d_objects/pipeline/depth_models/moge2.py`
- Test: `tests/test_moge2_depth_model.py`

**Interfaces:**
- Consumes: `DepthModel` from `sam3d_objects/pipeline/depth_models/base.py`. Its `__init__(self, model, device="cuda")` stores `self.model` / `self.device`, then calls `model.to(device)` and `model.eval()`.
- Produces: `MoGe2`, a `DepthModel` subclass whose `__call__(image)` returns the dict from `MoGeModel.infer()` with an added `"pointmaps"` key. `InferencePipelinePointMap` reads `output["pointmaps"]` at `sam3d_objects/pipeline/inference_pipeline_pointmap.py:292`.

Three behaviours are locked by tests because each encodes a deliberate design decision that is easy to "helpfully" break later: the `pointmaps` alias, the absence of kwargs, and leaving `inf` alone.

- [ ] **Step 1: Write the failing test**

Create `tests/test_moge2_depth_model.py`:

```python
"""Tests for the MoGe-2 depth model wrapper (no GPU, no real weights)."""
import pytest

torch = pytest.importorskip("torch")

from sam3d_objects.pipeline.depth_models.moge2 import MoGe2  # noqa: E402


class _FakeMoGe2Model:
    """Stands in for moge.model.v2.MoGeModel."""

    def __init__(self):
        self.infer_calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer(self, image, **kwargs):
        self.infer_calls.append(kwargs)
        points = torch.zeros((2, 2, 3))
        points[0, 0] = torch.inf  # MoGe-2 marks invalid pixels inf
        return {
            "points": points,
            "depth": torch.ones((2, 2)),
            "mask": torch.ones((2, 2), dtype=torch.bool),
            "intrinsics": torch.eye(3),
            "normal": torch.zeros((2, 2, 3)),
        }


def _wrapper():
    return MoGe2(_FakeMoGe2Model(), device="cpu")


def test_pointmaps_aliases_points():
    output = _wrapper()(torch.zeros((3, 2, 2)))
    assert output["pointmaps"] is output["points"]


def test_infer_called_with_v2_defaults():
    """No kwargs: MoGe-2's own defaults are the intended settings."""
    model = _FakeMoGe2Model()
    MoGe2(model, device="cpu")(torch.zeros((3, 2, 2)))
    assert model.infer_calls == [{}]


def test_invalid_pixels_stay_inf():
    """SAM3D filters with torch.isfinite; the wrapper must not rewrite inf."""
    output = _wrapper()(torch.zeros((3, 2, 2)))
    assert torch.isinf(output["pointmaps"][0, 0]).all()


def test_extra_v2_keys_pass_through():
    output = _wrapper()(torch.zeros((3, 2, 2)))
    assert "normal" in output and "mask" in output
```

- [ ] **Step 2: Do not run the test locally**

`torch` is absent on the dev machine, so `pytest.importorskip("torch")` would skip
the file and prove nothing. Report `UNRUN (deps unavailable locally)`. These tests
run on the GPU box via `setup-gpu-server.sh`.

- [ ] **Step 3: Write minimal implementation**

Create `sam3d_objects/pipeline/depth_models/moge2.py`:

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
from .base import DepthModel


class MoGe2(DepthModel):
    """MoGe-2 depth model.

    Unlike the v1 wrapper, no inference kwargs are passed: MoGe-2's defaults are
    already the best-quality settings (``resolution_level=9`` maps to the top of
    the token range, and ``apply_mask=True`` marks invalid pixels ``inf``, which
    is SAM3D's own convention -- see ``pipeline/utils/pointmap.py``). This
    matches how ``batch/run_moge2.py`` calls the model.
    """

    def __call__(self, image):
        output = self.model.infer(image.to(self.device))
        output["pointmaps"] = output["points"]
        return output
```

- [ ] **Step 4: Confirm the code is written, not that tests pass**

Do not run pytest and do not claim a result. Re-read `moge2.py` against the four
tests and confirm by inspection that each would pass. Report `UNRUN (deps
unavailable locally)`.

- [ ] **Step 5: Commit**

```bash
git add sam3d_objects/pipeline/depth_models/moge2.py tests/test_moge2_depth_model.py
git commit -m "feat: add MoGe-2 depth model wrapper"
```

---

### Task 2: Config rewrite script

**Files:**
- Create: `scripts/set_depth_model.py`
- Test: `tests/test_set_depth_model.py`

**Interfaces:**
- Produces: `DEPTH_MODELS`, a dict keyed `"moge2"` / `"moge1"`; `set_depth_model(config, variant)` which mutates and returns an OmegaConf config, raising `ValueError` on an unknown variant; `main(argv=None)` implementing the CLI.
- Consumed by: Task 4, which calls the CLI from `setup-gpu-server.sh`.

The `scripts/` directory does not exist yet and will be created by writing this file.

- [ ] **Step 1: Write the failing test**

Create `tests/test_set_depth_model.py`:

```python
"""Tests for scripts/set_depth_model.py (no GPU imports)."""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from set_depth_model import main, set_depth_model  # noqa: E402


def _v1_config():
    """A config shaped like the depth_model block of the downloaded pipeline.yaml."""
    return OmegaConf.create(
        {
            "pose_decoder_name": "ScaleShiftInvariant",
            "depth_model": {
                "_target_": "sam3d_objects.pipeline.depth_models.moge.MoGe",
                "model": {
                    "_target_": "moge.model.v1.MoGeModel.from_pretrained",
                    "pretrained_model_name_or_path": "Ruicheng/moge-vitl",
                },
            },
        }
    )


def test_moge2_rewrites_all_three_fields():
    config = set_depth_model(_v1_config(), "moge2")
    assert config.depth_model._target_ == (
        "sam3d_objects.pipeline.depth_models.moge2.MoGe2"
    )
    assert config.depth_model.model._target_ == (
        "moge.model.v2.MoGeModel.from_pretrained"
    )
    assert config.depth_model.model.pretrained_model_name_or_path == (
        "Ruicheng/moge-2-vitl-normal"
    )


def test_unrelated_keys_untouched():
    config = set_depth_model(_v1_config(), "moge2")
    assert config.pose_decoder_name == "ScaleShiftInvariant"


def test_idempotent():
    once = set_depth_model(_v1_config(), "moge2")
    twice = set_depth_model(set_depth_model(_v1_config(), "moge2"), "moge2")
    assert once == twice


def test_rollback_restores_v1():
    config = set_depth_model(set_depth_model(_v1_config(), "moge2"), "moge1")
    assert config.depth_model._target_ == (
        "sam3d_objects.pipeline.depth_models.moge.MoGe"
    )
    assert config.depth_model.model.pretrained_model_name_or_path == (
        "Ruicheng/moge-vitl"
    )


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        set_depth_model(_v1_config(), "moge3")


def test_main_round_trips_the_file(tmp_path):
    (tmp_path / "hf").mkdir()
    path = tmp_path / "hf" / "pipeline.yaml"
    OmegaConf.save(_v1_config(), path)

    main(["--tag", "hf", "--variant", "moge2", "--checkpoints-root", str(tmp_path)])

    reloaded = OmegaConf.load(path)
    assert reloaded.depth_model.model.pretrained_model_name_or_path == (
        "Ruicheng/moge-2-vitl-normal"
    )
    assert reloaded.pose_decoder_name == "ScaleShiftInvariant"


def test_main_exits_when_file_missing(tmp_path):
    with pytest.raises(SystemExit):
        main(["--tag", "nope", "--checkpoints-root", str(tmp_path)])
```

- [ ] **Step 2: Do not run the test locally**

`omegaconf` is absent on the dev machine, so this file would fail at collection
rather than fail meaningfully. Report `UNRUN (deps unavailable locally)`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/set_depth_model.py`:

```python
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
```

- [ ] **Step 4: Confirm the code is written, not that tests pass**

Do not run pytest and do not claim a result. Re-read `set_depth_model.py` against
the seven tests and confirm by inspection that each would pass — in particular
that `main()` accepts an `argv` list, and that a missing file raises `SystemExit`.
Report `UNRUN (deps unavailable locally)`.

- [ ] **Step 5: Commit**

```bash
git add scripts/set_depth_model.py tests/test_set_depth_model.py
git commit -m "feat: add set_depth_model.py to pin the checkpoint depth model"
```

---

### Task 3: Dependency pins and third-party API guard

**Files:**
- Modify: `requirements.txt:86`
- Test: `tests/test_dependency_api_surface.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: no importable symbols. Guarantees `moge.model.v2` and the utils3d / MoGe-internal functions SAM3D calls are importable after the pin bump.

SAM3D imports MoGe internals directly (`sam3d_objects/pipeline/utils/pointmap.py:11-18`), so a MoGe bump can break SAM3D even where no MoGe model is involved. This test makes that failure loud and immediate instead of surfacing deep inside a render call. All four internal helpers were verified present at `925b8ed`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dependency_api_surface.py`:

```python
"""Guards the third-party API surface SAM3D depends on.

Bumping the MoGe pin to get v2 also bumps utils3d. SAM3D calls utils3d and MoGe
internals directly, so a bad upgrade should fail here rather than inside a
rendering or focal-recovery call.
"""
import pytest

utils3d = pytest.importorskip("utils3d")

UTILS3D_TORCH_FUNCTIONS = [
    "extrinsics_look_at",
    "intrinsics_from_fov_xy",
    "perspective_from_fov_xy",
    "view_look_at",
    "RastContext",
    "rasterize_triangle_faces",
    "compute_edges",
    "compute_connected_components",
    "compute_dual_graph",
    "compute_edge_connected_components",
    "remove_unreferenced_vertices",
    "extrinsics_to_view",
    "intrinsics_to_perspective",
    "intrinsics_from_focal_center",
]


@pytest.mark.parametrize("name", UTILS3D_TORCH_FUNCTIONS)
def test_utils3d_torch_surface(name):
    assert hasattr(utils3d.torch, name), f"utils3d.torch.{name} is missing"


def test_utils3d_numpy_depth_edge():
    assert hasattr(utils3d.numpy, "depth_edge")


def test_utils3d_io_write_ply():
    assert hasattr(utils3d.io, "write_ply")


def test_moge_v2_importable():
    pytest.importorskip("moge")
    from moge.model.v2 import MoGeModel

    assert hasattr(MoGeModel, "from_pretrained")


def test_moge_v1_still_importable():
    """v1 is the rollback path and must keep working."""
    pytest.importorskip("moge")
    from moge.model.v1 import MoGeModel

    assert hasattr(MoGeModel, "from_pretrained")


def test_moge_internal_helpers_importable():
    """sam3d_objects/pipeline/utils/pointmap.py imports these directly."""
    pytest.importorskip("moge")
    from moge.utils.geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift
    from moge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift

    assert all(
        callable(function)
        for function in (
            normalized_view_plane_uv,
            recover_focal_shift,
            solve_optimal_focal_shift,
            solve_optimal_shift,
        )
    )
```

- [ ] **Step 2: Do not run the test locally**

`utils3d` and `moge` are absent on the dev machine, so every test here would skip
and prove nothing. Report `UNRUN (deps unavailable locally)`. The real gate is
`setup-gpu-server.sh`, which runs this file after `pip install` (Task 4).

- [ ] **Step 3: Update the dependency pins**

In `requirements.txt`, replace line 86:

```
MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b
```

with these two lines:

```
MoGe @ git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030
utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
```

utils3d is pinned explicitly, not left transitive: MoGe declares it as a bare git URL, and pip does not install a git dependency's own `requirements.txt` unless it appears in `install_requires`.

- [ ] **Step 4: Verification is wired into setup, not run here**

The install-and-verify commands belong in `setup-gpu-server.sh` (Task 4), so the
operator runs one script. Nothing to execute in this task.

For reference, the gate Task 4 installs enforces **0 skipped** (the exact pass count
is not asserted, since the guard grew to 24 cases once `depth_to_points` and the
`utils3d.numpy` surface were added). Any skip means MoGe or utils3d did not install.
If a `utils3d.torch.*` assertion fails, that is the drift risk the spec flags — it
changes the approach rather than being a small fix, so report it rather than patching
around it.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/test_dependency_api_surface.py
git commit -m "feat: pin MoGe 925b8ed for v2 support and pin utils3d explicitly"
```

---

### Task 4: Wire the config patch and weight prefetch into server setup

**Files:**
- Modify: `gpu-server-scripts/setup-gpu-server.sh:325-332`

**Interfaces:**
- Consumes: `scripts/set_depth_model.py` CLI from Task 2 — `--tag`, `--variant`, `--checkpoints-root`.
- Produces: no importable symbols.

This task commits in the **`gpu-server-scripts` repo**, not `sam-3d-objects`.

Context that makes the placement correct: the `sam3d-objects` mamba env is activated at line 286 and is still active here, so `python` resolves to the env that has OmegaConf. The working directory is `$SAM3D_REPO_DIR` from line 272, so the relative `scripts/` path resolves. `hf download` with no `--local-dir` populates the default HuggingFace cache, which is the same cache `start_server.sh` reads, since neither script sets `HF_HOME`.

- [ ] **Step 1: Apply the edit**

Replace lines 325-332 of `gpu-server-scripts/setup-gpu-server.sh`:

```bash
echo "[+] Downloading SAM-3D model"
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download facebook/sam-3d-objects

mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download

cd "$WORKDIR"
```

with:

```bash
echo "[+] Downloading SAM-3D model"
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download facebook/sam-3d-objects

mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download

# The downloaded pipeline.yaml defaults to MoGe v1. Re-apply our depth model
# choice; this file is overwritten on every re-download, so it cannot be
# committed. Set SAM3D_DEPTH_MODEL=moge1 to roll back.
echo "[+] Setting depth model to ${SAM3D_DEPTH_MODEL:-moge2}"
python scripts/set_depth_model.py --tag "${TAG}" --variant "${SAM3D_DEPTH_MODEL:-moge2}"

# Pre-fetch MoGe-2 weights into the HF cache so the first /infer request after
# deploy does not trigger a multi-hundred-MB download inside a live request.
echo "[+] Pre-fetching MoGe-2 weights"
hf download Ruicheng/moge-2-vitl-normal

# Verify the MoGe/utils3d pin bump did not break the API surface SAM3D calls.
# These tests cannot run on the Windows dev machine (no torch/utils3d), so this
# is their first real execution. A skip here means a dependency did not install.
echo "[+] Verifying dependency API surface and depth model wrapper"
python -m pytest tests/test_dependency_api_surface.py tests/test_moge2_depth_model.py \
    tests/test_set_depth_model.py -v

cd "$WORKDIR"
```

The `pytest` line is deliberately not guarded with `|| true`: if the pin bump broke
`utils3d.torch.rasterize_triangle_faces` or a MoGe internal helper, setup must fail
loudly rather than leave a server that dies at first inference.

- [ ] **Step 2: Verify the script still parses**

Run: `bash -n gpu-server-scripts/setup-gpu-server.sh`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit (in the gpu-server-scripts repo)**

```bash
cd /d/Projects/imersual/beestoon/gpu-server-scripts
git add setup-gpu-server.sh
git commit -m "feat: set MoGe-2 as SAM3D depth model and prefetch its weights"
```

---

### Task 5: Update the repo's reference pipeline.yaml — DROPPED during execution

> **DROPPED (2026-07-23).** This task assumed `checkpoints/pipeline.yaml` was a
> tracked reference file. It is not: `checkpoints/.gitignore` is `*` / `!.gitignore`,
> deliberately excluding all of `checkpoints/` from git because `pipeline.yaml` ships
> from the HuggingFace repo `facebook/sam-3d-objects` and is downloaded, not committed.
> Force-committing it would fight that convention, and the file is read by nothing
> (production loads `checkpoints/hf/pipeline.yaml`). The v2-default statement is already
> carried by tracked code — `scripts/set_depth_model.py`'s `DEPTH_MODELS["moge2"]` — and
> applied to the downloaded file at setup time. The steps below are left for the record
> but were not committed.

**Files:**
- Modify: `checkpoints/pipeline.yaml:61-65`

**Interfaces:**
- Consumes: the target strings from Task 2's `DEPTH_MODELS["moge2"]`. They must match exactly.
- Produces: no importable symbols.

Nothing loads this file directly — `server.py` and `run_inference.py` both read `checkpoints/<tag>/pipeline.yaml` — but leaving it advertising v1 would contradict the running system for anyone reading the repo.

- [ ] **Step 1: Apply the edit**

Replace lines 61-65 of `checkpoints/pipeline.yaml`:

```yaml
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge.MoGe
  model:
    _target_: moge.model.v1.MoGeModel.from_pretrained
    pretrained_model_name_or_path: Ruicheng/moge-vitl
```

with:

```yaml
depth_model:
  _target_: sam3d_objects.pipeline.depth_models.moge2.MoGe2
  model:
    _target_: moge.model.v2.MoGeModel.from_pretrained
    pretrained_model_name_or_path: Ruicheng/moge-2-vitl-normal
```

- [ ] **Step 2: Verify by inspection that it matches what the script writes**

`omegaconf` is unavailable locally, so compare the two by eye. Open
`scripts/set_depth_model.py` and confirm all three strings under
`DEPTH_MODELS["moge2"]` are character-identical to what you just wrote into
`checkpoints/pipeline.yaml`:

- `sam3d_objects.pipeline.depth_models.moge2.MoGe2`
- `moge.model.v2.MoGeModel.from_pretrained`
- `Ruicheng/moge-2-vitl-normal`

A mismatch here is silent: the reference yaml and the deployed config would
disagree with nobody noticing. Check each string, do not skim.

- [ ] **Step 3: Commit**

```bash
git add checkpoints/pipeline.yaml
git commit -m "docs: point reference pipeline.yaml at MoGe-2"
```

---

### Task 6: Deploy and smoke-verify on the GPU box

**Files:** none modified.

**Interfaces:**
- Consumes: everything from Tasks 1-5.

Per the spec there is no v1-versus-v2 comparison. This task confirms MoGe-2 loads and produces valid geometry.

This task is run by the operator on the box, not by an implementer subagent.

- [ ] **Step 1: Pull both repos and run setup**

```bash
cd /workspace/sam-3d-objects && git pull
cd /workspace/gpu-server-scripts && git pull
./setup-gpu-server.sh
```

`setup-gpu-server.sh` now installs the new pins, rewrites the depth model in
`checkpoints/hf/pipeline.yaml`, pre-fetches the MoGe-2 weights, and runs the test
suite. It fails loudly if any of those steps fail.

- [ ] **Step 2: Confirm the config landed**

```bash
grep -A4 '^depth_model:' /workspace/sam-3d-objects/checkpoints/hf/pipeline.yaml
```

Expected output contains `moge.model.v2.MoGeModel.from_pretrained` and `Ruicheng/moge-2-vitl-normal`.

- [ ] **Step 4: Restart the server and confirm the model loads**

```bash
pkill -f 'process/3d-generator/server.py' || true
nohup /workspace/sam-3d-objects/process/3d-generator/start_server.sh 8000 1 > /var/log/sam3d-server.log 2>&1 &
sleep 60
curl -s http://localhost:8000/health
grep -i 'model loaded\|error\|traceback' /var/log/sam3d-server.log | tail -20
```

Expected: `{"status":"ok"}` and `Model loaded and ready.` with no traceback. A dtype error here is the fp16-inside-autocast risk from the spec — report it rather than patching around it.

- [ ] **Step 5: Run one end-to-end inference**

Use an existing image/mask pair already on the box. Submit it directly to `/infer`:

```bash
curl -s -X POST http://localhost:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "image_path": "/workspace/inputs/sam3d/smoke/image.jpg",
    "mask_paths": ["/workspace/inputs/sam3d/smoke/mask.png"],
    "output_path": "/workspace/outputs/smoke/moge2.glb",
    "seed": 1
  }'
```

Substitute real paths for `image_path` / `mask_paths`. Expected: a success response and `moge2.glb` written. Then check the geometry is finite and sanely scaled:

```bash
python -c "
import sys, trimesh
scene = trimesh.load(sys.argv[1])
geometry = list(scene.geometry.values())[0] if hasattr(scene, 'geometry') else scene
bounds = geometry.bounds
print('bounds:', bounds)
print('extent (m):', bounds[1] - bounds[0])
assert geometry.vertices.size and (geometry.vertices == geometry.vertices).all(), 'NaN or empty vertices'
print('OK')
" /path/to/produced.glb
```

Expected: `OK`, with extents in a plausible object range rather than collapsed to ~0 or exploded to a huge value. MoGe-2 is metric, so extents now read directly in metres.

- [ ] **Step 6: Record the result**

No commit. Report: whether the model loaded, whether the GLB is valid, and the observed extents. If quality regressed, the spec names re-tuning the occlusion thresholds at `layout_post_optimization_utils.py:50,99` as the first thing to try, since those are absolute and v2 changed the units.

**Rollback if needed:**

```bash
python scripts/set_depth_model.py --tag hf --variant moge1
pkill -f 'process/3d-generator/server.py'
nohup /workspace/sam-3d-objects/process/3d-generator/start_server.sh 8000 1 > /var/log/sam3d-server.log 2>&1 &
```
