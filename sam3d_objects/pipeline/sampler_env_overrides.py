# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Environment-variable overrides for SAM3D's reconstruction-quality knobs.

``InferencePipeline.__init__`` (sam3d_objects/pipeline/inference_pipeline.py)
exposes several numeric knobs that control sampling quality: how many
denoising steps run and how strongly classifier-free guidance is applied,
for the sparse-structure (ss) and structured-latent (slat) stages, plus how
aggressively the sparse-structure voxel grid gets pruned before decoding
(``downsample_ss_dist``). Ordinarily these are fixed at whatever
``checkpoints/<tag>/pipeline.yaml`` says (falling back to the Python-level
defaults in ``InferencePipeline.__init__`` for anything the yaml omits).

This module lets each knob be overridden per-process via an environment
variable, so quality/guidance settings can be experimented with on a GPU box
without editing code or the checkpoint config -- e.g. to chase the "plate
reconstructs as a flat disc" bug, which is sensitive to sparse-structure
sampling quality.

It is intentionally dependency-free (stdlib only) so it can be unit-tested
without importing torch/hydra/loguru, none of which are needed to parse and
validate a handful of numbers.

Reads are defensive by design: this runs unattended on a GPU box, so a
malformed value must never raise or crash the server. A bad value is
reported back as a warning string (the caller decides how to log it) and
the corresponding knob is left out of the returned overrides entirely --
whatever value pipeline.yaml / InferencePipeline's own defaults already
resolve to is untouched.
"""
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Union

Number = Union[int, float]


def _parse_int(raw: str) -> int:
    # `int(raw)` already rejects "7.5" (ValueError), which is what we want:
    # inference_steps is a step count, not a strength.
    return int(raw.strip())


def _parse_float(raw: str) -> float:
    return float(raw.strip())


def _positive_int(value: Number) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: Number) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_number(value: Number) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class SamplerKnob:
    """One overridable knob.

    ``config_key`` is the exact ``InferencePipeline.__init__`` / pipeline.yaml
    kwarg name. ``code_default`` mirrors that constructor's own signature
    default, purely for logging -- it is what gets reported as "resolved
    value" when neither the env var nor pipeline.yaml sets the knob; it is
    never used to override anything.
    """

    config_key: str
    env_var: str
    parse: Callable[[str], Number]
    validate: Callable[[Number], bool]
    code_default: Number
    description: str


# Order matches the task description and InferencePipeline.__init__'s own
# parameter order.
SAMPLER_KNOBS: tuple[SamplerKnob, ...] = (
    SamplerKnob(
        "ss_inference_steps",
        "SAM3D_SS_INFERENCE_STEPS",
        _parse_int,
        _positive_int,
        25,
        "sparse-structure denoising steps",
    ),
    SamplerKnob(
        "ss_cfg_strength",
        "SAM3D_SS_CFG_STRENGTH",
        _parse_float,
        _finite_number,
        7,
        "sparse-structure classifier-free-guidance strength",
    ),
    SamplerKnob(
        "ss_cfg_strength_pm",
        "SAM3D_SS_CFG_STRENGTH_PM",
        _parse_float,
        _finite_number,
        0.0,
        "sparse-structure pointmap classifier-free-guidance strength",
    ),
    SamplerKnob(
        "slat_inference_steps",
        "SAM3D_SLAT_INFERENCE_STEPS",
        _parse_int,
        _positive_int,
        25,
        "structured-latent denoising steps",
    ),
    SamplerKnob(
        "slat_cfg_strength",
        "SAM3D_SLAT_CFG_STRENGTH",
        _parse_float,
        _finite_number,
        5,
        "structured-latent classifier-free-guidance strength",
    ),
    SamplerKnob(
        "downsample_ss_dist",
        "SAM3D_DOWNSAMPLE_SS_DIST",
        _parse_int,
        _non_negative_int,
        0,
        # Despite the name this is a discrete voxel-neighbor radius, not a
        # continuous distance: prune_sparse_structure() (inference_utils.py)
        # feeds it straight into a tensor kernel dimension
        # (2 * max_neighbor_axes_dist + 1), which requires an int -- a float
        # like 1.5 would pass a naive numeric check but crash later inside
        # torch.ones(). Parsed/validated as int for exactly that reason.
        "sparse-structure downsample distance (voxel-neighbor radius; 0 disables pruning)",
    ),
)


@dataclass(frozen=True)
class EnvOverrideResult:
    overrides: dict  # config_key -> parsed value; only knobs that were set and valid
    warnings: list  # one human-readable message per malformed env var


def resolve_env_overrides(
    env: Mapping[str, str],
    knobs: tuple[SamplerKnob, ...] = SAMPLER_KNOBS,
) -> EnvOverrideResult:
    """Read and validate each knob's environment variable.

    An unset or blank env var is the normal case (use whatever pipeline.yaml
    / the code default already says) and produces neither an override nor a
    warning. A set-but-malformed value (not parseable as a number, or out of
    range for that knob -- e.g. a non-positive step count) produces a
    warning and is otherwise ignored, exactly as if it had not been set.

    Returns an ``EnvOverrideResult`` with ``overrides`` keyed by the
    ``InferencePipeline`` kwarg name, ready to assign into a config dict
    before instantiation.
    """
    overrides = {}
    warnings = []
    for knob in knobs:
        raw = env.get(knob.env_var)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = knob.parse(raw)
            if not knob.validate(value):
                raise ValueError(f"parsed value {value!r} is out of range")
        except (TypeError, ValueError) as exc:
            warnings.append(
                f"Ignoring {knob.env_var}={raw!r} ({exc}); "
                f"falling back to the existing default for {knob.config_key} "
                f"({knob.description})."
            )
            continue
        overrides[knob.config_key] = value
    return EnvOverrideResult(overrides=overrides, warnings=warnings)


def resolved_settings_for_log(
    config: Mapping[str, Number],
    knobs: tuple[SamplerKnob, ...] = SAMPLER_KNOBS,
) -> dict:
    """Build the {knob_name: effective_value} mapping for a startup log line.

    ``config`` should be the pipeline config *after* ``resolve_env_overrides``'s
    overrides have been applied, so this reflects what the pipeline will
    actually be constructed with -- env override, else whatever pipeline.yaml
    set, else the code default (only for logging; never assigned back).
    """
    return {
        knob.config_key: config.get(knob.config_key, knob.code_default)
        for knob in knobs
    }
