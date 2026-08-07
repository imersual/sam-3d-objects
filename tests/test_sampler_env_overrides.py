"""Tests for sam3d_objects/pipeline/sampler_env_overrides.py.

The module under test is dependency-free (stdlib only), so it is imported
directly from its file path -- exactly like test_combine_view_scales.py does
for multi_view_utils.py -- so this test never triggers
sam3d_objects/__init__.py or sam3d_objects/pipeline/__init__.py, which do
import torch/hydra and are unavailable on a non-GPU dev machine.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    name = "sampler_env_overrides_under_test"
    spec = importlib.util.spec_from_file_location(
        name,
        _ROOT / "sam3d_objects" / "pipeline" / "sampler_env_overrides.py",
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines frozen dataclasses, and
    # dataclasses resolves annotations via sys.modules[cls.__module__] --
    # without this it raises AttributeError on a None module.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


_seo = _load_module()
resolve_env_overrides = _seo.resolve_env_overrides
resolved_settings_for_log = _seo.resolved_settings_for_log
SAMPLER_KNOBS = _seo.SAMPLER_KNOBS


# ── contract: exact env var names the task asked for ────────────────────────


def test_env_var_names_match_the_documented_contract():
    """Locks in the exact SAM3D_* names/config keys so a rename is a
    deliberate, visible change rather than an accident."""
    by_config_key = {knob.config_key: knob.env_var for knob in SAMPLER_KNOBS}
    assert by_config_key == {
        "ss_inference_steps": "SAM3D_SS_INFERENCE_STEPS",
        "ss_cfg_strength": "SAM3D_SS_CFG_STRENGTH",
        "ss_cfg_strength_pm": "SAM3D_SS_CFG_STRENGTH_PM",
        "slat_inference_steps": "SAM3D_SLAT_INFERENCE_STEPS",
        "slat_cfg_strength": "SAM3D_SLAT_CFG_STRENGTH",
        "downsample_ss_dist": "SAM3D_DOWNSAMPLE_SS_DIST",
    }


def test_code_defaults_match_inference_pipeline_signature():
    """These mirror InferencePipeline.__init__'s own signature defaults
    (sam3d_objects/pipeline/inference_pipeline.py) -- pinned here so the two
    can't silently drift apart."""
    by_config_key = {knob.config_key: knob.code_default for knob in SAMPLER_KNOBS}
    assert by_config_key == {
        "ss_inference_steps": 25,
        "ss_cfg_strength": 7,
        "ss_cfg_strength_pm": 0.0,
        "slat_inference_steps": 25,
        "slat_cfg_strength": 5,
        "downsample_ss_dist": 0,
    }


# ── no env vars set: the default, unattended-server case ────────────────────


def test_no_env_vars_produces_no_overrides_and_no_warnings():
    result = resolve_env_overrides({})
    assert result.overrides == {}
    assert result.warnings == []


def test_unrelated_env_vars_are_ignored():
    result = resolve_env_overrides({"PATH": "/usr/bin", "HOME": "/root"})
    assert result.overrides == {}
    assert result.warnings == []


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_value_is_treated_as_unset(blank):
    """A blank env var (e.g. `SAM3D_SS_CFG_STRENGTH=` in a shell script)
    must behave exactly like the var not being set at all, not like '0'."""
    result = resolve_env_overrides({"SAM3D_SS_CFG_STRENGTH": blank})
    assert result.overrides == {}
    assert result.warnings == []


# ── one valid override per knob ──────────────────────────────────────────────


def test_valid_ss_inference_steps():
    result = resolve_env_overrides({"SAM3D_SS_INFERENCE_STEPS": "40"})
    assert result.overrides == {"ss_inference_steps": 40}
    assert result.warnings == []


def test_valid_ss_cfg_strength():
    result = resolve_env_overrides({"SAM3D_SS_CFG_STRENGTH": "9.5"})
    assert result.overrides == {"ss_cfg_strength": 9.5}
    assert result.warnings == []


def test_valid_ss_cfg_strength_pm():
    result = resolve_env_overrides({"SAM3D_SS_CFG_STRENGTH_PM": "0.3"})
    assert result.overrides == {"ss_cfg_strength_pm": 0.3}
    assert result.warnings == []


def test_valid_slat_inference_steps():
    result = resolve_env_overrides({"SAM3D_SLAT_INFERENCE_STEPS": "50"})
    assert result.overrides == {"slat_inference_steps": 50}
    assert result.warnings == []


def test_valid_slat_cfg_strength():
    result = resolve_env_overrides({"SAM3D_SLAT_CFG_STRENGTH": "3.25"})
    assert result.overrides == {"slat_cfg_strength": 3.25}
    assert result.warnings == []


def test_valid_downsample_ss_dist():
    result = resolve_env_overrides({"SAM3D_DOWNSAMPLE_SS_DIST": "2"})
    assert result.overrides == {"downsample_ss_dist": 2}
    assert result.warnings == []


def test_all_knobs_set_at_once():
    env = {
        "SAM3D_SS_INFERENCE_STEPS": "30",
        "SAM3D_SS_CFG_STRENGTH": "8",
        "SAM3D_SS_CFG_STRENGTH_PM": "0.1",
        "SAM3D_SLAT_INFERENCE_STEPS": "35",
        "SAM3D_SLAT_CFG_STRENGTH": "6",
        "SAM3D_DOWNSAMPLE_SS_DIST": "1",
    }
    result = resolve_env_overrides(env)
    assert result.overrides == {
        "ss_inference_steps": 30,
        "ss_cfg_strength": 8.0,
        "ss_cfg_strength_pm": 0.1,
        "slat_inference_steps": 35,
        "slat_cfg_strength": 6.0,
        "downsample_ss_dist": 1,
    }
    assert result.warnings == []


# ── numeric edge cases that must not silently corrupt a run ─────────────────


def test_cfg_strength_accepts_zero_and_negative_values():
    """Guidance strength 0 (no guidance) or negative (anti-guidance) are
    legitimate experimental settings the team may want to try -- not errors."""
    result = resolve_env_overrides(
        {"SAM3D_SS_CFG_STRENGTH": "0", "SAM3D_SLAT_CFG_STRENGTH": "-2.5"}
    )
    assert result.overrides == {"ss_cfg_strength": 0.0, "slat_cfg_strength": -2.5}
    assert result.warnings == []


def test_downsample_ss_dist_accepts_zero():
    result = resolve_env_overrides({"SAM3D_DOWNSAMPLE_SS_DIST": "0"})
    assert result.overrides == {"downsample_ss_dist": 0}
    assert result.warnings == []


@pytest.mark.parametrize("bad_steps", ["0", "-5"])
def test_non_positive_inference_steps_rejected(bad_steps):
    result = resolve_env_overrides({"SAM3D_SS_INFERENCE_STEPS": bad_steps})
    assert result.overrides == {}
    assert len(result.warnings) == 1
    assert "SAM3D_SS_INFERENCE_STEPS" in result.warnings[0]


def test_fractional_inference_steps_rejected():
    """A step count must be an int; '7.5' should fall back, not truncate."""
    result = resolve_env_overrides({"SAM3D_SLAT_INFERENCE_STEPS": "7.5"})
    assert result.overrides == {}
    assert len(result.warnings) == 1


@pytest.mark.parametrize("bad_dist", ["-1", "1.5"])
def test_downsample_ss_dist_rejects_negative_and_fractional(bad_dist):
    """downsample_ss_dist feeds a tensor kernel dimension
    (2 * max_neighbor_axes_dist + 1) in prune_sparse_structure(); a float
    would parse but crash later inside torch.ones(), and a negative value
    produces a nonsensical (or negative) kernel size. Both must be caught
    here, at parse time, not at a GPU crash three steps later."""
    result = resolve_env_overrides({"SAM3D_DOWNSAMPLE_SS_DIST": bad_dist})
    assert result.overrides == {}
    assert len(result.warnings) == 1


@pytest.mark.parametrize(
    "env_var", ["SAM3D_SS_CFG_STRENGTH", "SAM3D_SLAT_CFG_STRENGTH", "SAM3D_SS_CFG_STRENGTH_PM"]
)
@pytest.mark.parametrize("non_finite", ["nan", "inf", "-inf"])
def test_non_finite_cfg_strength_rejected(env_var, non_finite):
    """float('nan')/float('inf') parse successfully in Python but must not
    be allowed through: they would propagate into every downstream tensor."""
    result = resolve_env_overrides({env_var: non_finite})
    assert result.overrides == {}
    assert len(result.warnings) == 1


@pytest.mark.parametrize(
    "env_var",
    [
        "SAM3D_SS_INFERENCE_STEPS",
        "SAM3D_SS_CFG_STRENGTH",
        "SAM3D_SS_CFG_STRENGTH_PM",
        "SAM3D_SLAT_INFERENCE_STEPS",
        "SAM3D_SLAT_CFG_STRENGTH",
        "SAM3D_DOWNSAMPLE_SS_DIST",
    ],
)
def test_garbage_string_is_ignored_not_raised(env_var):
    """The core promise: a stray typo on a GPU box must never crash the
    server. Every knob, given garbage, falls back cleanly with a warning."""
    result = resolve_env_overrides({env_var: "banana"})
    assert result.overrides == {}
    assert len(result.warnings) == 1
    assert env_var in result.warnings[0]
    assert "banana" in result.warnings[0]


def test_one_bad_knob_does_not_block_others():
    result = resolve_env_overrides(
        {"SAM3D_SS_INFERENCE_STEPS": "not-a-number", "SAM3D_SLAT_INFERENCE_STEPS": "42"}
    )
    assert result.overrides == {"slat_inference_steps": 42}
    assert len(result.warnings) == 1
    assert "SAM3D_SS_INFERENCE_STEPS" in result.warnings[0]


def test_warning_mentions_the_default_fallback():
    result = resolve_env_overrides({"SAM3D_SS_CFG_STRENGTH": "not-a-number"})
    assert "default" in result.warnings[0].lower()
    assert "ss_cfg_strength" in result.warnings[0]


# ── resolved_settings_for_log ────────────────────────────────────────────────


def test_resolved_settings_for_log_uses_code_defaults_when_config_is_empty():
    assert resolved_settings_for_log({}) == {
        "ss_inference_steps": 25,
        "ss_cfg_strength": 7,
        "ss_cfg_strength_pm": 0.0,
        "slat_inference_steps": 25,
        "slat_cfg_strength": 5,
        "downsample_ss_dist": 0,
    }


def test_resolved_settings_for_log_reflects_yaml_values_not_in_env():
    """Mirrors checkpoints/pipeline.yaml, which sets slat_cfg_strength=1 and
    downsample_ss_dist=1 (overriding the code defaults of 5 and 0) while
    leaving the rest to InferencePipeline's own defaults."""
    config = {"slat_cfg_strength": 1, "downsample_ss_dist": 1}
    resolved = resolved_settings_for_log(config)
    assert resolved["slat_cfg_strength"] == 1
    assert resolved["downsample_ss_dist"] == 1
    assert resolved["ss_inference_steps"] == 25  # untouched -> code default


def test_resolved_settings_for_log_reflects_env_overrides():
    config = {}
    result = resolve_env_overrides({"SAM3D_SS_INFERENCE_STEPS": "99"})
    config.update(result.overrides)
    resolved = resolved_settings_for_log(config)
    assert resolved["ss_inference_steps"] == 99
    assert resolved["slat_inference_steps"] == 25  # untouched -> code default
