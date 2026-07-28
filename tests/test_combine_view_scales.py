"""CPU-only tests for combining per-view metric size estimates.

Stage 1 fuses one shared latent across all views, so decoding that single
prediction against each view's pointmap gives one estimate per view of the same
object's size. These tests pin down how those are reduced to one number.

The module is loaded straight from its file so the heavy sam3d_objects package
(spconv/CUDA) is not imported; the helper under test is pure Python.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_multi_view_utils():
    """Import multi_view_utils, standing in for torch/loguru when absent.

    The helper under test is pure Python, but its module imports torch and
    loguru at the top. Any stub is removed again in the finally block: leaving
    a fake torch in sys.modules would turn other test modules' clean ImportError
    into confusing failures, and the loaded module keeps working because it
    never touches either name.
    """
    stubs = {"torch": {}, "loguru": {"logger": types.SimpleNamespace()}}
    installed = []
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(stub, key, value)
            sys.modules[name] = stub
            installed.append(name)

    try:
        spec = importlib.util.spec_from_file_location(
            "multi_view_utils_for_scales",
            _ROOT / "sam3d_objects" / "pipeline" / "multi_view_utils.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in installed:
            del sys.modules[name]


_mvu = _load_multi_view_utils()
combine_view_scales = _mvu.combine_view_scales
SCALE_DISAGREEMENT_RATIO = _mvu.SCALE_DISAGREEMENT_RATIO


def test_single_view_returns_its_own_estimate_and_no_ratio():
    median, ratio = combine_view_scales([0.42])
    assert median == pytest.approx(0.42)
    assert ratio is None


def test_odd_count_takes_the_middle_value():
    median, _ = combine_view_scales([0.30, 0.10, 0.20])
    assert median == pytest.approx(0.20)


def test_even_count_averages_the_two_middle_values():
    median, _ = combine_view_scales([0.10, 0.20, 0.30, 0.40])
    assert median == pytest.approx(0.25)


def test_median_ignores_one_wildly_wrong_view():
    """The whole point of median over mean: one bad depth must not move it."""
    median, _ = combine_view_scales([0.20, 0.21, 0.19, 50.0])
    assert median == pytest.approx(0.205)
    mean = (0.20 + 0.21 + 0.19 + 50.0) / 4
    assert abs(median - 0.20) < abs(mean - 0.20)


def test_ratio_reports_the_spread():
    _, ratio = combine_view_scales([0.20, 0.30])
    assert ratio == pytest.approx(1.5)


def test_agreeing_views_stay_under_the_warning_threshold():
    _, ratio = combine_view_scales([0.20, 0.22, 0.21])
    assert ratio < SCALE_DISAGREEMENT_RATIO


def test_disagreeing_views_exceed_the_warning_threshold():
    _, ratio = combine_view_scales([0.20, 0.90])
    assert ratio > SCALE_DISAGREEMENT_RATIO


def test_unusable_estimates_are_dropped_not_counted():
    median, ratio = combine_view_scales(
        [0.20, 0.0, -1.0, None, float("nan"), float("inf"), 0.22]
    )
    assert median == pytest.approx(0.21)
    assert ratio == pytest.approx(0.22 / 0.20)


def test_no_usable_estimate_returns_nothing():
    assert combine_view_scales([]) == (None, None)
    assert combine_view_scales([0.0, -3.0, float("nan")]) == (None, None)
