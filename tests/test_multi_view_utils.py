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
    # NB: == not `is` — accessing a method creates a fresh bound-method
    # object each time; equality checks the underlying function + instance.
    gen = FakeGenerator()
    original = gen._generate_dynamics
    with inject_generator_multi_view(gen, num_views=2, num_steps=4):
        assert gen._generate_dynamics != original
    assert gen._generate_dynamics == original


def test_restores_original_dynamics_on_exception():
    gen = FakeGenerator()
    original = gen._generate_dynamics
    with pytest.raises(RuntimeError):
        with inject_generator_multi_view(gen, num_views=2, num_steps=4):
            raise RuntimeError("boom")
    assert gen._generate_dynamics == original


def test_unsupported_mode_raises():
    gen = FakeGenerator()
    with pytest.raises(ValueError):
        with inject_generator_multi_view(gen, num_views=2, num_steps=4, mode="bogus"):
            pass
