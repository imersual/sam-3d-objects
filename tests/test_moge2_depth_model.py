"""Tests for the MoGe-2 depth model wrapper (no GPU, no real weights)."""
import os

# Importing anything under the `sam3d_objects` package runs its __init__.py, which
# pulls in the heavy `sam3d_objects.init` module unless LIDRA_SKIP_INIT is set. That
# module is not needed to import the depth-model class, and the production entrypoint
# (notebook/inference.py) sets this same flag for the same reason. Must be set before
# the first sam3d_objects import below.
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

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
