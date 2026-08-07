# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for InferencePipelinePointMap.compute_pointmap's additive normal/
intrinsics passthrough (see sam3d_objects/pipeline/inference_pipeline_pointmap.py).

inference_pipeline_pointmap.py imports pytorch3d at module level (Transform3d,
look_at_view_transform), so this whole file needs both torch AND pytorch3d --
neither is installed on this non-GPU dev box, so it skips here (one skip, at
the first importorskip below, same as tests/test_moge2_depth_model.py already
does for torch alone). It is written to run wherever both ARE available (the
GPU server: setup-gpu-server.sh installs pytorch3d from source) and was
reviewed carefully for shape/API correctness against the unmodified
surrounding code, but could not be executed in this session -- see this
task's report for the full list of what that means is unverified.
"""
import os

# Same reasoning as tests/test_moge2_depth_model.py: avoid pulling in
# sam3d_objects.init via the package __init__.
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch3d")
np = pytest.importorskip("numpy")

from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # noqa: E402
    InferencePipelinePointMap,
)


class _FakeDepthModel:
    """Stands in for the MoGe2 wrapper (sam3d_objects/pipeline/depth_models/
    moge2.py): returns a fixed output dict, optionally overridden per test."""

    def __init__(self, extra=None):
        self._extra = extra or {}

    def __call__(self, image):
        # image is (3, H, W), as compute_pointmap prepares it.
        h, w = image.shape[-2], image.shape[-1]
        points = torch.zeros((h, w, 3))
        points[..., 2] = 1.0  # z=1 for every pixel: avoids a degenerate/zero pointmap
        out = {
            "pointmaps": points,
            "intrinsics": torch.eye(3),
        }
        out.update(self._extra)
        return out


def _fake_pipeline(depth_model):
    """A bare InferencePipelinePointMap instance with __init__ skipped.

    __init__ loads real hydra-instantiated models, which needs live
    checkpoints and is out of reach in a unit test. compute_pointmap only
    touches self.device / self.dtype / self.depth_model /
    self.clip_pointmap_beyond_scale plus inherited pure methods
    (image_to_float, _clip_pointmap), so those are set directly on a
    bare instance instead.
    """
    pipeline = object.__new__(InferencePipelinePointMap)
    pipeline.device = torch.device("cpu")
    pipeline.dtype = torch.float32
    pipeline.depth_model = depth_model
    pipeline.clip_pointmap_beyond_scale = None  # skip the mask-based clip entirely
    return pipeline


def _rgba_image(h=2, w=2):
    image = np.zeros((h, w, 4), dtype=np.uint8)
    image[..., 3] = 255  # opaque alpha; irrelevant since clip is disabled above
    return image


def test_normal_is_carried_through_unmodified():
    """The core of Change 1: normal must reach compute_pointmap's return
    dict byte-for-byte, not be recomputed or transformed."""
    normal = torch.zeros((2, 2, 3))
    normal[0, 0] = torch.tensor([0.1, -0.2, -0.97])
    pipeline = _fake_pipeline(_FakeDepthModel({"normal": normal}))

    result = pipeline.compute_pointmap(_rgba_image())

    assert torch.equal(result["normal"], normal)


def test_normal_is_none_when_depth_model_has_no_normal_head():
    """MoGe v1, or a MoGe-2 checkpoint without a normal head: must not
    fabricate a normal map when the depth model didn't provide one."""
    pipeline = _fake_pipeline(_FakeDepthModel())  # no "normal" key at all

    result = pipeline.compute_pointmap(_rgba_image())

    assert result["normal"] is None


def test_normal_is_none_for_externally_supplied_pointmap():
    """No depth-model inference runs on this path (an external pointmap was
    passed in directly), so there is no per-pixel normal to offer."""
    pipeline = _fake_pipeline(_FakeDepthModel({"normal": torch.zeros((2, 2, 3))}))
    external_pointmap = torch.ones((2, 2, 3))

    result = pipeline.compute_pointmap(_rgba_image(), pointmap=external_pointmap)

    assert result["normal"] is None


def test_intrinsics_passthrough_is_not_rotated_by_camera_convention_transform():
    """intrinsics comes straight from the depth model's own output -- unlike
    `pointmap`, it is never run through camera_convention_transform. A
    non-trivial (non-identity) matrix makes an accidental rotation visible."""
    intrinsics = torch.tensor([[1.2, 0.0, 0.5], [0.0, 1.5, 0.5], [0.0, 0.0, 1.0]])
    pipeline = _fake_pipeline(_FakeDepthModel({"intrinsics": intrinsics}))

    result = pipeline.compute_pointmap(_rgba_image())

    assert torch.equal(result["intrinsics"], intrinsics)


def test_normal_and_intrinsics_keys_always_present():
    """Purely additive: the two new keys exist on every call (None when
    unavailable), alongside the pre-existing pts_color/intrinsics/pointmap."""
    pipeline = _fake_pipeline(_FakeDepthModel())

    result = pipeline.compute_pointmap(_rgba_image())

    assert set(result.keys()) == {"pts_color", "intrinsics", "normal", "pointmap"}
