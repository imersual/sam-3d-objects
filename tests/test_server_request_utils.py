"""Tests for the /infer request/response helpers (no GPU imports)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "process" / "3d-generator")
)

from request_utils import (  # noqa: E402
    MAX_VIEWS,
    extract_metric_scale,
    normalize_views,
)


def test_legacy_request_becomes_single_view():
    specs = normalize_views("img.png", ["m1.png", "m2.png"], None)
    assert specs == [("img.png", ["m1.png", "m2.png"])]


def test_views_request_returns_one_spec_per_view():
    views = [
        {"image_path": "a.png", "mask_paths": ["am.png"]},
        {"image_path": "b.png", "mask_paths": ["bm1.png", "bm2.png"]},
    ]
    specs = normalize_views(None, None, views)
    assert specs == [("a.png", ["am.png"]), ("b.png", ["bm1.png", "bm2.png"])]


def test_views_take_precedence_over_legacy_fields():
    views = [{"image_path": "a.png", "mask_paths": ["am.png"]}]
    assert normalize_views("legacy.png", ["lm.png"], views) == [("a.png", ["am.png"])]


def test_rejects_more_than_max_views():
    views = [
        {"image_path": f"{i}.png", "mask_paths": ["m.png"]}
        for i in range(MAX_VIEWS + 1)
    ]
    with pytest.raises(ValueError, match="at most 4"):
        normalize_views(None, None, views)


def test_rejects_view_without_masks():
    views = [{"image_path": "a.png", "mask_paths": []}]
    with pytest.raises(ValueError, match="mask_path"):
        normalize_views(None, None, views)


def test_rejects_view_without_image():
    views = [{"image_path": "", "mask_paths": ["m.png"]}]
    with pytest.raises(ValueError, match="image_path"):
        normalize_views(None, None, views)


def test_rejects_empty_request():
    with pytest.raises(ValueError, match="image_path"):
        normalize_views(None, None, None)


def test_rejects_legacy_request_without_masks():
    with pytest.raises(ValueError, match="mask_paths"):
        normalize_views("img.png", [], None)


# ── extract_metric_scale ─────────────────────────────────────────────────────


def test_metric_scale_averages_the_three_components():
    assert extract_metric_scale({"scale": [[0.2, 0.2, 0.2]]}) == pytest.approx(0.2)


def test_metric_scale_accepts_a_bare_float():
    assert extract_metric_scale({"scale": 1.75}) == pytest.approx(1.75)


def test_metric_scale_reads_tensor_like_objects():
    class FakeTensor:
        """Stands in for the torch tensor SAM3D actually returns."""

        def __init__(self, values):
            self._values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, _shape):
            return self

        def tolist(self):
            return self._values

    assert extract_metric_scale({"scale": FakeTensor([0.5, 0.5, 0.5])}) == pytest.approx(
        0.5
    )


def test_metric_scale_is_none_when_absent():
    # multiview results carry no usable scale
    assert extract_metric_scale({"glb": object()}) is None
    assert extract_metric_scale({"scale": None}) is None


def test_metric_scale_rejects_non_positive_and_non_finite_values():
    assert extract_metric_scale({"scale": [0.0, 0.0, 0.0]}) is None
    assert extract_metric_scale({"scale": [-1.0]}) is None
    assert extract_metric_scale({"scale": [float("nan")]}) is None
    assert extract_metric_scale({"scale": [float("inf")]}) is None


def test_metric_scale_rejects_garbage():
    assert extract_metric_scale({"scale": ["big"]}) is None
    assert extract_metric_scale({"scale": []}) is None


def test_metric_scale_honours_the_not_metric_flag():
    """A scale in SSI units must never be baked into the mesh."""
    assert extract_metric_scale({"scale": [0.2], "scale_is_metric": False}) is None
    assert extract_metric_scale(
        {"scale": [0.2], "scale_is_metric": True}
    ) == pytest.approx(0.2)
    # absent flag stays metric — the single-view path predates it
    assert extract_metric_scale({"scale": [0.2]}) == pytest.approx(0.2)
