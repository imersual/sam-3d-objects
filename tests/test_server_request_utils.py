"""Tests for the /infer request normalization helper (no GPU imports)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "process" / "3d-generator")
)

from request_utils import MAX_VIEWS, normalize_views  # noqa: E402


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
