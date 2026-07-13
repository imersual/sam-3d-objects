"""Tests for run_inference.py pure helpers (no GPU imports at module level)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from run_inference import default_out_dir, parse_decode_formats, parse_image_names


def test_parse_image_names_none_and_empty():
    assert parse_image_names(None) is None
    assert parse_image_names("") is None
    assert parse_image_names(" , ,") is None


def test_parse_image_names_list():
    assert parse_image_names("image1, view_a,2") == ["image1", "view_a", "2"]
    assert parse_image_names("solo") == ["solo"]


def test_parse_decode_formats():
    assert parse_decode_formats("gaussian,mesh") == ["gaussian", "mesh"]
    assert parse_decode_formats(" gaussian ") == ["gaussian"]
    assert parse_decode_formats("") == ["gaussian", "mesh"]  # default fallback


def test_default_out_dir_multiview_all_images():
    out = default_out_dir(Path("data/images_and_masks"), None, None, False)
    assert out == Path("output/multiview/images_and_masks_multiview")


def test_default_out_dir_mask_prompt():
    out = default_out_dir(Path("data"), "stuffed_toy", None, False)
    assert out == Path("output/multiview/stuffed_toy_multiview")


def test_default_out_dir_single_view():
    out = default_out_dir(Path("data"), "toy", ["3"], True)
    assert out == Path("output/multiview/toy_3")


def test_default_out_dir_named_views():
    out = default_out_dir(Path("data"), None, ["1", "2", "3", "4"], False)
    assert out == Path("output/multiview/data_1_2_3_and_1_more")
