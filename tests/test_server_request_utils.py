"""Tests for the /infer request/response helpers (no GPU imports)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "process" / "3d-generator")
)

from request_utils import (  # noqa: E402
    MAX_VIEWS,
    extract_intrinsics,
    extract_metric_scale,
    extract_pose,
    normal_map_sidecar_path,
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


# ── extract_pose ──────────────────────────────────────────────────────────────


class _FakeTensor:
    """Stands in for the small torch tensors SAM3D actually returns for
    rotation/translation/scale (see extract_metric_scale's FakeTensor above,
    duplicated at module scope so every extract_pose test can share it)."""

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


def _full_pose_output(**overrides):
    output = {
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.1, 0.2, 0.3],
        "scale": [0.5, 0.5, 0.5],
        "scale_is_metric": True,
        "iou": 0.87,
    }
    output.update(overrides)
    return output


def test_pose_extracts_every_field_when_fully_available():
    assert extract_pose(_full_pose_output()) == {
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.1, 0.2, 0.3],
        "scale": [0.5, 0.5, 0.5],
        "scale_is_metric": True,
        "iou": 0.87,
    }


def test_pose_reads_tensor_like_objects():
    output = _full_pose_output(
        rotation=_FakeTensor([1.0, 0.0, 0.0, 0.0]),
        translation=_FakeTensor([0.1, 0.2, 0.3]),
        scale=_FakeTensor([0.5, 0.5, 0.5]),
    )
    pose = extract_pose(output)
    assert pose["rotation"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert pose["translation"] == pytest.approx([0.1, 0.2, 0.3])
    assert pose["scale"] == pytest.approx([0.5, 0.5, 0.5])


def test_pose_is_all_none_for_multiview():
    """run_multi_view never attempts layout postprocess. server.py passes
    None for multiview results -- nulls, not an invented pose."""
    assert extract_pose(None) == {
        "rotation": None,
        "translation": None,
        "scale": None,
        "scale_is_metric": None,
        "iou": None,
    }


def test_pose_is_all_none_for_empty_dict():
    assert extract_pose({}) == {
        "rotation": None,
        "translation": None,
        "scale": None,
        "scale_is_metric": None,
        "iou": None,
    }


def test_pose_iou_minus_one_means_post_optimization_was_skipped():
    """-1.0 is a real, meaningful sentinel (occlusion check / no-target-points
    bail in layout_post_optimization): the pose fields are still the
    unrefined estimate, not garbage, and iou must be surfaced as -1.0
    verbatim -- not None, not clamped to 0.0."""
    pose = extract_pose(_full_pose_output(iou=-1.0))
    assert pose["iou"] == -1.0
    assert pose["rotation"] == [1.0, 0.0, 0.0, 0.0]


def test_pose_iou_absent_means_post_optimization_threw_or_never_ran():
    """When run_post_optimization[_GS] raises, InferencePipelinePointMap.run
    catches and logs it without updating ss_return_dict, so `iou` is simply
    absent while rotation/translation/scale (the pre-optimization pose_decoder
    estimate) remain. Must come back as iou=None, not fabricated."""
    output = _full_pose_output()
    del output["iou"]
    pose = extract_pose(output)
    assert pose["iou"] is None
    assert pose["rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert pose["translation"] == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("bad_iou", [float("nan"), float("inf"), float("-inf")])
def test_pose_rejects_non_finite_iou(bad_iou):
    assert extract_pose(_full_pose_output(iou=bad_iou))["iou"] is None


def test_pose_rejects_garbage_iou():
    assert extract_pose(_full_pose_output(iou="not-a-number"))["iou"] is None


@pytest.mark.parametrize("bad_flag", [None, "true", 1, 0])
def test_pose_scale_is_metric_must_be_an_actual_bool(bad_flag):
    """0/1/"true" look boolish but are not bool; only True/False count."""
    assert extract_pose(_full_pose_output(scale_is_metric=bad_flag))[
        "scale_is_metric"
    ] is None


def test_pose_scale_is_metric_false_is_preserved():
    """False is a legitimate, meaningful value -- must not collapse to None."""
    assert extract_pose(_full_pose_output(scale_is_metric=False))[
        "scale_is_metric"
    ] is False


def test_pose_rotation_garbage_becomes_none_independently():
    """One malformed field must not blank out the rest of the pose."""
    pose = extract_pose(_full_pose_output(rotation=["not", "numbers"]))
    assert pose["rotation"] is None
    assert pose["translation"] == [0.1, 0.2, 0.3]
    assert pose["scale"] == [0.5, 0.5, 0.5]
    assert pose["iou"] == 0.87


def test_pose_return_has_exactly_the_documented_keys():
    assert set(extract_pose(_full_pose_output()).keys()) == {
        "rotation",
        "translation",
        "scale",
        "scale_is_metric",
        "iou",
    }


# ── extract_intrinsics ───────────────────────────────────────────────────────

_SAMPLE_INTRINSICS = [[1.2, 0.0, 0.5], [0.0, 1.5, 0.5], [0.0, 0.0, 1.0]]
_SAMPLE_INTRINSICS_FLAT = [1.2, 0.0, 0.5, 0.0, 1.5, 0.5, 0.0, 0.0, 1.0]


def test_intrinsics_extracts_a_3x3_nested_list():
    assert extract_intrinsics({"intrinsics": _SAMPLE_INTRINSICS}) == _SAMPLE_INTRINSICS


def test_intrinsics_reads_tensor_like_objects():
    """Same tensor-like duck type extract_pose/extract_metric_scale accept."""
    assert (
        extract_intrinsics({"intrinsics": _FakeTensor(_SAMPLE_INTRINSICS_FLAT)})
        == _SAMPLE_INTRINSICS
    )


def test_intrinsics_is_none_when_absent():
    assert extract_intrinsics({"glb": object()}) is None
    assert extract_intrinsics({"intrinsics": None}) is None


def test_intrinsics_is_none_for_multiview():
    """run_multi_view keeps no single intrinsics matrix (one per view,
    computed and discarded internally); server.py passes None for multiview
    results -- nulls, not an invented frame, same as extract_pose(None)."""
    assert extract_intrinsics(None) is None
    assert extract_intrinsics({}) is None


def test_intrinsics_rejects_wrong_element_count():
    assert extract_intrinsics({"intrinsics": [1.0, 2.0, 3.0]}) is None
    assert extract_intrinsics({"intrinsics": []}) is None
    assert extract_intrinsics({"intrinsics": _SAMPLE_INTRINSICS_FLAT + [1.0]}) is None


def test_intrinsics_rejects_non_finite_values():
    for bad in (float("nan"), float("inf"), float("-inf")):
        flat = list(_SAMPLE_INTRINSICS_FLAT)
        flat[-1] = bad
        assert extract_intrinsics({"intrinsics": flat}) is None


def test_intrinsics_rejects_garbage():
    assert extract_intrinsics({"intrinsics": ["a", "b", "c"]}) is None


def test_intrinsics_accepts_a_flat_9_element_list_row_major():
    """A flat (non-nested) 9-element input is reshaped row-major into 3x3."""
    assert extract_intrinsics({"intrinsics": _SAMPLE_INTRINSICS_FLAT}) == _SAMPLE_INTRINSICS


# ── normal_map_sidecar_path ──────────────────────────────────────────────────


def test_normal_map_sidecar_path_same_directory_and_stem():
    assert normal_map_sidecar_path("out/mesh.glb") == "out/mesh_normal.npy"


def test_normal_map_sidecar_path_with_no_directory():
    assert normal_map_sidecar_path("mesh.glb") == "mesh_normal.npy"


def test_normal_map_sidecar_path_preserves_dots_in_the_stem():
    # pathlib's .stem strips only the LAST suffix.
    assert normal_map_sidecar_path("out/mesh.v2.glb") == "out/mesh.v2_normal.npy"


def test_normal_map_sidecar_path_accepts_path_like_input():
    from pathlib import PurePosixPath

    assert (
        normal_map_sidecar_path(PurePosixPath("out/mesh.glb"))
        == "out/mesh_normal.npy"
    )


def test_normal_map_sidecar_path_is_always_forward_slash():
    """Production only ever runs on the same (Linux) machine as its caller;
    as_posix() keeps the result deterministic under Windows-run tests too."""
    assert "\\" not in normal_map_sidecar_path("out/mesh.glb")
