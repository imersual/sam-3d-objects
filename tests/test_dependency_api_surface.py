"""Guards the third-party API surface SAM3D depends on.

Bumping the MoGe pin to get v2 also bumps utils3d. SAM3D calls utils3d and MoGe
internals directly, so a bad upgrade should fail here rather than inside a
rendering or focal-recovery call.
"""
import pytest

utils3d = pytest.importorskip("utils3d")

UTILS3D_TORCH_FUNCTIONS = [
    "extrinsics_look_at",
    "intrinsics_from_fov_xy",
    "perspective_from_fov_xy",
    "view_look_at",
    "RastContext",
    "rasterize_triangle_faces",
    "compute_edges",
    "compute_connected_components",
    "compute_dual_graph",
    "compute_edge_connected_components",
    "remove_unreferenced_vertices",
    "extrinsics_to_view",
    "intrinsics_to_perspective",
    "intrinsics_from_focal_center",
]


@pytest.mark.parametrize("name", UTILS3D_TORCH_FUNCTIONS)
def test_utils3d_torch_surface(name):
    assert hasattr(utils3d.torch, name), f"utils3d.torch.{name} is missing"


def test_utils3d_numpy_depth_edge():
    assert hasattr(utils3d.numpy, "depth_edge")


def test_utils3d_io_write_ply():
    assert hasattr(utils3d.io, "write_ply")


def test_moge_v2_importable():
    pytest.importorskip("moge")
    from moge.model.v2 import MoGeModel

    assert hasattr(MoGeModel, "from_pretrained")


def test_moge_v1_still_importable():
    """v1 is the rollback path and must keep working."""
    pytest.importorskip("moge")
    from moge.model.v1 import MoGeModel

    assert hasattr(MoGeModel, "from_pretrained")


def test_moge_internal_helpers_importable():
    """sam3d_objects/pipeline/utils/pointmap.py imports these directly."""
    pytest.importorskip("moge")
    from moge.utils.geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift
    from moge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift

    assert all(
        callable(function)
        for function in (
            normalized_view_plane_uv,
            recover_focal_shift,
            solve_optimal_focal_shift,
            solve_optimal_shift,
        )
    )
