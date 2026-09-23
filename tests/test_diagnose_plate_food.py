"""Tests for the pure helpers of scripts/diagnose_plate_food.py (no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_plate_food import depth_relief, voxel_height_ratio  # noqa: E402


def _voxels(x, y, z):
    """Stage-1 coords for the box x0..x1 by y0..y1 by z0..z1 (inclusive): the
    (N, 4) [batch, x, y, z] layout sample_sparse_structure returns, z up."""
    grid = np.stack(np.meshgrid(np.arange(*x), np.arange(*y), np.arange(*z), indexing="ij"), -1)
    grid = grid.reshape(-1, 3)
    return np.concatenate([np.zeros((len(grid), 1), dtype=int), grid], axis=1)


def test_voxel_height_ratio_of_a_bare_plate_is_small():
    plate = _voxels((0, 20), (0, 20), (0, 2))  # 20 wide, 2 tall
    assert voxel_height_ratio(plate) == pytest.approx(2 / 20)


def test_voxel_height_ratio_sees_food_standing_on_the_plate():
    plate = _voxels((0, 20), (0, 20), (0, 2))
    cake = _voxels((6, 12), (6, 12), (2, 10))  # 8 voxels tall, on top
    assert voxel_height_ratio(np.concatenate([plate, cake])) == pytest.approx(10 / 20)


def test_voxel_height_ratio_uses_the_wider_horizontal_side():
    tray = _voxels((0, 30), (0, 10), (0, 3))
    assert voxel_height_ratio(tray) == pytest.approx(3 / 30)


def test_depth_relief_is_how_far_the_food_stands_out_toward_the_camera():
    """Pointmap in the pipeline's convention: Z is distance from the camera.
    The plate is 0.30 wide at Z=1.00; the cake's visible top is 0.05 nearer.
    Relief is that gap as a fraction of the plate's width: 0.05 / 0.30."""
    height, width = 4, 6
    pointmap = np.zeros((height, width, 3))
    pointmap[..., 0] = np.linspace(-0.15, 0.15, width)[None, :]
    pointmap[..., 2] = 1.0
    cake = np.zeros((height, width), bool)
    cake[1:3, 2:4] = True
    pointmap[cake, 2] = 0.95
    plate = ~cake

    assert depth_relief(pointmap, cake, plate) == pytest.approx(0.05 / 0.30)


def test_depth_relief_ignores_invalid_depth():
    """Pixels the depth model could not predict are inf (or nan); they must
    not drag the medians anywhere."""
    pointmap = np.zeros((2, 3, 3))
    pointmap[..., 0] = [[-0.1, 0.0, 0.1], [-0.1, 0.0, 0.1]]
    pointmap[..., 2] = 1.0
    cake = np.array([[False, True, False], [False, False, False]])
    pointmap[0, 1, 2] = 0.9
    pointmap[1, 1, 2] = np.inf
    plate = ~cake

    assert depth_relief(pointmap, cake, plate) == pytest.approx(0.1 / 0.2)
