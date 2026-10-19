"""Tests for the pure helpers of scripts/diagnose_plate_food.py (no GPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_plate_food import (  # noqa: E402
    _srgb_to_lab,
    ab_summary_lines,
    background_colour,
    context_variants,
    depth_relief,
    mask_groups,
    preview_sheet,
    share_like_support,
    support_colour_model,
    surface_samples,
    voxel_height_ratio,
)


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


# ------------------------- --context-ab -------------------------


def test_mask_groups_join_comma_separated_masks():
    assert mask_groups(["a.png,b.png", "c.png"]) == [["a.png", "b.png"], ["c.png"]]
    assert mask_groups(["https://cdn/x/o_1.png"]) == [["https://cdn/x/o_1.png"]]


def _photo():
    """A 4x6 photo: a food pixel block in a ring of bowl pixels on a table."""
    image = np.zeros((4, 6, 3), np.uint8)
    image[...] = (90, 90, 90)  # table
    bowl = np.zeros((4, 6), bool)
    bowl[0:4, 1:5] = True
    food = np.zeros((4, 6), bool)
    food[1:3, 2:4] = True
    bowl &= ~food
    image[bowl] = (230, 230, 225)
    image[food] = (200, 160, 40)
    return image, food, bowl


def test_context_variants_paint_only_what_they_should():
    image, food, bowl = _photo()
    before = image.copy()

    variants = context_variants(image, food, bowl, np.array([1, 2, 3], np.uint8))

    assert set(variants) == {"as_is", "support_hidden", "cutout"}
    assert np.array_equal(image, before), "the photo itself is untouched"
    assert np.array_equal(variants["as_is"], image)
    hidden = variants["support_hidden"]
    assert (hidden[bowl] == (1, 2, 3)).all() and np.array_equal(hidden[~bowl], image[~bowl])
    cutout = variants["cutout"]
    assert (cutout[~food] == (1, 2, 3)).all() and np.array_equal(cutout[food], image[food])


def test_the_fill_is_the_photos_background():
    image, food, bowl = _photo()
    assert background_colour(image, [food, bowl]).tolist() == [90, 90, 90]
    everything = np.ones((4, 6), bool)
    assert background_colour(image, [everything]).tolist() == [128, 128, 128]


def test_srgb_to_lab_matches_reference_values():
    lab = _srgb_to_lab([[255, 255, 255], [0, 0, 0], [255, 0, 0]])
    assert lab[0] == pytest.approx([100.0, 0.0, 0.0], abs=0.05)
    assert lab[1] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


def _white_bowl_pixels(count=400):
    """A white bowl's pixels under shading: one colour, lightness 120-245."""
    shade = np.linspace(120, 245, count)
    return np.stack([shade, shade, shade * 0.98], axis=1).round().astype(np.uint8)


def test_a_white_bowl_and_its_food_are_told_apart():
    model = support_colour_model(_white_bowl_pixels())
    food = np.array([[240, 200, 60], [190, 140, 90], [200, 90, 90], [40, 120, 40]], np.uint8)

    assert share_like_support(_white_bowl_pixels(), model) > 0.8
    assert share_like_support(food, model) == 0.0


def test_dark_grey_food_is_not_counted_as_a_white_bowl():
    """Mushroom and truffle are as colourless as white ceramic, only dark."""
    model = support_colour_model(_white_bowl_pixels())
    assert share_like_support(np.array([[45, 45, 44], [70, 68, 66]], np.uint8), model) == 0.0


def test_a_coloured_plate_works_too():
    shade = np.linspace(0.5, 1.0, 300)[:, None]
    blue_plate = (np.array([40, 90, 160]) * shade).round().astype(np.uint8)
    model = support_colour_model(blue_plate)
    shrimp = np.array([[230, 120, 60], [250, 200, 170]], np.uint8)

    assert share_like_support(blue_plate, model) > 0.8
    assert share_like_support(shrimp, model) == 0.0


def _textured(mesh, colour):
    import trimesh
    from PIL import Image

    uv = np.random.default_rng(1).uniform(0.1, 0.9, (len(mesh.vertices), 2))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.new("RGB", (16, 16), colour))
    return mesh


def test_surface_samples_read_the_texture_and_weigh_by_area():
    """A dish under the food matters for its area, not its vertex count."""
    import trimesh

    big = _textured(trimesh.creation.box(extents=(3.0, 0.1, 3.0)), (230, 230, 230))
    points, colours = surface_samples(big, count=2000)
    assert (colours == (230, 230, 230)).all()
    assert points.min(axis=0) == pytest.approx([-1.5, -0.05, -1.5], abs=0.05)

    small = trimesh.creation.icosphere(subdivisions=4, radius=0.3)
    small.apply_translation([0.0, 1.0, 0.0])
    whole = trimesh.util.concatenate([trimesh.Trimesh(big.vertices, big.faces), small])
    whole.visual = trimesh.visual.ColorVisuals(
        whole, vertex_colors=np.vstack([
            np.tile([230, 230, 230, 255], (len(big.vertices), 1)),
            np.tile([200, 40, 40, 255], (len(small.vertices), 1)),
        ]).astype(np.uint8),
    )
    _, colours = surface_samples(whole, count=20000)
    red_share = float((colours[:, 1] < 100).mean())
    assert red_share == pytest.approx(small.area / (small.area + big.area), abs=0.02)
    assert len(small.vertices) > len(big.vertices), "fixture: more vertices, less area"


def test_preview_sheet_shows_four_views(tmp_path):
    import trimesh
    from PIL import Image

    mesh = _textured(trimesh.creation.box(extents=(1.0, 0.4, 1.0)), (200, 30, 30))
    path = tmp_path / "preview.png"

    preview_sheet(mesh, path, size=64, samples=5000)

    sheet = np.asarray(Image.open(path).convert("RGB")).astype(int)
    assert sheet.shape == (64, 4 * 64, 3)
    for tile in np.split(sheet, 4, axis=1):
        red = (tile[..., 0] > 150) & (tile[..., 1] < 80)
        assert red.sum() > 200


def test_ab_summary_reports_the_excess_over_the_photo():
    rows = [
        {"variant": "as_is", "seed": 1, "like_support": 0.334, "height_ratio": 0.31, "glb": "ab_as_is_seed1.glb"},
        {"variant": "cutout", "seed": 1, "like_support": 0.02, "height_ratio": 0.42, "glb": "ab_cutout_seed1.glb"},
    ]

    lines = ab_summary_lines(rows, photo_share=0.018)

    assert lines[0] == "food pixels in the photo coloured like the support: 1.8%"
    assert "+31.6" in lines[3] and "ab_as_is_seed1.glb" in lines[3]
    assert "+0.2" in lines[4]


def test_context_ab_options():
    from diagnose_plate_food import _parse_args

    args = _parse_args(["--image", "i.jpg", "--masks", "f.png,g.png", "b.png", "--out", "o", "--context-ab"])

    assert args.context_ab and args.ab_seeds == 2 and args.fill == "background"
    assert mask_groups(args.masks) == [["f.png", "g.png"], ["b.png"]]
