"""Tests for scripts/set_depth_model.py (no GPU imports)."""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from set_depth_model import main, set_depth_model  # noqa: E402


def _v1_config():
    """A config shaped like the depth_model block of the downloaded pipeline.yaml."""
    return OmegaConf.create(
        {
            "pose_decoder_name": "ScaleShiftInvariant",
            "depth_model": {
                "_target_": "sam3d_objects.pipeline.depth_models.moge.MoGe",
                "model": {
                    "_target_": "moge.model.v1.MoGeModel.from_pretrained",
                    "pretrained_model_name_or_path": "Ruicheng/moge-vitl",
                },
            },
        }
    )


def test_moge2_rewrites_all_three_fields():
    config = set_depth_model(_v1_config(), "moge2")
    assert config.depth_model._target_ == (
        "sam3d_objects.pipeline.depth_models.moge2.MoGe2"
    )
    assert config.depth_model.model._target_ == (
        "moge.model.v2.MoGeModel.from_pretrained"
    )
    assert config.depth_model.model.pretrained_model_name_or_path == (
        "Ruicheng/moge-2-vitl-normal"
    )


def test_unrelated_keys_untouched():
    config = set_depth_model(_v1_config(), "moge2")
    assert config.pose_decoder_name == "ScaleShiftInvariant"


def test_idempotent():
    once = set_depth_model(_v1_config(), "moge2")
    twice = set_depth_model(set_depth_model(_v1_config(), "moge2"), "moge2")
    assert once == twice


def test_rollback_restores_v1():
    config = set_depth_model(set_depth_model(_v1_config(), "moge2"), "moge1")
    assert config.depth_model._target_ == (
        "sam3d_objects.pipeline.depth_models.moge.MoGe"
    )
    assert config.depth_model.model.pretrained_model_name_or_path == (
        "Ruicheng/moge-vitl"
    )


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        set_depth_model(_v1_config(), "moge3")


def test_main_round_trips_the_file(tmp_path):
    (tmp_path / "hf").mkdir()
    path = tmp_path / "hf" / "pipeline.yaml"
    OmegaConf.save(_v1_config(), path)

    main(["--tag", "hf", "--variant", "moge2", "--checkpoints-root", str(tmp_path)])

    reloaded = OmegaConf.load(path)
    assert reloaded.depth_model.model.pretrained_model_name_or_path == (
        "Ruicheng/moge-2-vitl-normal"
    )
    assert reloaded.pose_decoder_name == "ScaleShiftInvariant"


def test_main_exits_when_file_missing(tmp_path):
    with pytest.raises(SystemExit):
        main(["--tag", "nope", "--checkpoints-root", str(tmp_path)])
