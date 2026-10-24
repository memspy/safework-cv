from pathlib import Path

import pytest
import yaml

from safework_cv.config import Config

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_yaml_matches_code_defaults():
    assert Config.from_file(DEFAULT_YAML) == Config()


def test_default_yaml_covers_every_field():
    keys = set(yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8")))
    assert keys == set(Config().to_mapping())


def test_partial_file_overrides_only_given_keys(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text("run_speed: 2.0\ncamera_up: [0.2, -1.0]\n", encoding="utf-8")
    cfg = Config.from_file(path)
    assert cfg.run_speed == 2.0
    assert cfg.camera_up == (0.2, -1.0)
    assert cfg.immobile_seconds == Config().immobile_seconds


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown config keys: run_spead"):
        Config.from_mapping({"run_spead": 2.0})


@pytest.mark.parametrize(
    "bad",
    [
        {"pose_variant": "huge"},
        {"camera_up": (0.0, 0.0)},
        {"camera_up": (1.0, 2.0, 3.0)},
        {"smoothing_alpha": 0.0},
        {"yolo_imgsz": 8},
    ],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValueError):
        Config(**bad)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("run_speed", "fast"),
        ("enable_pose", "yes"),
        ("color_fire", 1),
        ("yolo_imgsz", 640.5),
        ("device", 0),
        ("camera_up", [0.0]),
        ("camera_up", "up"),
    ],
)
def test_wrong_value_types_are_rejected_with_the_key_name(key, value):
    with pytest.raises(ValueError, match=key):
        Config.from_mapping({key: value})


def test_integers_are_accepted_for_floats_and_integral_floats_for_integers():
    cfg = Config.from_mapping({"run_speed": 2, "yolo_imgsz": 640.0, "camera_up": [0, -1]})
    assert cfg.run_speed == 2.0 and isinstance(cfg.run_speed, float)
    assert cfg.yolo_imgsz == 640 and isinstance(cfg.yolo_imgsz, int)
    assert cfg.camera_up == (0.0, -1.0)


def test_non_mapping_file_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        Config.from_file(path)


def test_with_overrides_skips_none():
    cfg = Config().with_overrides(device="cpu", yolo_weights=None, enable_pose=False)
    assert cfg.device == "cpu"
    assert cfg.enable_pose is False
    assert cfg.yolo_weights == Config().yolo_weights


def test_roundtrip_through_mapping():
    cfg = Config(run_speed=2.5, camera_up=(0.1, -1.0))
    assert Config.from_mapping(cfg.to_mapping()) == cfg
