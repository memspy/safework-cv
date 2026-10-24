from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

POSE_VARIANTS = ("lite", "full", "heavy")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce(name: str, default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        expected = "true or false"
    elif isinstance(default, int):
        if _is_number(value) and float(value).is_integer():
            return int(value)
        expected = "an integer"
    elif isinstance(default, float):
        if _is_number(value):
            return float(value)
        expected = "a number"
    elif isinstance(default, str):
        if isinstance(value, str):
            return value
        expected = "a string"
    else:
        numbers = isinstance(value, (list, tuple)) and all(_is_number(v) for v in value)
        if numbers and len(value) == len(default):
            return tuple(float(v) for v in value)
        expected = f"a list of {len(default)} numbers"
    raise ValueError(f"config key '{name}' must be {expected}, got {value!r}")


@dataclass(frozen=True)
class Config:
    device: str = ""
    model_dir: str = ""

    yolo_weights: str = "yolo11n.pt"
    yolo_conf: float = 0.15
    yolo_imgsz: int = 640
    yolo_nms_iou: float = 0.6

    rotated_scan_seconds: float = 1.2
    rotated_conf: float = 0.4
    rotated_min_aspect: float = 1.1
    rotated_dedup_iou: float = 0.3
    recovery_min_gap_seconds: float = 0.3
    recovery_interval_seconds: float = 0.6
    recovery_crop_ratio: float = 2.2
    max_rotated_regions: int = 2

    color_fire: bool = True
    hazard_weights: str = ""
    hazard_conf: float = 0.5
    hazard_interval_seconds: float = 0.4
    hazard_hold_seconds: float = 2.0
    hazard_gap_seconds: float = 1.0
    fire_hue_min: int = 5
    fire_hue_max: int = 32
    fire_sat_min: int = 110
    fire_val_min: int = 210
    fire_work_width: int = 320
    fire_window_seconds: float = 1.5
    fire_min_frames: int = 6
    fire_min_area_ratio: float = 0.004
    fire_max_area_ratio: float = 0.5
    fire_min_flicker: float = 0.25
    fire_person_margin: float = 0.1

    enable_pose: bool = True
    pose_variant: str = "lite"
    pose_confidence: float = 0.3
    pose_interval_seconds: float = 0.2
    pose_hold_seconds: float = 1.0
    pose_trust_seconds: float = 0.5
    crop_padding: float = 0.15
    min_visibility: float = 0.5
    smoothing_alpha: float = 0.6

    new_track_score: float = 0.45
    confirm_min_score: float = 0.35
    confirm_hits: int = 4
    confirm_seconds: float = 0.8
    slow_confirm_hits: int = 2
    slow_ttl_seconds: float = 4.0
    match_gate: float = 0.8
    relink_radius_ratio: float = 1.5
    tentative_ttl_seconds: float = 0.6
    present_grace_seconds: float = 2.5
    track_reset_gap_seconds: float = 3.0
    teleport_ratio: float = 1.2
    duplicate_overlap: float = 0.5

    camera_up: tuple[float, float] = (0.0, -1.0)
    upright_angle_deg: float = 35.0
    lying_angle_deg: float = 60.0
    upright_extent_ratio: float = 1.2
    lying_strict_ratio: float = 0.6
    lying_loose_ratio: float = 0.9
    posture_release_seconds: float = 0.6

    edge_distance_ratio: float = 0.3
    edge_hold_seconds: float = 0.3

    immobile_seconds: float = 8.0
    motion_eps: float = 0.05
    box_motion_eps: float = 0.1

    fall_window_seconds: float = 2.0
    fall_min_drop_ratio: float = 0.25
    fall_min_speed: float = 0.5
    fall_confirm_seconds: float = 0.4
    fall_alert_seconds: float = 8.0

    speed_window_seconds: float = 1.2
    speed_min_span_seconds: float = 0.5
    depth_gain: float = 1.5
    run_speed: float = 1.4
    run_hold_seconds: float = 0.6

    occlusion_seconds: float = 12.0
    occlusion_min_pose_hits: int = 3
    occlusion_min_hits: int = 12
    occlusion_min_height_ratio: float = 0.08
    occlusion_forget_seconds: float = 600.0
    exit_margin_ratio: float = 0.02

    history_seconds: float = 3.0
    panel_max_lines: int = 6
    alert_clear_seconds: float = 2.0
    reconnect_seconds: float = 3.0

    def __post_init__(self):
        if self.pose_variant not in POSE_VARIANTS:
            raise ValueError(f"pose_variant must be one of {POSE_VARIANTS}")
        if len(self.camera_up) != 2 or not any(self.camera_up):
            raise ValueError("camera_up must be a non-zero 2D vector")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.yolo_imgsz < 32:
            raise ValueError("yolo_imgsz must be at least 32")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        defaults = cls()
        values = {
            name: _coerce(name, getattr(defaults, name), value) for name, value in data.items()
        }
        return cls(**values)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("config file must contain a mapping")
        return cls.from_mapping(data)

    def with_overrides(self, **overrides: Any) -> Config:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        data["camera_up"] = list(self.camera_up)
        return data
