from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
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

    camera_up: tuple = (0.0, -1.0)
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
    occlusion_min_height_ratio: float = 0.08
    occlusion_forget_seconds: float = 600.0
    exit_margin_ratio: float = 0.02

    history_seconds: float = 3.0
    panel_max_lines: int = 6
    display_scale: float = 1.0
    edges_path: str = "edges.json"

    @property
    def pose_model_path(self):
        return f"models/pose_landmarker_{self.pose_variant}.task"

    @property
    def pose_model_url(self):
        name = f"pose_landmarker_{self.pose_variant}"
        return (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"{name}/float16/latest/{name}.task"
        )