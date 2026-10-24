from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import Config
from .geometry import (
    angle_between_deg,
    box_center,
    box_corners,
    box_cover_fraction,
    box_scale,
    projected_extents,
    unit_vector,
)
from .tracker import Tracker

SHOULDERS = [11, 12]
HIPS = [23, 24]


@dataclass
class Posture:
    angle: float | None
    upright: bool
    lying: bool


@dataclass
class Sample:
    t: float
    centroid: np.ndarray
    raw_centroid: np.ndarray
    foot: np.ndarray
    scale: float
    upright: bool


def torso_angle(keypoints, mask, up):
    if not (mask[SHOULDERS].all() and mask[HIPS].all()):
        return None
    shoulders = keypoints[SHOULDERS].mean(axis=0)
    hips = keypoints[HIPS].mean(axis=0)
    return angle_between_deg(shoulders - hips, up)


def classify_posture(cfg, keypoints, visibility, box):
    up = unit_vector(cfg.camera_up)
    ext_up, ext_side = projected_extents(box_corners(box), up)
    ratio = ext_up / max(ext_side, 1e-6)
    angle = None
    if keypoints is not None:
        angle = torso_angle(keypoints, visibility >= cfg.min_visibility, up)
    upright = ratio >= cfg.upright_extent_ratio and (angle is None or angle < cfg.upright_angle_deg)
    loose = ratio <= cfg.lying_loose_ratio and angle is not None and angle > cfg.lying_angle_deg
    lying = ratio <= cfg.lying_strict_ratio or loose
    return Posture(angle, upright, lying)


class PersonState:
    def __init__(self, track_id, cfg: Config):
        self.track_id = track_id
        self.cfg = cfg
        self.up = unit_vector(cfg.camera_up)
        self.history = deque()
        self.anchors = {}
        self.box = None
        self.keypoints = None
        self.visibility = None
        self.pose_t = None
        self.pose_attempt_t = None
        self.pose_fresh = False
        self.pose_hits = 0
        self.hits = 0
        self.centroid = None
        self.raw_centroid = None
        self.foot = None
        self.scale = None
        self.last_seen = None
        self.last_point = None
        self.present = False
        self.coasting = False
        self.exited = False
        self.confirmed = False
        self.occlusion_eligible = False
        self.posture = None
        self.lying_since = None
        self.not_lying_since = None
        self.last_motion_t = None
        self.speed = 0.0
        self.running_since = None
        self.fall_candidate = None
        self.fall_t = None
        self.edge_since = None
        self.edge_ratio = None

    @property
    def lying(self):
        return self.lying_since is not None

    def pose_due(self, t):
        return (
            self.pose_attempt_t is None or t - self.pose_attempt_t >= self.cfg.pose_interval_seconds
        )

    def update(self, det, track, t):
        if self._should_reset(det, t):
            self._reset_tracking()
        self.present = True
        self.coasting = False
        self.exited = False
        self.confirmed = track.confirmed
        self.hits = track.hits
        self.last_seen = t
        self._smooth(det, t)
        self.pose_fresh = (
            self.keypoints is not None and t - self.pose_t <= self.cfg.pose_trust_seconds
        )
        keypoints = self.keypoints if self.pose_fresh else None
        self.posture = classify_posture(self.cfg, keypoints, self.visibility, self.box)
        self._update_lying(t)
        self._update_motion(t)
        self._push_sample(t)
        self._update_speed(t)
        self._update_running(t)
        self._update_fall(t)

    def mark_missing(self, t, frame_w, frame_h):
        if not self.present or t - self.last_seen <= self.cfg.present_grace_seconds:
            return
        self.present = False
        m = self.cfg.exit_margin_ratio
        x1, y1, x2, y2 = self.box
        self.exited = (
            x1 < m * frame_w or x2 > (1 - m) * frame_w or y1 < m * frame_h or y2 > (1 - m) * frame_h
        )
        self.occlusion_eligible = (
            self.confirmed
            and self._observed_as_person()
            and self.scale >= self.cfg.occlusion_min_height_ratio * frame_h
        )

    def _observed_as_person(self):
        if self.cfg.enable_pose:
            return self.pose_hits >= self.cfg.occlusion_min_pose_hits
        return self.hits >= self.cfg.occlusion_min_hits

    def _should_reset(self, det, t):
        if self.last_seen is None or self.centroid is None:
            return True
        if t - self.last_seen > self.cfg.track_reset_gap_seconds:
            return True
        jump = np.linalg.norm(box_center(det.box) - self.centroid) / self.scale
        return jump > self.cfg.teleport_ratio

    def _reset_tracking(self):
        self.history.clear()
        self.anchors = {}
        self.centroid = None
        self.foot = None
        self.keypoints = None
        self.running_since = None
        self.fall_candidate = None

    def _smooth(self, det, t):
        a = self.cfg.smoothing_alpha
        box = det.box
        centroid = box_center(box)
        foot = np.array([(box[0] + box[2]) / 2, box[3]])
        scale = box_scale(box)
        old_centroid = self.centroid
        old_scale = self.scale
        self.raw_centroid = centroid
        if old_centroid is None:
            self.centroid = centroid
            self.foot = foot
            self.scale = scale
        else:
            self.centroid = a * centroid + (1 - a) * old_centroid
            self.foot = a * foot + (1 - a) * self.foot
            self.scale = a * scale + (1 - a) * old_scale
        self.box = box
        self.last_point = self.centroid.copy()
        self._update_pose(det, old_centroid, old_scale, t)

    def _update_pose(self, det, old_centroid, old_scale, t):
        a = self.cfg.smoothing_alpha
        if det.keypoints is not None:
            self.pose_hits += 1
            self.pose_t = t
            if self.keypoints is None:
                self.keypoints = det.keypoints.copy()
            else:
                self.keypoints = a * det.keypoints + (1 - a) * self.keypoints
            self.visibility = det.visibility
            return
        expired = self.pose_t is None or t - self.pose_t > self.cfg.pose_hold_seconds
        if self.keypoints is None or old_centroid is None or expired:
            self.keypoints = None
            self.visibility = None
            return
        ratio = self.scale / max(old_scale, 1e-6)
        self.keypoints = self.centroid + (self.keypoints - old_centroid) * ratio

    def _update_lying(self, t):
        if self.posture.lying:
            self.not_lying_since = None
            if self.lying_since is None:
                self.lying_since = t
        elif self.lying_since is not None:
            if self.not_lying_since is None:
                self.not_lying_since = t
            elif t - self.not_lying_since > self.cfg.posture_release_seconds:
                self.lying_since = None
                self.not_lying_since = None

    def _motion_sources(self):
        sources = [("box", box_corners(self.box), np.ones(4, dtype=bool), self.cfg.box_motion_eps)]
        if self.pose_fresh:
            mask = self.visibility >= self.cfg.min_visibility
            if mask.sum() >= 8:
                sources.append(("pose", self.keypoints, mask, self.cfg.motion_eps))
        return sources

    def _score(self, points, mask, anchor):
        if anchor is None:
            return None
        anchor_points, anchor_mask = anchor
        common = mask & anchor_mask
        if common.sum() < 4:
            return None
        diff = np.linalg.norm(points[common] - anchor_points[common], axis=1)
        return float(diff.mean() / self.scale)

    def _update_motion(self, t):
        moved = False
        measured = False
        for kind, points, mask, eps in self._motion_sources():
            score = self._score(points, mask, self.anchors.get(kind))
            if score is None or score > eps:
                self.anchors[kind] = (points.copy(), mask.copy())
            if score is not None:
                measured = True
                moved = moved or score > eps
        if moved or not measured:
            self.last_motion_t = t

    def _push_sample(self, t):
        self.history.append(
            Sample(
                t,
                self.centroid.copy(),
                self.raw_centroid.copy(),
                self.foot.copy(),
                self.scale,
                self.posture.upright,
            )
        )
        while self.history and t - self.history[0].t > self.cfg.history_seconds:
            self.history.popleft()

    def _update_speed(self, t):
        cfg = self.cfg
        samples = [s for s in self.history if t - s.t <= cfg.speed_window_seconds]
        if len(samples) < 3 or samples[-1].t - samples[0].t < cfg.speed_min_span_seconds:
            self.speed = 0.0
            return
        times = np.array([s.t for s in samples]) - samples[0].t
        feet = np.array([s.foot for s in samples])
        velocity = np.polyfit(times, feet, 1)[0]
        along = float(velocity @ self.up)
        lateral = velocity - along * self.up
        ground = lateral + cfg.depth_gain * along * self.up
        scale = float(np.mean([s.scale for s in samples]))
        self.speed = float(np.linalg.norm(ground) / scale)

    def _update_running(self, t):
        if self.speed >= self.cfg.run_speed and not self.lying:
            if self.running_since is None:
                self.running_since = t
        elif self.speed < self.cfg.run_speed * 0.6 or self.lying:
            self.running_since = None

    def _detect_drop(self, t):
        cfg = self.cfg
        for sample in self.history:
            dt = t - sample.t
            if dt <= 0 or dt > cfg.fall_window_seconds or not sample.upright:
                continue
            drop = float((sample.raw_centroid - self.raw_centroid) @ self.up) / sample.scale
            if drop >= cfg.fall_min_drop_ratio and drop / dt >= cfg.fall_min_speed:
                return t
        return None

    def _update_fall(self, t):
        cfg = self.cfg
        if self.fall_t is not None and t - self.fall_t < cfg.fall_alert_seconds:
            return
        if not self.posture.lying:
            self.fall_candidate = None
            return
        if self.fall_candidate is None:
            self.fall_candidate = self._detect_drop(t)
            return
        if t - self.fall_candidate >= cfg.fall_confirm_seconds:
            self.fall_t = t
            self.fall_candidate = None


class PersonRegistry:
    def __init__(self, cfg: Config, pose):
        self.cfg = cfg
        self.pose = pose
        self.tracker = Tracker(cfg)
        self.states: dict[int, PersonState] = {}

    def update(self, frame_bgr, detections, t, frame_w, frame_h):
        seen = set()
        for track, det in self.tracker.update(detections, t):
            state = self.states.get(track.track_id)
            if state is None:
                state = PersonState(track.track_id, self.cfg)
                self.states[track.track_id] = state
            if self.pose is not None and track.hits >= 2 and state.pose_due(t):
                det.keypoints, det.visibility = self.pose.estimate(frame_bgr, det.box)
                state.pose_attempt_t = t
            state.update(det, track, t)
            seen.add(track.track_id)
        seen_boxes = [self.states[track_id].box for track_id in seen]
        for track_id in list(self.states):
            if track_id not in self.tracker.tracks:
                del self.states[track_id]
                continue
            if track_id in seen:
                continue
            state = self.states[track_id]
            state.mark_missing(t, frame_w, frame_h)
            state.coasting = state.present
            if self._redundant(state, seen_boxes) or self._discardable(state, t):
                self.tracker.drop(track_id)
                del self.states[track_id]

    def lost_boxes(self, t):
        out = []
        for track_id, state in self.states.items():
            if not state.confirmed or state.exited:
                continue
            if t - state.last_seen >= self.cfg.recovery_min_gap_seconds:
                out.append((track_id, state.box))
        return out

    def _redundant(self, state, seen_boxes):
        return any(
            box_cover_fraction(state.box, box) >= self.cfg.duplicate_overlap for box in seen_boxes
        )

    def _discardable(self, state, t):
        if state.present:
            return False
        gap = t - state.last_seen
        if state.exited or not state.occlusion_eligible:
            return gap > 1.0
        return gap > self.cfg.occlusion_forget_seconds
