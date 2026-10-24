from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .alerts import Alert
from .config import Config

FIRE_NAMES = ("fire", "flame")
SMOKE_NAMES = ("smoke",)


class HazardMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        if cfg.hazard_weights:
            from ultralytics import YOLO

            self.model = YOLO(cfg.hazard_weights)
        self.frames = deque()
        self.last_run = None
        self.boxes = []
        self.since = {}
        self.last_hit = {}
        self.points = {}

    def update(self, frame, t, person_boxes):
        found = self._detect(frame, t, person_boxes)
        if found is not None:
            self.boxes = found
            for kind, box, _ in found:
                last = self.last_hit.get(kind)
                if last is None or t - last > self.cfg.hazard_gap_seconds:
                    self.since[kind] = t
                self.last_hit[kind] = t
                self.points[kind] = (
                    float((box[0] + box[2]) / 2),
                    float((box[1] + box[3]) / 2),
                )
        alerts = []
        for kind, since in self.since.items():
            last = self.last_hit[kind]
            if t - last > self.cfg.hazard_gap_seconds:
                continue
            if last - since < self.cfg.hazard_hold_seconds:
                continue
            alerts.append(
                Alert(kind, None, self.points[kind], f"{kind.lower()} for {last - since:.0f}s")
            )
        return alerts

    def _detect(self, frame, t, person_boxes):
        if self.model is not None:
            return self._model_hazards(frame, t)
        if self.cfg.color_fire:
            return self._color_fire(frame, t, person_boxes)
        return []

    def _model_hazards(self, frame, t):
        if self.last_run is not None and t - self.last_run < self.cfg.hazard_interval_seconds:
            return None
        self.last_run = t
        result = self.model(
            frame,
            conf=self.cfg.hazard_conf,
            imgsz=self.cfg.yolo_imgsz,
            device=self.cfg.device or None,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        found = []
        xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for box, cls, score in zip(xyxy, classes, scores, strict=True):
            name = str(result.names[int(cls)]).lower()
            if any(n in name for n in FIRE_NAMES):
                kind = "FIRE"
            elif any(n in name for n in SMOKE_NAMES):
                kind = "SMOKE"
            else:
                continue
            found.append((kind, box.astype(np.float64), float(score)))
        return found

    def _color_fire(self, frame, t, person_boxes):
        cfg = self.cfg
        ratio = min(1.0, cfg.fire_work_width / frame.shape[1])
        if ratio < 1.0:
            small = cv2.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        else:
            small = frame
        h, w = small.shape[:2]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (cfg.fire_hue_min, cfg.fire_sat_min, cfg.fire_val_min),
            (cfg.fire_hue_max, 255, 255),
        )
        covered = np.zeros((h, w), dtype=bool)
        for box in person_boxes:
            pad_x = (box[2] - box[0]) * cfg.fire_person_margin
            pad_y = (box[3] - box[1]) * cfg.fire_person_margin
            x1 = max(0, int((box[0] - pad_x) * ratio))
            y1 = max(0, int((box[1] - pad_y) * ratio))
            x2 = min(w, int((box[2] + pad_x) * ratio) + 1)
            y2 = min(h, int((box[3] + pad_y) * ratio) + 1)
            covered[y1:y2, x1:x2] = True
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        active = (mask > 0) & ~covered
        if self.frames and self.frames[0][1].shape != active.shape:
            self.frames.clear()
        self.frames.append((t, active, covered))
        while self.frames and t - self.frames[0][0] > cfg.fire_window_seconds:
            self.frames.popleft()
        if len(self.frames) < cfg.fire_min_frames:
            return []
        area = float(active.mean())
        if area < cfg.fire_min_area_ratio or area > cfg.fire_max_area_ratio:
            return []
        stack = np.stack([f[1] for f in self.frames]).astype(np.int8)
        blocked = np.stack([f[2] for f in self.frames]).any(axis=0)
        union = (stack.max(axis=0) > 0) & ~blocked
        flicker = (np.abs(np.diff(stack, axis=0)).sum(axis=0) >= 2) & ~blocked
        if flicker.sum() / max(int(union.sum()), 1) < cfg.fire_min_flicker:
            return []
        if flicker.mean() < cfg.fire_min_area_ratio * 0.5:
            return []
        ys, xs = np.nonzero(flicker)
        x1, x2 = np.percentile(xs, [2, 98])
        y1, y2 = np.percentile(ys, [2, 98])
        box = np.array([x1, y1, x2, y2], dtype=np.float64) / ratio
        return [("FIRE", box, 1.0)]
