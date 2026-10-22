import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from config import Config
from geometry import box_center, box_iou, box_scale


@dataclass
class Detection:
    box: np.ndarray
    score: float
    rotated: bool = False
    keypoints: np.ndarray | None = None
    visibility: np.ndarray | None = None


class PersonDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = YOLO(cfg.yolo_weights)
        self.last_scan = None
        self.last_recovery = {}

    def detect(self, frame, t, lost):
        boxes, scores = self._infer([frame], self.cfg.yolo_conf)[0]
        detections = [Detection(b, float(s)) for b, s in zip(boxes, scores)]
        for region in self._regions(frame.shape, t, lost):
            for det in self._rotated(frame, region):
                if all(box_iou(det.box, d.box) < self.cfg.rotated_dedup_iou for d in detections):
                    detections.append(det)
        return detections

    def _infer(self, images, conf):
        results = self.model(
            images,
            classes=[0],
            conf=conf,
            imgsz=self.cfg.yolo_imgsz,
            iou=self.cfg.yolo_nms_iou,
            verbose=False,
        )
        out = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                out.append((np.zeros((0, 4)), np.zeros(0)))
                continue
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float64)
            out.append((xyxy, result.boxes.conf.cpu().numpy()))
        return out

    def _regions(self, shape, t, lost):
        cfg = self.cfg
        h, w = shape[:2]
        regions = []
        if self.last_scan is None or t - self.last_scan >= cfg.rotated_scan_seconds:
            self.last_scan = t
            regions.append((0, 0, w, h))
        pending = sorted(lost, key=lambda item: self.last_recovery.get(item[0], -1.0))
        for track_id, box in pending:
            if len(regions) >= cfg.max_rotated_regions:
                break
            last = self.last_recovery.get(track_id)
            if last is not None and t - last < cfg.recovery_interval_seconds:
                continue
            self.last_recovery[track_id] = t
            cx, cy = box_center(box)
            half = max(box_scale(box) * cfg.recovery_crop_ratio / 2, 48.0)
            regions.append(
                (
                    int(max(0, cx - half)),
                    int(max(0, cy - half)),
                    int(min(w, cx + half)),
                    int(min(h, cy + half)),
                )
            )
        return regions

    def _rotated(self, frame, region):
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h < 64 or w < 64:
            return []
        ccw = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cw = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        (boxes_ccw, scores_ccw), (boxes_cw, scores_cw) = self._infer(
            [ccw, cw], self.cfg.rotated_conf
        )
        offset = np.array([x1, y1, x1, y1], dtype=np.float64)
        mapped = []
        for nb, score in zip(boxes_ccw, scores_ccw):
            mapped.append((np.array([w - nb[3], nb[0], w - nb[1], nb[2]]), score))
        for nb, score in zip(boxes_cw, scores_cw):
            mapped.append((np.array([nb[1], h - nb[2], nb[3], h - nb[0]]), score))
        out = []
        for box, score in mapped:
            if (box[2] - box[0]) < self.cfg.rotated_min_aspect * (box[3] - box[1]):
                continue
            out.append(Detection(box + offset, float(score), rotated=True))
        return out


class PoseEstimator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.landmarker = self._build()

    def _build(self):
        path = self.cfg.pose_model_path
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            urllib.request.urlretrieve(self.cfg.pose_model_url, path)
        conf = self.cfg.pose_confidence
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=conf,
            min_pose_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame_rgb, box):
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * self.cfg.crop_padding
        pad_y = (y2 - y1) * self.cfg.crop_padding
        cx1 = int(max(0, x1 - pad_x))
        cy1 = int(max(0, y1 - pad_y))
        cx2 = int(min(w, x2 + pad_x))
        cy2 = int(min(h, y2 + pad_y))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return None, None
        crop = np.ascontiguousarray(frame_rgb[cy1:cy2, cx1:cx2])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = self.landmarker.detect(image)
        if not result.pose_landmarks:
            return None, None
        landmarks = result.pose_landmarks[0]
        ch, cw = crop.shape[:2]
        points = np.array(
            [[lm.x * cw + cx1, lm.y * ch + cy1] for lm in landmarks], dtype=np.float64
        )
        visibility = np.array(
            [lm.visibility if lm.visibility is not None else 0.0 for lm in landmarks],
            dtype=np.float64,
        )
        return points, visibility