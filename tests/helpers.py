from __future__ import annotations

import numpy as np

from safework_cv.analyzer import Analyzer
from safework_cv.config import Config
from safework_cv.perception import Detection
from safework_cv.sources import Frame
from safework_cv.zones import ZoneSet

FRAME_W = 1280
FRAME_H = 720


def make_config(**overrides) -> Config:
    base = {"enable_pose": False, "color_fire": False}
    base.update(overrides)
    return Config(**base)


def standing(cx, foot_y, height=200.0, aspect=0.4):
    width = height * aspect
    return np.array([cx - width / 2, foot_y - height, cx + width / 2, foot_y], dtype=np.float64)


def lying(cx, ground_y, length=200.0, thickness=70.0):
    return np.array(
        [cx - length / 2, ground_y - thickness, cx + length / 2, ground_y], dtype=np.float64
    )


def detection(box, score=0.9, rotated=False):
    return Detection(np.asarray(box, dtype=np.float64), score, rotated)


class ScriptedDetector:
    def __init__(self, script):
        self.script = script
        self.lost_requests = []

    def detect(self, image, t, lost):
        if lost:
            self.lost_requests.append((t, [track_id for track_id, _ in lost]))
        return self.script(t, lost)


class QuietHazards:
    boxes: list = []

    def update(self, frame, t, person_boxes):
        return []


class FakePose:
    def __init__(self, lying_pose=lambda box: False):
        self.lying_pose = lying_pose
        self.calls = 0

    def estimate(self, frame_bgr, box):
        self.calls += 1
        return skeleton(box, self.lying_pose(box)), np.full(33, 0.9)


def skeleton(box, horizontal):
    x1, y1, x2, y2 = box
    points = np.tile([(x1 + x2) / 2, (y1 + y2) / 2], (33, 1)).astype(np.float64)
    w = x2 - x1
    h = y2 - y1
    if not horizontal:
        cx = (x1 + x2) / 2
        points[11] = [cx - 0.2 * w, y1 + 0.2 * h]
        points[12] = [cx + 0.2 * w, y1 + 0.2 * h]
        points[23] = [cx - 0.15 * w, y1 + 0.55 * h]
        points[24] = [cx + 0.15 * w, y1 + 0.55 * h]
        points[25:27] = [[cx, y1 + 0.78 * h]]
        points[27:33] = [[cx, y2]]
    else:
        cy = (y1 + y2) / 2
        points[11] = [x1 + 0.2 * w, cy - 0.2 * h]
        points[12] = [x1 + 0.2 * w, cy + 0.2 * h]
        points[23] = [x1 + 0.55 * w, cy - 0.15 * h]
        points[24] = [x1 + 0.55 * w, cy + 0.15 * h]
        points[25:27] = [[x1 + 0.78 * w, cy]]
        points[27:33] = [[x2, cy]]
    return points


def make_analyzer(script, cfg=None, zones=None, pose=None):
    detector = ScriptedDetector(script)
    analyzer = Analyzer(cfg or make_config(), detector, pose, QuietHazards(), zones or ZoneSet())
    return analyzer, detector


def run_scenario(analyzer, seconds, fps=10.0):
    image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    results = []
    for index in range(int(seconds * fps)):
        results.append(analyzer.process(Frame(image, index / fps, index)))
    return results


def kinds(result):
    return {alert.kind for alert in result.alerts}


def all_kinds(results):
    out = set()
    for result in results:
        out |= kinds(result)
    return out


def first_time(results, kind):
    return next((r.stream_time for r in results if kind in kinds(r)), None)
