import argparse
import time

import cv2
import numpy as np

from alerts import AlertLog
from capture import ScreenCapture
from config import Config
from edges import EdgeSet
from hazards import HazardMonitor
from overlay import render
from perception import PersonDetector, PoseEstimator
from rules import RuleEngine
from state import PersonRegistry

WINDOW = "safety monitor"


def parse_region(text):
    if not text:
        return None
    left, top, width, height = (int(v) for v in text.split(","))
    return {"left": left, "top": top, "width": width, "height": height}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--region", type=str, default="")
    parser.add_argument("--select-region", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--hazard-weights", type=str, default="")
    parser.add_argument("--no-color-fire", action="store_true")
    return parser.parse_args()


def handle_key(key, edges):
    if key == ord("q"):
        return False
    if key == ord("e"):
        edges.toggle_edit()
    elif key == 13:
        edges.finish()
    elif key == ord("u"):
        edges.undo()
    elif key == ord("c"):
        edges.clear()
    return True


def main():
    args = parse_args()
    cfg = Config(
        display_scale=args.scale,
        hazard_weights=args.hazard_weights,
        color_fire=not args.no_color_fire,
    )
    capture = ScreenCapture(args.monitor, parse_region(args.region))
    if args.select_region:
        capture.pick_region()
    detector = PersonDetector(cfg)
    pose = PoseEstimator(cfg)
    hazards = HazardMonitor(cfg)
    edges = EdgeSet(cfg.edges_path, capture.region, cfg.display_scale)
    registry = PersonRegistry(cfg, pose)
    rules = RuleEngine(cfg, edges)
    log = AlertLog()
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, edges.on_mouse)
    fps = 0.0
    previous = None
    try:
        while True:
            frame = capture.grab()
            t = time.monotonic()
            if previous is not None:
                instant = 1.0 / max(t - previous, 1e-3)
                fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
            previous = t
            height, width = frame.shape[:2]
            frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
            detections = detector.detect(frame, t, registry.lost_boxes(t))
            people = [
                d.box for d in detections if d.rotated or d.score >= cfg.new_track_score
            ]
            hazard_alerts = hazards.update(frame, t, people)
            registry.update(frame_rgb, detections, t, width, height)
            states = list(registry.states.values())
            alerts = sorted(hazard_alerts + rules.evaluate(states, t), key=lambda a: a.rank)
            log.update(alerts)
            canvas = render(frame, states, alerts, hazards.boxes, edges, t, fps, cfg)
            cv2.imshow(WINDOW, canvas)
            if not handle_key(cv2.waitKey(1) & 0xFF, edges):
                break
    finally:
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()