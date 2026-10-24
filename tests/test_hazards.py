import cv2
import numpy as np

from helpers import make_config
from safework_cv.hazards import HazardMonitor

ORANGE_BGR = (0, 140, 255)


def flickering_fire_frame(rng):
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    for gy in range(4):
        for gx in range(4):
            if rng.random() < 0.5:
                y, x = 200 + gy * 40, 300 + gx * 40
                frame[y : y + 40, x : x + 40] = ORANGE_BGR
    return frame


def static_orange_wall():
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    frame[200:360, 300:460] = ORANGE_BGR
    return frame


def run(monitor, frames, people=()):
    alerts_over_time = []
    for index, frame in enumerate(frames):
        alerts_over_time.append(monitor.update(frame, index / 10.0, list(people)))
    return alerts_over_time


def test_flickering_flame_colored_region_raises_fire():
    rng = np.random.default_rng(0)
    monitor = HazardMonitor(make_config(color_fire=True))
    timeline = run(monitor, [flickering_fire_frame(rng) for _ in range(50)])
    fire_times = [
        i / 10.0 for i, alerts in enumerate(timeline) if any(a.kind == "FIRE" for a in alerts)
    ]
    assert fire_times and 1.5 < fire_times[0] < 4.5
    box = monitor.boxes[0][1]
    assert 280 <= box[0] <= 340 and 440 <= box[2] <= 480
    assert 180 <= box[1] <= 240 and 340 <= box[3] <= 380


def test_static_orange_surface_is_not_fire():
    monitor = HazardMonitor(make_config(color_fire=True))
    timeline = run(monitor, [static_orange_wall() for _ in range(60)])
    assert all(not alerts for alerts in timeline)
    assert monitor.boxes == []


def test_flicker_under_a_person_is_ignored():
    rng = np.random.default_rng(0)
    monitor = HazardMonitor(make_config(color_fire=True))
    person = [260.0, 160.0, 500.0, 400.0]
    timeline = run(monitor, [flickering_fire_frame(rng) for _ in range(50)], people=[person])
    assert all(not alerts for alerts in timeline)


def test_heuristic_can_be_disabled():
    rng = np.random.default_rng(0)
    monitor = HazardMonitor(make_config(color_fire=False))
    timeline = run(monitor, [flickering_fire_frame(rng) for _ in range(50)])
    assert all(not alerts for alerts in timeline)


def test_resolution_change_resets_the_flicker_window():
    rng = np.random.default_rng(0)
    monitor = HazardMonitor(make_config(color_fire=True))
    frames = [flickering_fire_frame(rng) for _ in range(20)]
    frames += [cv2.resize(flickering_fire_frame(rng), (960, 540)) for _ in range(40)]
    timeline = run(monitor, frames)
    assert any(a.kind == "FIRE" for alerts in timeline[40:] for a in alerts)


def test_fire_alert_expires_after_the_flames_stop():
    rng = np.random.default_rng(0)
    monitor = HazardMonitor(make_config(color_fire=True))
    frames = [flickering_fire_frame(rng) for _ in range(40)]
    frames += [np.full((720, 1280, 3), 90, dtype=np.uint8) for _ in range(30)]
    timeline = run(monitor, frames)
    assert any(a.kind == "FIRE" for a in timeline[39])
    assert not timeline[-1]
