import math

import numpy as np

from helpers import (
    FakePose,
    all_kinds,
    detection,
    first_time,
    kinds,
    lying,
    make_analyzer,
    make_config,
    run_scenario,
    standing,
)
from safework_cv.zones import ZoneSet


def walking(t, lost):
    cx = 640 + 300 * math.sin(0.5 * t)
    return [detection(standing(cx, 600))]


def running(t, lost):
    if t > 2.8:
        return []
    return [detection(standing(100 + 400 * t, 600))]


def falling(t, lost):
    if t < 3.0:
        box = standing(640, 600)
    elif t < 3.5:
        k = (t - 3.0) / 0.5
        h = 200 - 130 * k
        w = 80 + 120 * k
        box = np.array([640 - w / 2, 600 - h, 640 + w / 2, 600])
    else:
        box = lying(640, 600)
    return [detection(box)]


def sitting_down(t, lost):
    k = min(max((t - 3.0) / 6.0, 0.0), 1.0)
    h = 200 - 130 * k
    w = 80 + 120 * k
    return [detection(np.array([640 - w / 2, 600 - h, 640 + w / 2, 600]))]


def test_walking_worker_raises_no_alerts():
    analyzer, _ = make_analyzer(walking)
    results = run_scenario(analyzer, 25)
    assert all_kinds(results) == set()
    assert any(s.confirmed for s in results[-1].states)


def test_running_is_reported():
    analyzer, _ = make_analyzer(running)
    results = run_scenario(analyzer, 5)
    assert first_time(results, "RUNNING") is not None
    assert first_time(results, "RUNNING") < 3.5


def test_fall_then_immobility():
    analyzer, _ = make_analyzer(falling)
    results = run_scenario(analyzer, 18)
    fall_at = first_time(results, "FALL")
    immobile_at = first_time(results, "IMMOBILE")
    assert fall_at is not None and 3.0 < fall_at < 5.0
    assert immobile_at is not None and 10.5 < immobile_at < 14.0
    assert "RUNNING" not in all_kinds(results)


def test_fall_is_detected_on_a_slow_five_fps_stream():
    analyzer, _ = make_analyzer(falling)
    results = run_scenario(analyzer, 18, fps=5)
    assert first_time(results, "FALL") is not None
    assert first_time(results, "IMMOBILE") is not None


def test_slowly_lying_down_is_not_a_fall():
    analyzer, _ = make_analyzer(sitting_down)
    results = run_scenario(analyzer, 20)
    assert "FALL" not in all_kinds(results)
    assert first_time(results, "IMMOBILE") is not None


def test_lost_worker_is_flagged_after_timeout_and_cleared_on_return():
    def script(t, lost):
        if t < 3.0 or t >= 18.0:
            return [detection(standing(640, 500))]
        return []

    analyzer, _ = make_analyzer(script)
    results = run_scenario(analyzer, 24)
    occluded_at = first_time(results, "OCCLUDED")
    assert occluded_at is not None and 14.5 < occluded_at < 16.5
    flagged = [r for r in results if "OCCLUDED" in kinds(r)]
    point = flagged[0].alerts[0].point
    assert abs(point[0] - 640) < 5 and abs(point[1] - 400) < 5
    assert "OCCLUDED" not in kinds(results[-1])
    phases = [(e.phase, e.kind) for r in results for e in r.events if e.kind == "OCCLUDED"]
    assert phases == [("start", "OCCLUDED"), ("end", "OCCLUDED")]


def test_worker_leaving_through_frame_edge_is_not_lost():
    def script(t, lost):
        cx = 640 + 150 * t
        if cx - 40 >= 1280:
            return []
        x2 = min(cx + 40, 1280)
        return [detection([cx - 40, 400, x2, 600])]

    analyzer, _ = make_analyzer(script)
    results = run_scenario(analyzer, 30)
    assert "OCCLUDED" not in all_kinds(results)
    assert not results[-1].states


def test_duplicate_ghost_track_does_not_become_a_lost_worker():
    def script(t, lost):
        real = detection(standing(640, 600))
        if t < 3.0:
            return [real, detection(standing(665, 610), score=0.8)]
        return [real]

    analyzer, _ = make_analyzer(script)
    results = run_scenario(analyzer, 25)
    assert "OCCLUDED" not in all_kinds(results)
    assert len(results[-1].states) == 1


def test_low_score_noise_never_creates_a_person():
    analyzer, _ = make_analyzer(lambda t, lost: [detection(standing(640, 600), score=0.3)])
    results = run_scenario(analyzer, 10)
    assert all(not r.states for r in results)


def test_edge_alert_depends_on_distance_to_the_edge():
    zones = ZoneSet()
    zones.add_edge([(0, 640), (1280, 640)], 1280, 720)
    near, _ = make_analyzer(lambda t, lost: [detection(standing(640, 600))], zones=zones)
    far, _ = make_analyzer(lambda t, lost: [detection(standing(640, 300))], zones=zones)
    near_results = run_scenario(near, 4)
    far_results = run_scenario(far, 4)
    assert first_time(near_results, "EDGE") is not None
    assert "EDGE" not in all_kinds(far_results)


def test_no_zones_means_no_edge_alerts():
    analyzer, _ = make_analyzer(lambda t, lost: [detection(standing(640, 715))])
    assert "EDGE" not in all_kinds(run_scenario(analyzer, 4))


def test_lying_worker_seen_only_by_rotated_pass_is_still_tracked():
    state = {"last": -10.0}

    def script(t, lost):
        due = bool(lost) and t - state["last"] >= 0.6
        scan = int(round(t * 10)) % 12 == 0
        if due or scan:
            state["last"] = t
            return [detection(lying(640, 600), score=0.6, rotated=True)]
        return []

    analyzer, detector = make_analyzer(script)
    results = run_scenario(analyzer, 20)
    assert first_time(results, "IMMOBILE") is not None
    assert "OCCLUDED" not in all_kinds(results)
    assert detector.lost_requests


def test_skeleton_path_runs_and_keeps_calm_for_walking_worker():
    pose = FakePose()
    analyzer, _ = make_analyzer(walking, cfg=make_config(enable_pose=True), pose=pose)
    results = run_scenario(analyzer, 15)
    assert pose.calls > 0
    assert all_kinds(results) == set()
    state = results[-1].states[0]
    assert state.pose_fresh and state.keypoints is not None


def test_skeleton_confirms_lying_posture():
    def script(t, lost):
        return [detection(np.array([540.0, 480.0, 740.0, 600.0]))]

    pose = FakePose(lying_pose=lambda box: True)
    analyzer, _ = make_analyzer(script, cfg=make_config(enable_pose=True), pose=pose)
    results = run_scenario(analyzer, 12)
    state = results[-1].states[0]
    assert state.posture.angle is not None and state.posture.angle > 60
    assert state.lying
    assert first_time(results, "IMMOBILE") is not None
