import numpy as np

from helpers import detection, make_config, standing
from safework_cv.tracker import Tracker


def feed(tracker, boxes_by_step, dt=0.1, **kwargs):
    t = 0.0
    out = []
    for boxes in boxes_by_step:
        out.append(tracker.update([detection(b, **kwargs) for b in boxes], t))
        t += dt
    return out


def test_track_is_confirmed_after_enough_hits_and_time():
    tracker = Tracker(make_config())
    feed(tracker, [[standing(500, 600)]] * 12)
    track = next(iter(tracker.tracks.values()))
    assert track.confirmed and track.hits == 12


def test_track_is_not_confirmed_too_early():
    tracker = Tracker(make_config())
    feed(tracker, [[standing(500, 600)]] * 5)
    assert not next(iter(tracker.tracks.values())).confirmed


def test_identity_survives_motion_and_two_people_do_not_swap():
    tracker = Tracker(make_config())
    steps = []
    for i in range(30):
        steps.append([standing(300 + 10 * i, 600), standing(900 - 10 * i, 600)])
    results = feed(tracker, steps)
    ids_left = {matches[0][0].track_id for matches in results}
    ids_right = {matches[1][0].track_id for matches in results}
    assert len(ids_left) == 1 and len(ids_right) == 1 and ids_left != ids_right


def test_low_score_detection_extends_an_existing_track_but_cannot_start_one():
    tracker = Tracker(make_config())
    feed(tracker, [[standing(500, 600)]] * 10)
    track_id = next(iter(tracker.tracks))
    matches = tracker.update([detection(standing(505, 600), score=0.2)], 1.0)
    assert matches[0][0].track_id == track_id
    fresh = Tracker(make_config())
    assert fresh.update([detection(standing(500, 600), score=0.2)], 0.0) == []
    assert fresh.tracks == {}


def test_unconfirmed_track_expires_quickly_and_confirmed_one_persists():
    tracker = Tracker(make_config())
    feed(tracker, [[standing(500, 600)]] * 12)
    tracker.update([detection(standing(100, 600), score=0.9)], 1.2)
    tracker.update([], 2.5)
    assert len(tracker.tracks) == 1
    assert next(iter(tracker.tracks.values())).confirmed


def test_rotated_detection_confirms_with_fewer_hits_and_lives_longer():
    tracker = Tracker(make_config())
    box = np.array([440.0, 530.0, 640.0, 600.0])
    tracker.update([detection(box, score=0.5, rotated=True)], 0.0)
    tracker.update([detection(box, score=0.5, rotated=True)], 1.2)
    track = next(iter(tracker.tracks.values()))
    assert track.confirmed
    young = Tracker(make_config())
    young.update([detection(box, score=0.5, rotated=True)], 0.0)
    young.update([], 3.0)
    assert len(young.tracks) == 1
    young.update([], 4.5)
    assert young.tracks == {}


def test_returning_person_is_relinked_to_the_same_track():
    tracker = Tracker(make_config())
    feed(tracker, [[standing(640, 500)]] * 15)
    track_id = next(iter(tracker.tracks))
    matches = tracker.update([detection(standing(660, 500))], 12.0)
    assert matches[0][0].track_id == track_id
