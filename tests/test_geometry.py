import numpy as np
import pytest

from safework_cv.geometry import (
    angle_between_deg,
    box_cover_fraction,
    box_iou,
    distance_to_polyline,
    projected_extents,
    unit_vector,
)


def test_unit_vector_has_length_one():
    assert np.linalg.norm(unit_vector([3, 4])) == pytest.approx(1.0)


def test_angle_between_vectors():
    assert angle_between_deg([0, -1], [0, -1]) == pytest.approx(0.0)
    assert angle_between_deg([1, 0], [0, -1]) == pytest.approx(90.0)
    assert angle_between_deg([0, 1], [0, -1]) == pytest.approx(180.0)
    assert angle_between_deg([0, 0], [0, -1]) is None


def test_box_iou_and_cover():
    a = [0, 0, 10, 10]
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, [20, 20, 30, 30]) == 0.0
    assert box_iou(a, [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_cover_fraction(a, [0, 0, 20, 20]) == pytest.approx(1.0)
    assert box_cover_fraction(a, [5, 0, 15, 10]) == pytest.approx(0.5)


def test_projected_extents_follow_camera_up():
    tall = [[0, 0], [10, 0], [10, 40], [0, 40]]
    up = unit_vector([0, -1])
    along, across = projected_extents(tall, up)
    assert along == pytest.approx(40.0) and across == pytest.approx(10.0)
    tilted = unit_vector([1, 0])
    along, across = projected_extents(tall, tilted)
    assert along == pytest.approx(10.0) and across == pytest.approx(40.0)


def test_distance_to_polyline():
    line = [[0, 0], [100, 0], [100, 100]]
    assert distance_to_polyline([[50, 10]], line) == pytest.approx(10.0)
    assert distance_to_polyline([[110, 50]], line) == pytest.approx(10.0)
    assert distance_to_polyline([[-30, 0], [50, 3]], line) == pytest.approx(3.0)
    assert distance_to_polyline([[103, 104]], line) == pytest.approx(np.hypot(3, 4))
