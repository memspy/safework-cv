import numpy as np
import pytest

from helpers import lying, make_config, skeleton, standing
from safework_cv.state import classify_posture

VISIBLE = np.full(33, 0.9)


@pytest.mark.parametrize(
    ("box", "horizontal", "upright", "is_lying"),
    [
        (standing(500, 600), False, True, False),
        (lying(500, 600), True, False, True),
        (standing(500, 600), None, True, False),
        (lying(500, 600), None, False, True),
    ],
)
def test_posture_from_box_and_skeleton(box, horizontal, upright, is_lying):
    keypoints = None if horizontal is None else skeleton(box, horizontal)
    posture = classify_posture(make_config(), keypoints, VISIBLE, box)
    assert posture.upright is upright
    assert posture.lying is is_lying


def test_crouching_worker_is_neither_upright_nor_lying():
    box = np.array([450.0, 500.0, 550.0, 600.0])
    posture = classify_posture(make_config(), skeleton(box, False), VISIBLE, box)
    assert not posture.lying and not posture.upright


def test_squarish_box_with_horizontal_torso_counts_as_lying():
    box = np.array([450.0, 500.0, 550.0, 590.0])
    posture = classify_posture(make_config(), skeleton(box, True), VISIBLE, box)
    assert posture.angle > 60
    assert posture.lying


def test_tilted_camera_up_vector_rotates_the_meaning_of_lying():
    cfg = make_config(camera_up=(1.0, 0.0))
    box = lying(500, 600)
    posture = classify_posture(cfg, None, VISIBLE, box)
    assert posture.upright and not posture.lying
