import json

import numpy as np
import pytest

from safework_cv.zones import ZoneSet


def test_edges_are_stored_normalized_and_scale_with_resolution():
    zones = ZoneSet()
    zones.add_edge([(100, 200), (500, 200)], 1000, 400)
    np.testing.assert_allclose(zones.edges[0], [[0.1, 0.5], [0.5, 0.5]])
    low = zones.edges_px(500, 200)[0]
    high = zones.edges_px(2000, 800)[0]
    np.testing.assert_allclose(low, [[50, 100], [250, 100]])
    np.testing.assert_allclose(high, [[200, 400], [1000, 400]])


def test_save_and_load_roundtrip(tmp_path):
    zones = ZoneSet()
    zones.add_edge([(10, 10), (90, 10), (90, 50)], 100, 100)
    path = tmp_path / "nested" / "zones.json"
    zones.save(path)
    loaded = ZoneSet.load(path)
    assert len(loaded.edges) == 1
    np.testing.assert_allclose(loaded.edges[0], zones.edges[0])
    assert not list(path.parent.glob("*.part"))


def test_edge_needs_two_points():
    with pytest.raises(ValueError):
        ZoneSet().add_edge([(1, 1)], 10, 10)


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "edges": []},
        {"version": 1, "edges": [[[0.1, 0.1]]]},
        {"version": 1, "edges": [[[0.1, 0.1], [1.5, 0.2]]]},
        [],
    ],
)
def test_invalid_files_are_rejected(tmp_path, payload):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        ZoneSet.load(path)


def test_clear_removes_everything():
    zones = ZoneSet()
    zones.add_edge([(0, 0), (1, 1)], 10, 10)
    zones.clear()
    assert zones.edges == [] and zones.edges_px(100, 100) == []
