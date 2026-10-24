import cv2
import numpy as np
import pytest
import yaml

from safework_cv.cli import build_parser, main
from safework_cv.config import Config
from safework_cv.zone_editor import ZoneEditor
from safework_cv.zones import ZoneSet


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_config_command_prints_loadable_defaults(capsys):
    assert main(["config"]) == 0
    data = yaml.safe_load(capsys.readouterr().out)
    assert Config.from_mapping(data) == Config()


def test_parser_requires_a_source_and_a_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    args = parser.parse_args(["run", "--source", "0", "--stride", "2", "--no-pose"])
    assert args.source == "0" and args.stride == 2 and args.no_pose


def test_missing_video_returns_error_code(capsys, tmp_path):
    assert main(["run", "--source", str(tmp_path / "nope.mp4")]) == 2


def test_unknown_config_key_returns_error_code(tmp_path, video):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run_spead: 1\n", encoding="utf-8")
    assert main(["run", "--source", str(video), "--config", bad.as_posix()]) == 2


def test_invalid_webhook_fails_before_any_model_is_loaded(video):
    assert main(["run", "--source", str(video), "--webhook", "ftp://example.com"]) == 2


def test_region_selection_is_only_for_the_screen_source(video):
    assert main(["run", "--source", str(video), "--select-region"]) == 2


def test_invalid_zones_file_returns_error_code(tmp_path, video):
    zones = tmp_path / "zones.json"
    zones.write_text("{}", encoding="utf-8")
    assert main(["run", "--source", str(video), "--zones", str(zones)]) == 2


def test_zone_editor_builds_edges_from_clicks():
    image = np.zeros((720, 1920, 3), dtype=np.uint8)
    editor = ZoneEditor(image, ZoneSet())
    assert editor.scale == pytest.approx(1280 / 1920)
    for x, y in [(0, 100), (640, 100), (1280, 200)]:
        editor._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    editor._handle_key(13)
    assert len(editor.zones.edges) == 1
    np.testing.assert_allclose(
        editor.zones.edges[0], [[0, 100 / 480], [0.5, 100 / 480], [1.0, 200 / 480]], atol=1e-6
    )
    assert editor._render().shape == (480, 1280, 3)


def test_zone_editor_undo_delete_and_clear():
    editor = ZoneEditor(np.zeros((100, 100, 3), dtype=np.uint8), ZoneSet())
    for point in [(10, 10), (50, 10), (50, 50)]:
        editor._on_mouse(cv2.EVENT_LBUTTONDOWN, *point, 0, None)
    editor._handle_key(ord("u"))
    assert len(editor.draft) == 2
    editor._handle_key(13)
    editor._handle_key(13)
    assert len(editor.zones.edges) == 1
    editor._on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    editor._on_mouse(cv2.EVENT_LBUTTONDOWN, 90, 5, 0, None)
    editor._handle_key(13)
    assert len(editor.zones.edges) == 2
    editor._handle_key(ord("d"))
    assert len(editor.zones.edges) == 1
    editor._handle_key(ord("c"))
    assert editor.zones.edges == []


def test_single_point_draft_is_discarded():
    editor = ZoneEditor(np.zeros((100, 100, 3), dtype=np.uint8), ZoneSet())
    editor._on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    editor._handle_key(13)
    assert editor.zones.edges == [] and editor.draft == []
