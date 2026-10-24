import json

import cv2
import numpy as np
import pytest

from helpers import QuietHazards, ScriptedDetector, detection, lying, make_config, standing
from safework_cv.analyzer import Analyzer
from safework_cv.sinks import JsonlSink
from safework_cv.sources import FileSource
from safework_cv.zones import ZoneSet

WIDTH, HEIGHT, FPS, SECONDS = 640, 360, 10, 18


def fall_script(t, lost):
    if t < 3.0:
        box = standing(320, 300, height=100)
    elif t < 3.5:
        k = (t - 3.0) / 0.5
        h, w = 100 - 65 * k, 40 + 60 * k
        box = np.array([320 - w / 2, 300 - h, 320 + w / 2, 300])
    else:
        box = lying(320, 300, length=100, thickness=35)
    return [detection(box)]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "site.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(FPS), (WIDTH, HEIGHT)
    )
    assert writer.isOpened()
    for i in range(FPS * SECONDS):
        writer.write(np.full((HEIGHT, WIDTH, 3), 60 + i % 40, dtype=np.uint8))
    writer.release()
    return path


def build_runner(tmp_path, **kwargs):
    from safework_cv.runner import Runner

    analyzer = Analyzer(
        make_config(), ScriptedDetector(fall_script), None, QuietHazards(), ZoneSet()
    )
    sink = JsonlSink(tmp_path / "events.jsonl")
    return Runner(analyzer, sink, **kwargs), sink


def read_events(tmp_path):
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_full_pipeline_writes_events_snapshots_and_video(tmp_path, video):
    runner, sink = build_runner(
        tmp_path, snapshot_dir=tmp_path / "snaps", video_path=tmp_path / "out" / "annotated.mp4"
    )
    with FileSource(video) as source:
        processed = runner.run(source)
    sink.close()
    assert processed == FPS * SECONDS

    events = read_events(tmp_path)
    started = {e["kind"]: e for e in events if e["phase"] == "start"}
    assert {"FALL", "IMMOBILE"} <= set(started)
    assert 3.0 < started["FALL"]["video_time"] < 5.0
    assert started["FALL"]["frame"] > 30

    snapshot = started["FALL"]["snapshot"]
    assert snapshot is not None
    image = cv2.imread(snapshot)
    assert image is not None and image.shape == (HEIGHT, WIDTH, 3)
    assert all(e["snapshot"] is None for e in events if e["phase"] == "end")

    capture = cv2.VideoCapture(str(tmp_path / "out" / "annotated.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= FPS * SECONDS - 2
    capture.release()


def test_runner_without_outputs_does_not_render_or_write_files(tmp_path, video):
    runner, sink = build_runner(tmp_path)
    with FileSource(video) as source:
        runner.run(source)
    sink.close()
    assert all(e["snapshot"] is None for e in read_events(tmp_path))
    assert not (tmp_path / "snaps").exists()


def test_max_frames_limits_processing(tmp_path, video):
    runner, sink = build_runner(tmp_path)
    with FileSource(video) as source:
        assert runner.run(source, max_frames=25) == 25
    sink.close()


def test_stride_shortens_the_run_without_distorting_time(tmp_path, video):
    runner, sink = build_runner(tmp_path)
    with FileSource(video, stride=2) as source:
        processed = runner.run(source)
    sink.close()
    assert processed == FPS * SECONDS // 2
    started = {e["kind"]: e for e in read_events(tmp_path) if e["phase"] == "start"}
    assert 3.0 < started["FALL"]["video_time"] < 5.5
