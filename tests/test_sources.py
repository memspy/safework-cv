import time

import cv2
import numpy as np
import pytest

from safework_cv import sources
from safework_cv.sources import (
    FileSource,
    SourceError,
    StreamSource,
    open_source,
    parse_region,
    redact,
)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


def test_file_source_timestamps_follow_the_video_clock(video):
    with FileSource(video) as source:
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)
    assert len(frames) == 30
    assert [f.index for f in frames[:3]] == [0, 1, 2]
    assert frames[10].timestamp == pytest.approx(0.5)
    assert frames[0].image.shape == (48, 64, 3)
    assert not source.live
    assert source.fps == pytest.approx(20.0)


def test_file_source_stride_keeps_every_nth_frame(video):
    with FileSource(video, stride=3) as source:
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)
    assert [f.index for f in frames] == list(range(0, 30, 3))
    assert frames[1].timestamp == pytest.approx(3 / 20.0)
    assert source.fps == pytest.approx(20.0 / 3)


def test_file_source_rejects_bad_input(tmp_path):
    with pytest.raises(SourceError):
        FileSource(tmp_path / "missing.avi")
    with pytest.raises(ValueError):
        FileSource(tmp_path / "missing.avi", stride=0)


def test_stream_source_delivers_the_latest_frame_and_stops(video):
    source = StreamSource(str(video), reconnect_seconds=0.05)
    try:
        frame = source.read()
        assert frame is not None and frame.image.shape == (48, 64, 3)
        assert source.live
    finally:
        source.close()
    assert not source._thread.is_alive()


def test_stream_source_survives_an_unavailable_stream(tmp_path):
    source = StreamSource(str(tmp_path / "nothing.avi"), reconnect_seconds=0.05)
    time.sleep(0.3)
    assert source._thread.is_alive()
    source.close()
    assert not source._thread.is_alive()
    assert source.read() is None


def test_open_source_routes_by_spec(monkeypatch, video):
    created = []
    monkeypatch.setattr(
        sources,
        "StreamSource",
        lambda target, reconnect: created.append((target, reconnect)) or "stream",
    )
    assert open_source("0", reconnect_seconds=7.0) == "stream"
    assert open_source("rtsp://cam.local/live") == "stream"
    assert open_source("http://cam.local/video.mjpg") == "stream"
    assert created[0] == (0, 7.0)
    assert created[1][0] == "rtsp://cam.local/live"
    assert isinstance(open_source(str(video)), FileSource)


def test_open_source_errors(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        open_source(str(tmp_path / "nope.mp4"))
    with pytest.raises(SourceError):
        open_source("screen:abc")
    with pytest.raises(SourceError):
        open_source("screen:99")


def test_parse_region():
    assert parse_region("10,20,300,200") == {"left": 10, "top": 20, "width": 300, "height": 200}
    for bad in ["1,2,3", "a,b,c,d", "0,0,0,10"]:
        with pytest.raises(SourceError):
            parse_region(bad)


def test_credentials_are_redacted_from_stream_urls():
    assert redact("rtsp://admin:s3cret@10.0.0.5:554/live") == "rtsp://***@10.0.0.5:554/live"
    assert redact("rtsp://10.0.0.5/live") == "rtsp://10.0.0.5/live"
    assert redact(0) == "0"
