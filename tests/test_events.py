import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from safework_cv.alerts import Alert, Event, EventTracker
from safework_cv.sinks import JsonlSink, LogSink, MultiSink, WebhookSink


def make_event(phase="start", kind="FALL", track_id=3, duration=None):
    return Event(
        phase=phase,
        kind=kind,
        track_id=track_id,
        text="sudden fall",
        point=(10.04, 20.0),
        video_time=1.23456,
        wall_time="2026-01-01T00:00:00.000+00:00",
        duration=duration,
        frame=12,
    )


def test_alert_label_and_priority_order():
    assert Alert("FALL", 4, (0, 0), "x").label == "FALL #4"
    assert Alert("FIRE", None, (0, 0), "x").label == "FIRE"
    assert Alert("FIRE", None, (0, 0), "x").rank < Alert("FALL", 1, (0, 0), "x").rank
    assert Alert("FALL", 1, (0, 0), "x").rank < Alert("OCCLUDED", 1, (0, 0), "x").rank


def test_event_serialization_is_stable_and_rounded():
    data = make_event().to_dict()
    assert data == {
        "phase": "start",
        "kind": "FALL",
        "track_id": 3,
        "text": "sudden fall",
        "point": [10.0, 20.0],
        "video_time": 1.235,
        "wall_time": "2026-01-01T00:00:00.000+00:00",
        "duration": None,
        "frame": 12,
        "snapshot": None,
    }
    json.dumps(data)


def test_event_lifecycle_has_start_hysteresis_and_end():
    tracker = EventTracker(release_seconds=2.0)
    alert = Alert("FALL", 1, (1.0, 2.0), "x")
    start = tracker.update([alert], 0.0)
    assert [(e.phase, e.kind) for e in start] == [("start", "FALL")]
    assert tracker.update([alert], 1.0) == []
    assert tracker.update([], 2.0) == []
    assert tracker.update([alert], 2.5) == []
    assert tracker.update([], 4.0) == []
    ended = tracker.update([], 4.6)
    assert [(e.phase, e.kind) for e in ended] == [("end", "FALL")]
    assert ended[0].duration == pytest.approx(4.6)


def test_alerts_are_tracked_per_kind_and_person():
    tracker = EventTracker(release_seconds=1.0)
    events = tracker.update(
        [
            Alert("FALL", 1, (0, 0), "x"),
            Alert("FALL", 2, (0, 0), "x"),
            Alert("FIRE", None, (0, 0), "x"),
        ],
        0.0,
    )
    assert len(events) == 3
    assert tracker.update([Alert("FALL", 1, (0, 0), "x")], 0.5) == []


def test_jsonl_sink_appends_one_object_per_line(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    sink = JsonlSink(path)
    sink.emit(make_event())
    sink.emit(make_event(phase="end", duration=4.0))
    sink.close()
    again = JsonlSink(path)
    again.emit(make_event(kind="FIRE", track_id=None))
    again.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["FALL", "FALL", "FIRE"]
    assert rows[1]["duration"] == 4.0


def test_log_sink_reports_start_and_end(caplog):
    sink = LogSink()
    with caplog.at_level(logging.INFO, logger="safework_cv.sinks"):
        sink.emit(make_event())
        sink.emit(make_event(phase="end", duration=3.0))
    levels = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert levels[0][0] == "WARNING" and "FALL #3" in levels[0][1]
    assert levels[1][0] == "INFO" and "ended after 3.0s" in levels[1][1]


def test_webhook_sink_posts_json():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append((self.headers["Content-Type"], json.loads(self.rfile.read(length))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        sink = WebhookSink(f"http://127.0.0.1:{server.server_port}/hook")
        sink.emit(make_event())
        sink.close()
    finally:
        server.shutdown()
        server.server_close()
    assert received == [("application/json", make_event().to_dict())]


def test_webhook_failure_is_logged_not_raised(caplog):
    with caplog.at_level(logging.WARNING, logger="safework_cv.sinks"):
        sink = WebhookSink("http://127.0.0.1:9/hook", timeout=0.5)
        sink.emit(make_event())
        sink.close()
    assert any("webhook delivery failed" in r.getMessage() for r in caplog.records)


def test_webhook_rejects_non_http_schemes():
    for url in ["file:///etc/passwd", "ftp://example.com/x", "example.com"]:
        with pytest.raises(ValueError):
            WebhookSink(url)


def test_multi_sink_fans_out_and_closes_everything():
    calls = []

    class Recorder:
        def emit(self, event):
            calls.append(("emit", event.kind))

        def close(self):
            calls.append(("close",))

    sink = MultiSink([Recorder(), Recorder()])
    sink.emit(make_event())
    sink.close()
    assert calls == [("emit", "FALL"), ("emit", "FALL"), ("close",), ("close",)]
