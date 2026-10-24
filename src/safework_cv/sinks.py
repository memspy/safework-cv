from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .alerts import Event

log = logging.getLogger(__name__)


class Sink:
    def emit(self, event: Event) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LogSink(Sink):
    def emit(self, event: Event) -> None:
        label = event.kind if event.track_id is None else f"{event.kind} #{event.track_id}"
        if event.phase == "start":
            log.warning("%s: %s (t=%.1fs)", label, event.text, event.video_time)
        else:
            log.info("%s ended after %.1fs", label, event.duration or 0.0)


class JsonlSink(Sink):
    def __init__(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(target, "a", encoding="utf-8")

    def emit(self, event: Event) -> None:
        self._file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class WebhookSink(Sink):
    def __init__(self, url: str, timeout: float = 5.0, queue_size: int = 100):
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("webhook url must start with http:// or https://")
        self.url = url
        self.timeout = timeout
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="webhook-sink", daemon=True)
        self._thread.start()

    def emit(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event.to_dict())
        except queue.Full:
            log.warning("webhook queue is full, dropping %s event", event.kind)

    def _run(self) -> None:
        while not (self._stop.is_set() and self._queue.empty()):
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._post(payload)

    def _post(self, payload: dict) -> None:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
        except (urllib.error.URLError, OSError) as exc:
            log.warning("webhook delivery failed: %s", exc)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)


class MultiSink(Sink):
    def __init__(self, sinks: list[Sink]):
        self.sinks = sinks

    def emit(self, event: Event) -> None:
        for sink in self.sinks:
            sink.emit(event)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
