from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_LIVE_FPS = 10.0


class SourceError(RuntimeError):
    pass


def redact(target) -> str:
    return re.sub(r"(?<=//)[^/@]*@", "***@", str(target))


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp: float
    index: int


class FrameSource:
    live = False
    fps = 0.0

    def read(self) -> Frame | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FileSource(FrameSource):
    def __init__(self, path: str | Path, stride: int = 1):
        if stride < 1:
            raise ValueError("stride must be at least 1")
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise SourceError(f"cannot open video file: {path}")
        native = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.native_fps = native if native > 1.0 else 25.0
        self.fps = self.native_fps / stride
        self.stride = stride
        self.index = -1

    def read(self) -> Frame | None:
        ok, image = self.capture.read()
        if not ok:
            return None
        self.index += 1
        frame = Frame(image, self.index / self.native_fps, self.index)
        for _ in range(self.stride - 1):
            if not self.capture.grab():
                break
            self.index += 1
        return frame

    def close(self) -> None:
        self.capture.release()


class StreamSource(FrameSource):
    live = True

    def __init__(self, target: str | int, reconnect_seconds: float = 3.0):
        if isinstance(target, str) and target.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.target = target
        self.reconnect_seconds = reconnect_seconds
        self.fps = DEFAULT_LIVE_FPS
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest: tuple[np.ndarray, float] | None = None
        self._produced = 0
        self._consumed = 0
        self._thread = threading.Thread(target=self._run, name="stream-reader", daemon=True)
        self._thread.start()

    def _open(self):
        capture = cv2.VideoCapture(self.target)
        if not capture.isOpened():
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self):
        capture = None
        while not self._stop.is_set():
            if capture is None:
                capture = self._open()
                if capture is None:
                    log.warning(
                        "cannot open %s, retrying in %.0fs",
                        redact(self.target),
                        self.reconnect_seconds,
                    )
                    self._stop.wait(self.reconnect_seconds)
                    continue
                log.info("stream connected: %s", redact(self.target))
            ok, image = capture.read()
            if not ok:
                log.warning("stream interrupted, reconnecting in %.0fs", self.reconnect_seconds)
                capture.release()
                capture = None
                self._stop.wait(self.reconnect_seconds)
                continue
            with self._condition:
                self._latest = (image, time.monotonic())
                self._produced += 1
                self._condition.notify_all()
        if capture is not None:
            capture.release()

    def read(self) -> Frame | None:
        with self._condition:
            while self._produced == self._consumed and not self._stop.is_set():
                self._condition.wait(timeout=1.0)
            if self._stop.is_set() or self._latest is None:
                return None
            self._consumed = self._produced
            image, timestamp = self._latest
            return Frame(image, timestamp, self._consumed)

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=5.0)


class ScreenSource(FrameSource):
    live = True

    def __init__(self, monitor: int = 1, region: dict | None = None):
        try:
            import mss
        except ImportError as exc:
            raise SourceError(
                "screen capture requires the 'screen' extra: pip install 'safework-cv[screen]'"
            ) from exc
        try:
            factory = getattr(mss, "MSS", None) or mss.mss
            self._sct = factory()
        except Exception as exc:
            raise SourceError(f"screen capture is not available: {exc}") from exc
        monitors = self._sct.monitors
        if not 0 <= monitor < len(monitors):
            self._sct.close()
            raise SourceError(f"monitor {monitor} not found, available: 0..{len(monitors) - 1}")
        self.monitor = dict(monitors[monitor])
        self.region = region or dict(self.monitor)
        self.fps = DEFAULT_LIVE_FPS
        self._index = 0

    def read(self) -> Frame | None:
        shot = self._sct.grab(self.region)
        self._index += 1
        image = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
        return Frame(image, time.monotonic(), self._index)

    def pick_region(self, window: str = "select region") -> None:
        full = np.ascontiguousarray(np.asarray(self._sct.grab(self.monitor))[:, :, :3])
        ratio = min(1.0, 1280 / full.shape[1])
        preview = cv2.resize(full, None, fx=ratio, fy=ratio) if ratio < 1.0 else full
        x, y, w, h = cv2.selectROI(window, preview, showCrosshair=True)
        cv2.destroyWindow(window)
        if w == 0 or h == 0:
            return
        self.region = {
            "left": self.monitor["left"] + int(x / ratio),
            "top": self.monitor["top"] + int(y / ratio),
            "width": int(w / ratio),
            "height": int(h / ratio),
        }

    def close(self) -> None:
        self._sct.close()


def parse_region(text: str) -> dict:
    try:
        left, top, width, height = (int(v) for v in text.split(","))
    except ValueError as exc:
        raise SourceError("region must be 'left,top,width,height'") from exc
    if width <= 0 or height <= 0:
        raise SourceError("region width and height must be positive")
    return {"left": left, "top": top, "width": width, "height": height}


def open_source(
    spec: str,
    *,
    stride: int = 1,
    region: dict | None = None,
    reconnect_seconds: float = 3.0,
) -> FrameSource:
    text = spec.strip()
    if text.isdigit():
        return StreamSource(int(text), reconnect_seconds)
    if text == "screen" or text.startswith("screen:"):
        monitor = 1
        if ":" in text:
            try:
                monitor = int(text.split(":", 1)[1])
            except ValueError as exc:
                raise SourceError("screen source must be 'screen' or 'screen:<monitor>'") from exc
        return ScreenSource(monitor, region)
    if "://" in text:
        return StreamSource(text, reconnect_seconds)
    if not Path(text).is_file():
        raise SourceError(f"video file not found: {text}")
    return FileSource(text, stride)
