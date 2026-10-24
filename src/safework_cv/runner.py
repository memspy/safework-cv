from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

import cv2

from .alerts import Event
from .analyzer import Analyzer
from .overlay import render
from .sinks import Sink
from .sources import FrameSource

log = logging.getLogger(__name__)

WINDOW = "safework-cv"


class FpsMeter:
    def __init__(self):
        self.value = 0.0
        self._previous = None

    def tick(self, now: float) -> float:
        if self._previous is not None:
            instant = 1.0 / max(now - self._previous, 1e-3)
            self.value = instant if self.value == 0.0 else 0.9 * self.value + 0.1 * instant
        self._previous = now
        return self.value


class Runner:
    def __init__(
        self,
        analyzer: Analyzer,
        sink: Sink,
        *,
        display: bool = False,
        display_scale: float = 1.0,
        snapshot_dir: Path | None = None,
        video_path: Path | None = None,
    ):
        self.analyzer = analyzer
        self.sink = sink
        self.display = display
        self.display_scale = display_scale
        self.snapshot_dir = snapshot_dir
        self.video_path = video_path
        self._writer = None

    def run(self, source: FrameSource, max_frames: int | None = None) -> int:
        meter = FpsMeter()
        processed = 0
        try:
            while max_frames is None or processed < max_frames:
                frame = source.read()
                if frame is None:
                    break
                result = self.analyzer.process(frame)
                fps = meter.tick(time.monotonic())
                canvas = self._canvas(result, fps)
                for event in result.events:
                    self.sink.emit(self._with_snapshot(event, canvas))
                if canvas is not None and self.video_path:
                    self._write_video(canvas, source.fps)
                processed += 1
                if canvas is not None and self.display and not self._show(canvas):
                    break
        finally:
            self._close()
        return processed

    def _canvas(self, result, fps):
        wants_snapshot = self.snapshot_dir is not None and any(
            event.phase == "start" for event in result.events
        )
        if not (self.display or self.video_path or wants_snapshot):
            return None
        return render(
            result.frame.image,
            result.states,
            result.alerts,
            result.hazard_boxes,
            result.edges,
            result.frame.timestamp,
            fps,
            self.analyzer.cfg.panel_max_lines,
        )

    def _with_snapshot(self, event: Event, canvas) -> Event:
        if event.phase != "start" or self.snapshot_dir is None or canvas is None:
            return event
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if event.track_id is None else f"_{event.track_id}"
        path = self.snapshot_dir / f"{event.frame:08d}_{event.kind.lower()}{suffix}.jpg"
        if not cv2.imwrite(str(path), canvas):
            log.warning("could not write snapshot %s", path)
            return event
        return replace(event, snapshot=str(path))

    def _write_video(self, canvas, fps):
        if self._writer is None:
            height, width = canvas.shape[:2]
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 10.0, (width, height)
            )
            if not writer.isOpened():
                raise OSError(f"cannot write video: {self.video_path}")
            self._writer = writer
        self._writer.write(canvas)

    def _show(self, canvas) -> bool:
        if self.display_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.display_scale, fy=self.display_scale)
        cv2.imshow(WINDOW, canvas)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def _close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.display:
            cv2.destroyAllWindows()
