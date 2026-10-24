from __future__ import annotations

import cv2
import numpy as np

from .zones import ZoneSet

WINDOW = "safework-cv zones"
MAX_WIDTH = 1280
LINE_COLOR = (0, 255, 255)
DRAFT_COLOR = (255, 200, 0)
HELP = "click: add point | Enter: finish edge | u: undo point | d: delete last edge | c: clear | s: save | q: quit"


class ZoneEditor:
    def __init__(self, image: np.ndarray, zones: ZoneSet):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.scale = min(1.0, MAX_WIDTH / self.width)
        self.zones = ZoneSet([points.copy() for points in zones.edges])
        self.draft: list[tuple[float, float]] = []

    def run(self) -> ZoneSet | None:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self._render())
                key = cv2.waitKey(30) & 0xFF
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return None
                if key == ord("q"):
                    return None
                if key == ord("s"):
                    self._finish()
                    return self.zones
                self._handle_key(key)
        finally:
            cv2.destroyWindow(WINDOW)

    def _handle_key(self, key: int) -> None:
        if key == 13:
            self._finish()
        elif key == ord("u") and self.draft:
            self.draft.pop()
        elif key == ord("d") and self.zones.edges:
            self.zones.edges.pop()
        elif key == ord("c"):
            self.zones.clear()
            self.draft = []

    def _finish(self) -> None:
        if len(self.draft) >= 2:
            self.zones.add_edge(self.draft, self.width, self.height)
        self.draft = []

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draft.append((x / self.scale, y / self.scale))

    def _render(self) -> np.ndarray:
        canvas = self.image.copy()
        for polyline in self.zones.edges_px(self.width, self.height):
            cv2.polylines(canvas, [polyline.astype(np.int32)], False, LINE_COLOR, 2)
        if self.draft:
            pts = np.array(self.draft, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, DRAFT_COLOR, 2)
            for p in pts:
                cv2.circle(canvas, tuple(int(v) for v in p), 4, DRAFT_COLOR, -1)
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale)
        cv2.putText(
            canvas, HELP, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, HELP, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        return canvas
