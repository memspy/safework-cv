import cv2
import mss
import numpy as np


class ScreenCapture:
    def __init__(self, monitor_index=1, region=None):
        self.sct = mss.mss()
        self.monitor = dict(self.sct.monitors[monitor_index])
        self.region = region or dict(self.monitor)

    def grab(self, region=None):
        shot = self.sct.grab(region or self.region)
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])

    def pick_region(self, window="select region"):
        frame = self.grab(self.monitor)
        ratio = min(1.0, 1280 / frame.shape[1])
        preview = cv2.resize(frame, None, fx=ratio, fy=ratio) if ratio < 1.0 else frame
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

    def close(self):
        self.sct.close()