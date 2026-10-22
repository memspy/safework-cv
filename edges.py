import json
import os

import cv2
import numpy as np


class EdgeSet:
    def __init__(self, path, region, display_scale=1.0):
        self.path = path
        self.region = [region["left"], region["top"], region["width"], region["height"]]
        self.display_scale = display_scale
        self.polylines = []
        self.draft = []
        self.editing = False
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("region") != self.region:
            print("edges file was saved for a different capture region")
        self.polylines = [np.asarray(p, dtype=np.float64) for p in data.get("edges", [])]

    def save(self):
        data = {"region": self.region, "edges": [p.tolist() for p in self.polylines]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def on_mouse(self, event, x, y, flags, param):
        if self.editing and event == cv2.EVENT_LBUTTONDOWN:
            self.draft.append((x / self.display_scale, y / self.display_scale))

    def toggle_edit(self):
        self.editing = not self.editing
        self.draft = []

    def finish(self):
        if len(self.draft) >= 2:
            self.polylines.append(np.array(self.draft, dtype=np.float64))
            self.save()
        self.draft = []

    def undo(self):
        if self.draft:
            self.draft.pop()

    def clear(self):
        self.polylines = []
        self.draft = []
        self.save()