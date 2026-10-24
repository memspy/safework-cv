import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class ZoneSet:
    edges: list = field(default_factory=list)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported zones file, expected version {SCHEMA_VERSION}")
        edges = []
        for index, raw in enumerate(data.get("edges", [])):
            points = np.asarray(raw, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                raise ValueError(
                    f"{path}: edge {index} must be a list of at least two [x, y] points"
                )
            if points.min() < 0.0 or points.max() > 1.0:
                raise ValueError(f"{path}: edge {index} has coordinates outside [0, 1]")
            edges.append(points)
        return cls(edges)

    def save(self, path):
        data = {
            "version": SCHEMA_VERSION,
            "edges": [np.round(points, 5).tolist() for points in self.edges],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(partial, target)

    def add_edge(self, points_px, width, height):
        points = np.asarray(points_px, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("an edge needs at least two points")
        scale = np.array([width, height], dtype=np.float64)
        self.edges.append(np.clip(points / scale, 0.0, 1.0))

    def edges_px(self, width, height):
        scale = np.array([width, height], dtype=np.float64)
        return [points * scale for points in self.edges]

    def clear(self):
        self.edges = []
