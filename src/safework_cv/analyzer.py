from __future__ import annotations

from dataclasses import dataclass, replace

from .alerts import Alert, Event, EventTracker
from .config import Config
from .rules import RuleEngine
from .sources import Frame
from .state import PersonRegistry, PersonState
from .zones import ZoneSet


@dataclass
class FrameResult:
    frame: Frame
    stream_time: float
    states: list[PersonState]
    alerts: list[Alert]
    events: list[Event]
    hazard_boxes: list
    edges: list


class Analyzer:
    def __init__(self, cfg: Config, detector, pose, hazards, zones: ZoneSet | None = None):
        self.cfg = cfg
        self.detector = detector
        self.hazards = hazards
        self.zones = zones or ZoneSet()
        self.registry = PersonRegistry(cfg, pose)
        self.rules = RuleEngine(cfg)
        self.tracker = EventTracker(cfg.alert_clear_seconds)
        self.t0 = None
        self._edges_size = None
        self._edges = []

    def process(self, frame: Frame) -> FrameResult:
        t = frame.timestamp
        if self.t0 is None:
            self.t0 = t
        height, width = frame.image.shape[:2]
        detections = self.detector.detect(frame.image, t, self.registry.lost_boxes(t))
        people = [d.box for d in detections if d.rotated or d.score >= self.cfg.new_track_score]
        hazard_alerts = self.hazards.update(frame.image, t, people)
        self.registry.update(frame.image, detections, t, width, height)
        states = list(self.registry.states.values())
        edges = self._edges_for(width, height)
        rule_alerts = self.rules.evaluate(states, t, edges)
        alerts = sorted(hazard_alerts + rule_alerts, key=lambda a: a.rank)
        stream_time = t - self.t0
        events = [
            replace(event, frame=frame.index) for event in self.tracker.update(alerts, stream_time)
        ]
        return FrameResult(frame, stream_time, states, alerts, events, self.hazards.boxes, edges)

    def _edges_for(self, width, height):
        if self._edges_size != (width, height):
            self._edges = self.zones.edges_px(width, height)
            self._edges_size = (width, height)
        return self._edges
