from dataclasses import dataclass
from datetime import datetime, timezone

PRIORITY = ["FIRE", "SMOKE", "FALL", "EDGE", "IMMOBILE", "RUNNING", "OCCLUDED"]


@dataclass(frozen=True)
class Alert:
    kind: str
    track_id: int | None
    point: tuple
    text: str

    @property
    def rank(self):
        return PRIORITY.index(self.kind)

    @property
    def label(self):
        if self.track_id is None:
            return self.kind
        return f"{self.kind} #{self.track_id}"


@dataclass(frozen=True)
class Event:
    phase: str
    kind: str
    track_id: int | None
    text: str
    point: tuple
    video_time: float
    wall_time: str
    duration: float | None = None
    frame: int = 0
    snapshot: str | None = None

    def to_dict(self):
        return {
            "phase": self.phase,
            "kind": self.kind,
            "track_id": self.track_id,
            "text": self.text,
            "point": [round(float(self.point[0]), 1), round(float(self.point[1]), 1)],
            "video_time": round(self.video_time, 3),
            "wall_time": self.wall_time,
            "duration": None if self.duration is None else round(self.duration, 3),
            "frame": self.frame,
            "snapshot": self.snapshot,
        }


class EventTracker:
    def __init__(self, release_seconds):
        self.release_seconds = release_seconds
        self.active = {}

    def update(self, alerts, t):
        events = []
        current = set()
        for alert in alerts:
            key = (alert.kind, alert.track_id)
            current.add(key)
            entry = self.active.get(key)
            if entry is None:
                events.append(self._event("start", alert, t))
                self.active[key] = {"alert": alert, "since": t, "last": t}
            else:
                entry["alert"] = alert
                entry["last"] = t
        for key in list(self.active):
            if key in current:
                continue
            entry = self.active[key]
            if t - entry["last"] >= self.release_seconds:
                events.append(self._event("end", entry["alert"], t, t - entry["since"]))
                del self.active[key]
        return events

    @staticmethod
    def _event(phase, alert, t, duration=None):
        return Event(
            phase=phase,
            kind=alert.kind,
            track_id=alert.track_id,
            text=alert.text,
            point=alert.point,
            video_time=t,
            wall_time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            duration=duration,
        )
