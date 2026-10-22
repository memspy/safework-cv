import time
from dataclasses import dataclass

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


class AlertLog:
    def __init__(self):
        self.active = set()

    def update(self, alerts):
        stamp = time.strftime("%H:%M:%S")
        for alert in alerts:
            if (alert.kind, alert.track_id) not in self.active:
                print(f"[{stamp}] {alert.label}: {alert.text}")
        self.active = {(a.kind, a.track_id) for a in alerts}