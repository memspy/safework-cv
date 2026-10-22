import numpy as np

from alerts import Alert
from config import Config
from geometry import distance_to_polyline

FEET = [27, 28, 29, 30, 31, 32]


class RuleEngine:
    def __init__(self, cfg: Config, edges):
        self.cfg = cfg
        self.edges = edges

    def evaluate(self, states, t):
        alerts = []
        for state in states:
            if not state.confirmed:
                continue
            if state.present:
                alerts.extend(self._present_alerts(state, t))
            else:
                alerts.extend(self._missing_alerts(state, t))
        return alerts

    def _present_alerts(self, state, t):
        cfg = self.cfg
        point = (float(state.centroid[0]), float(state.centroid[1]))
        alerts = []

        edge = self._edge_alert(state, t, point)
        if edge:
            alerts.append(edge)

        if state.lying:
            idle = t - state.last_motion_t
            if idle >= cfg.immobile_seconds:
                alerts.append(
                    Alert("IMMOBILE", state.track_id, point, f"lying without motion {idle:.0f}s")
                )

        fall_recent = state.fall_t is not None and t - state.fall_t < cfg.fall_alert_seconds
        if fall_recent:
            alerts.append(Alert("FALL", state.track_id, point, "sudden fall"))

        running = (
            state.running_since is not None
            and not state.coasting
            and t - state.running_since >= cfg.run_hold_seconds
        )
        if running and not fall_recent:
            alerts.append(
                Alert("RUNNING", state.track_id, point, f"running {state.speed:.1f} body/s")
            )
        return alerts

    def _missing_alerts(self, state, t):
        if state.exited or not state.occlusion_eligible:
            return []
        gap = t - state.last_seen
        if gap < self.cfg.occlusion_seconds:
            return []
        point = (float(state.last_point[0]), float(state.last_point[1]))
        return [Alert("OCCLUDED", state.track_id, point, f"out of sight {gap:.0f}s")]

    def _edge_alert(self, state, t, point):
        polylines = self.edges.polylines
        if not polylines:
            state.edge_since = None
            return None
        feet = self._ground_points(state)
        dist = min(distance_to_polyline(feet, poly) for poly in polylines)
        state.edge_ratio = dist / state.scale
        if state.edge_ratio > self.cfg.edge_distance_ratio:
            state.edge_since = None
            return None
        if state.edge_since is None:
            state.edge_since = t
        if t - state.edge_since < self.cfg.edge_hold_seconds:
            return None
        return Alert("EDGE", state.track_id, point, f"near edge {state.edge_ratio:.2f} body")

    def _ground_points(self, state):
        if state.pose_fresh:
            mask = state.visibility[FEET] >= self.cfg.min_visibility
            if mask.any():
                return state.keypoints[FEET][mask]
        x1, _, x2, y2 = state.box
        return np.array([[(x1 + x2) / 2, y2]])