from dataclasses import dataclass

import numpy as np

from config import Config
from geometry import box_center, box_iou, box_scale


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    born_t: float
    last_t: float
    velocity: np.ndarray
    slow: bool = False
    hits: int = 1
    confirmed: bool = False


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, t):
        matches = []
        used = set()
        for track, index in self._match(detections, t):
            used.add(index)
            self._apply(track, detections[index], t)
            matches.append((track, detections[index]))
        for index, det in enumerate(detections):
            if index in used:
                continue
            if not det.rotated and det.score < self.cfg.new_track_score:
                continue
            track = Track(
                self.next_id, det.box, det.score, t, t, np.zeros(2), slow=det.rotated
            )
            self.next_id += 1
            self.tracks[track.track_id] = track
            matches.append((track, det))
        self._prune(t)
        return matches

    def drop(self, track_id):
        self.tracks.pop(track_id, None)

    def _match(self, detections, t):
        cfg = self.cfg
        candidates = []
        for track in self.tracks.values():
            gap = t - track.last_t
            predicted = box_center(track.box) + track.velocity * min(gap, 1.0)
            gate = cfg.match_gate if gap <= 1.0 else cfg.relink_radius_ratio
            for index, det in enumerate(detections):
                if gap > 1.0 and not det.rotated and det.score < cfg.new_track_score:
                    continue
                scale = max(box_scale(track.box), box_scale(det.box))
                dist = float(np.linalg.norm(predicted - box_center(det.box))) / scale
                if dist <= gate:
                    candidates.append((dist - box_iou(track.box, det.box), track.track_id, index))
        candidates.sort()
        used_tracks = set()
        used_dets = set()
        pairs = []
        for _, track_id, index in candidates:
            if track_id in used_tracks or index in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(index)
            pairs.append((self.tracks[track_id], index))
        return pairs

    def _apply(self, track, det, t):
        cfg = self.cfg
        gap = t - track.last_t
        if 1e-3 < gap <= 1.0:
            shift = (box_center(det.box) - box_center(track.box)) / gap
            track.velocity = 0.5 * track.velocity + 0.5 * shift
        elif gap > 1.0:
            track.velocity = np.zeros(2)
        track.box = det.box
        track.score = 0.7 * track.score + 0.3 * det.score
        track.hits += 1
        track.last_t = t
        if not det.rotated:
            track.slow = False
        need = cfg.slow_confirm_hits if track.slow else cfg.confirm_hits
        if (
            not track.confirmed
            and track.hits >= need
            and t - track.born_t >= cfg.confirm_seconds
            and track.score >= cfg.confirm_min_score
        ):
            track.confirmed = True

    def _prune(self, t):
        cfg = self.cfg
        for track_id, track in list(self.tracks.items()):
            if track.confirmed:
                limit = cfg.occlusion_forget_seconds
            elif track.slow:
                limit = cfg.slow_ttl_seconds
            else:
                limit = cfg.tentative_ttl_seconds
            if t - track.last_t > limit:
                del self.tracks[track_id]