import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
]

COLORS = {
    "FIRE": (0, 0, 255),
    "SMOKE": (180, 180, 180),
    "EDGE": (0, 0, 255),
    "IMMOBILE": (0, 0, 255),
    "FALL": (0, 0, 255),
    "RUNNING": (0, 140, 255),
    "OCCLUDED": (0, 215, 255),
}

OK_COLOR = (0, 200, 0)
EDGE_COLOR = (0, 255, 255)
DRAFT_COLOR = (255, 200, 0)


def _pt(p):
    return int(p[0]), int(p[1])


def _text(canvas, text, org, color, scale=0.6):
    cv2.putText(canvas, text, org, FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, FONT, scale, color, 2, cv2.LINE_AA)


def _draw_person(canvas, state, kind):
    color = COLORS.get(kind, OK_COLOR)
    x1, y1, x2, y2 = state.box.astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1 if state.coasting else 2)
    if state.keypoints is not None:
        visible = state.visibility >= 0.5
        for a, b in CONNECTIONS:
            if visible[a] and visible[b]:
                cv2.line(
                    canvas, _pt(state.keypoints[a]), _pt(state.keypoints[b]), (255, 255, 255), 2
                )
    label = f"#{state.track_id}{' lying' if state.lying else ''} v={state.speed:.1f}"
    if state.coasting:
        label += " held"
    _text(canvas, label, (x1, max(18, y1 - 6)), color, 0.5)


def _draw_last_known(canvas, state, now):
    color = COLORS["OCCLUDED"]
    p = _pt(state.last_point)
    cv2.circle(canvas, p, 12, color, 2)
    cv2.drawMarker(canvas, p, color, cv2.MARKER_CROSS, 18, 2)
    _text(
        canvas,
        f"#{state.track_id} lost {now - state.last_seen:.0f}s",
        (p[0] + 16, p[1] - 8),
        color,
        0.5,
    )


def _draw_panel(canvas, alerts, editing, max_lines):
    shown = alerts[:max_lines]
    lines = [(f"{a.label}: {a.text}", COLORS[a.kind]) for a in shown]
    hidden = len(alerts) - len(shown)
    if hidden > 0:
        lines.append((f"+{hidden} more", (200, 200, 200)))
    if editing:
        lines.append(("EDGE EDIT: click points, Enter to finish, u undo, c clear", DRAFT_COLOR))
    y = 26
    for text, color in lines:
        _text(canvas, text, (12, y), color)
        y += 26


def render(frame, states, alerts, hazard_boxes, edges, now, fps, cfg):
    canvas = frame.copy()
    for poly in edges.polylines:
        cv2.polylines(canvas, [poly.astype(np.int32)], False, EDGE_COLOR, 2)
    if edges.draft:
        pts = np.array(edges.draft, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, DRAFT_COLOR, 2)
        for p in pts:
            cv2.circle(canvas, _pt(p), 4, DRAFT_COLOR, -1)
    for kind, box, _ in hazard_boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS[kind], 2)
        _text(canvas, kind, (x1, max(18, y1 - 6)), COLORS[kind], 0.5)
    flagged = {}
    for alert in alerts:
        if alert.track_id is not None:
            flagged.setdefault(alert.track_id, alert.kind)
    for state in states:
        if not state.confirmed:
            continue
        if state.present:
            _draw_person(canvas, state, flagged.get(state.track_id))
        elif flagged.get(state.track_id) == "OCCLUDED":
            _draw_last_known(canvas, state, now)
    _draw_panel(canvas, alerts, edges.editing, cfg.panel_max_lines)
    _text(canvas, f"{fps:.1f} fps", (12, canvas.shape[0] - 12), (200, 200, 200), 0.5)
    if cfg.display_scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=cfg.display_scale, fy=cfg.display_scale)
    return canvas