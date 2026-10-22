import numpy as np


def unit_vector(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def angle_between_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def box_corners(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def box_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float64)


def box_scale(box):
    return float(max(box[2] - box[0], box[3] - box[1]))


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def box_iou(a, b):
    inter = _intersection(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def box_cover_fraction(a, b):
    area = box_area(a)
    return _intersection(a, b) / area if area > 0 else 0.0


def projected_extents(points, up):
    pts = np.asarray(points, dtype=np.float64)
    side = np.array([-up[1], up[0]])
    return float(np.ptp(pts @ up)), float(np.ptp(pts @ side))


def distance_to_polyline(points, polyline):
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    poly = np.asarray(polyline, dtype=np.float64)
    a = poly[:-1][None, :, :]
    ab = (poly[1:] - poly[:-1])[None, :, :]
    ap = pts[:, None, :] - a
    denom = np.maximum((ab * ab).sum(axis=2), 1e-9)
    t = np.clip((ap * ab).sum(axis=2) / denom, 0.0, 1.0)
    closest = a + t[:, :, None] * ab
    return float(np.linalg.norm(pts[:, None, :] - closest, axis=2).min())