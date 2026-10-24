"""Landmark geometry: EAR, MAR, simple head-pose proxies."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

# Standard MediaPipe Face Mesh indices for Soukupová–Čech EAR.
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
# Inner lips (13, 14) and mouth corners (78, 308).
MOUTH_MAR = [13, 14, 78, 308]
FACE_HULL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
FOREHEAD_IDX = 10
CHIN_IDX = 152
NOSE_IDX = 1


def distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def landmark_xy(landmarks, index: int, width: float, height: float) -> Optional[tuple[float, float]]:
    if index < 0 or index >= len(landmarks):
        return None
    lm = landmarks[index]
    x = getattr(lm, "x", None)
    y = getattr(lm, "y", None)
    if x is None or y is None:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (float(x) * width, float(y) * height)


def landmark_coords(
    landmarks, indices: Iterable[int], width: float, height: float
) -> Optional[list[tuple[float, float]]]:
    coords = []
    for idx in indices:
        pt = landmark_xy(landmarks, idx, width, height)
        if pt is None:
            return None
        coords.append(pt)
    return coords


def eye_aspect_ratio(eye_coords: Optional[Sequence[Sequence[float]]]) -> Optional[float]:
    """EAR = (|p1-p5| + |p2-p4|) / (2 |p0-p3|). None if landmarks are unusable."""
    if eye_coords is None or len(eye_coords) < 6:
        return None
    p = eye_coords
    a = distance(p[1], p[5])
    b = distance(p[2], p[4])
    c = distance(p[0], p[3])
    if c <= 1e-6:
        return None
    ear = (a + b) / (2.0 * c)
    if not math.isfinite(ear) or ear < 0.0 or ear > 1.5:
        return None
    return float(ear)


def mouth_aspect_ratio(mouth_coords: Optional[Sequence[Sequence[float]]]) -> Optional[float]:
    """MAR = inner-lip opening / mouth width. None if unusable."""
    if mouth_coords is None or len(mouth_coords) < 4:
        return None
    vertical = distance(mouth_coords[0], mouth_coords[1])
    horizontal = distance(mouth_coords[2], mouth_coords[3])
    if horizontal <= 1e-6:
        return None
    mar = vertical / horizontal
    if not math.isfinite(mar) or mar < 0.0 or mar > 5.0:
        return None
    return float(mar)


def combined_ear(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return (left + right) / 2.0


def pitch_from_matrix(matrix) -> Optional[float]:
    """Approximate pitch in degrees from a 4x4 facial transformation matrix."""
    try:
        import numpy as np

        arr = np.asarray(matrix, dtype=float).reshape(-1)
        if arr.size < 16:
            r20 = float(matrix[2][0])
            r21 = float(matrix[2][1])
            r22 = float(matrix[2][2])
        else:
            m = arr.reshape(4, 4)
            r20 = float(m[2, 0])
            r21 = float(m[2, 1])
            r22 = float(m[2, 2])
    except (TypeError, IndexError, ValueError):
        return None
    denom = math.hypot(r21, r22)
    if denom <= 1e-6:
        return None
    pitch = math.degrees(math.atan2(-r20, denom))
    if not math.isfinite(pitch):
        return None
    return pitch


def looking_down_from_signals(
    pitch_deg: Optional[float],
    look_ratio: Optional[float],
    pitch_thresh: float,
    ratio_thresh: float,
) -> bool:
    """True if matrix pitch *or* landmark ratio indicates a downward head pose."""
    if pitch_deg is not None and pitch_deg <= pitch_thresh:
        return True
    if look_ratio is not None and look_ratio >= ratio_thresh:
        return True
    return False


def classify_head_pose(
    face_detected: bool,
    looking_down: bool,
    down_duration_s: float,
    head_drop_min_s: float,
) -> str:
    """level | glance_down | head_dropping | unknown. Duration in seconds."""
    if not face_detected:
        return "unknown"
    if not looking_down or down_duration_s <= 0.0:
        return "level"
    if down_duration_s >= head_drop_min_s:
        return "head_dropping"
    return "glance_down"


def looking_down_from_landmarks(landmarks, width: float, height: float) -> Optional[float]:
    """
    Nose position between eye-line and chin, in [0, 1].
    Larger values usually mean the face is pitched down toward the camera/keyboard.
    """
    forehead = landmark_xy(landmarks, FOREHEAD_IDX, width, height)
    chin = landmark_xy(landmarks, CHIN_IDX, width, height)
    nose = landmark_xy(landmarks, NOSE_IDX, width, height)
    left = landmark_xy(landmarks, RIGHT_EYE_EAR[0], width, height)
    right = landmark_xy(landmarks, LEFT_EYE_EAR[0], width, height)
    if not all((forehead, chin, nose, left, right)):
        return None
    eye_y = (left[1] + right[1]) * 0.5
    span = chin[1] - eye_y
    if span <= 1e-6:
        return None
    return float((nose[1] - eye_y) / span)


def face_bbox_from_landmarks(landmarks, width: int, height: int, pad: int = 12) -> tuple[int, int, int, int]:
    xs, ys = [], []
    for i in FACE_HULL_IDX:
        pt = landmark_xy(landmarks, i, width, height)
        if pt is not None:
            xs.append(pt[0])
            ys.append(pt[1])
    if not xs:
        for lm in landmarks:
            xs.append(float(lm.x) * width)
            ys.append(float(lm.y) * height)
    if not xs:
        return (0, 0, width, height)
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(width, int(max(xs)) + pad)
    y2 = min(height, int(max(ys)) + pad)
    return (x1, y1, x2, y2)
