"""On-screen diagnostics. The index is labeled as an index, not a probability."""

from __future__ import annotations

import cv2


STATE_COLORS = {
    "ALERT": (0, 220, 0),
    "DROWSY": (0, 0, 255),
}


def format_status_headline(status: str, activity: str = "") -> str:
    title = str(status or "").replace("?", "").strip()
    if title not in STATE_COLORS:
        title = "ALERT"
    activity = str(activity or "").replace("?", "").strip()
    if activity:
        return f"{title} - {activity}"
    return title


def draw_status(frame, status, confidence, features) -> None:
    title = str(status or "").replace("?", "").strip()
    if title not in STATE_COLORS:
        title = "ALERT"
    color = STATE_COLORS.get(title, (0, 220, 0))
    if "face_bbox" in features:
        x1, y1, x2, y2 = features["face_bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    activity = str(features.get("activity_label") or "").replace("?", "").strip()
    headline = format_status_headline(title, activity)
    cv2.putText(
        frame,
        headline,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    ear = features.get("ear")
    left = features.get("left_ear")
    right = features.get("right_ear")
    eye_state = features.get("eye_state", "UNKNOWN")
    baseline = features.get("open_ear_baseline")
    mar = features.get("mar")
    mouth_class = features.get("mouth_class", "unknown")
    src = features.get("eye_source", "")
    cal = "CALIBRATING" if features.get("calibrating") else (
        f"base {baseline:.3f}" if baseline is not None else "uncalibrated"
    )

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    y = 72
    lines = [
        f"index {confidence:.0f}  {eye_state}  EAR {fmt(ear)}  L {fmt(left)}  R {fmt(right)}  [{src}]",
        f"eyes {cal}  close<{features.get('closed_threshold', 0):.3f}  open>{features.get('open_threshold', 0):.3f}",
        f"MAR {fmt(mar)}  mouth={mouth_class}  activity={activity or '—'}",
        f"PERCLOS {features.get('perclos', 0):.2f}  closed_s {features.get('closed_duration_s', 0):.2f}",
    ]
    onnx_err = features.get("onnx_error")
    if onnx_err:
        lines.append(f"ONNX: {onnx_err[:70]}")
    elif features.get("model_score") is not None:
        lines.append(
            f"face CNN softmax[{features.get('onnx_drowsy_idx', 1)}]={features.get('model_score'):.2f} (confirm only)"
        )

    reasons = features.get("reasons") or []
    if reasons:
        lines.append("why: " + "; ".join(reasons[:3]))

    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
        y += 22

    cv2.putText(
        frame,
        "C=recalibrate eyes   Q=quit   index is not a probability",
        (20, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
    )
