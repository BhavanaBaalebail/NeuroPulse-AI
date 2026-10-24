"""
NeuroPulse vision pipeline: calibrated EAR, temporal yawn candidates, and
explainable fusion. The on-screen index is not a calibrated probability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import cv2

from vision.detector import FatigueDetector, VISION_OUTPUT_JSON
from vision.overlay import draw_status

last_beep_time = 0.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def beep():
    if sys.platform == "darwin":
        os.system("afplay /System/Library/Sounds/Glass.aiff &")


def write_vision_bridge(status: str, confidence: float, features: dict) -> None:
    payload = {
        "fatigue_score": float(features.get("fatigue_score", confidence / 100.0)),
        "label": status if status in ("ALERT", "DROWSY") else "ALERT",
        "activity_label": features.get("activity_label") or "",
        "confidence_pct": float(confidence),
        "reasons": features.get("reasons", []),
        "eye_state": features.get("eye_state"),
        "mouth_class": features.get("mouth_class"),
        "head_pose_status": features.get("head_pose_status", "unknown"),
        "looking_down": features.get("looking_down", False),
        "looking_down_duration_s": features.get("looking_down_duration_s", 0.0),
        "pitch_deg": features.get("pitch_deg"),
        "index_is_probability": False,
        "updated_at": time.time(),
        "ear": features.get("ear"),
        "mar": features.get("mar"),
        "perclos": features.get("perclos"),
        "model_score": features.get("model_score"),
    }
    try:
        with open(VISION_OUTPUT_JSON, "w", encoding="utf-8") as vf:
            json.dump(payload, vf)
    except OSError:
        pass


def main():
    global last_beep_time

    print("\n" + "=" * 75)
    print("NeuroPulse vision — temporal drowsiness cues (not a medical score)")
    print("=" * 75)
    print("Look at the camera with eyes open for ~2s to calibrate.")
    print("Keys: C = recalibrate   Q = quit")
    print("=" * 75 + "\n")

    cap = cv2.VideoCapture(0)
    detector = FatigueDetector()
    cooldown = detector.config.beep_cooldown_s

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            status, confidence, features = detector.detect(frame)
            draw_status(frame, status, confidence, features)
            write_vision_bridge(status, confidence, features)

            now = time.time()
            if status == "DROWSY" and (now - last_beep_time > cooldown):
                beep()
                last_beep_time = now

            cv2.imshow("Fatigue Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                detector.recalibrate()
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetection stopped")


if __name__ == "__main__":
    main()
