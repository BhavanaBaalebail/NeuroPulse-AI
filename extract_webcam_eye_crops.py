#!/usr/bin/env python3
"""
Extract verified RGB eye crops from local webcam stills using MediaPipe FaceLandmarker.
Only high-confidence open (EAR >= 0.24) and closed (EAR <= 0.15) eyes are included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vision.dataset_split import source_key
from vision.geometry import LEFT_EYE_EAR, RIGHT_EYE_EAR, eye_aspect_ratio, landmark_coords

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "derived" / "webcam_eyes"


def crop_eye_with_margin(img: np.ndarray, coords: list[tuple[float, float]], margin_scale: float = 0.5) -> np.ndarray | None:
    if not coords or len(coords) < 6:
        return None
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 4 or h <= 2:
        return None

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    size = max(w, h) * (1.0 + margin_scale)
    x1 = max(0, int(cx - size / 2.0))
    y1 = max(0, int(cy - size / 2.0))
    x2 = min(img.shape[1], int(cx + size / 2.0))
    y2 = min(img.shape[0], int(cy + size / 2.0))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--open-ear-min", type=float, default=0.24)
    ap.add_argument("--closed-ear-max", type=float, default=0.15)
    args = ap.parse_args()

    out_dir = Path(args.out)
    open_dir = out_dir / "open"
    closed_dir = out_dir / "closed"
    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    task_path = ROOT / "models" / "mediapipe" / "face_landmarker.task"
    if not task_path.is_file():
        raise SystemExit(f"MediaPipe task model not found: {task_path}")

    base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    data_root = Path(args.data)
    counts = {"open": 0, "closed": 0, "ambiguous_skipped": 0, "no_face": 0}

    for class_name, default_label in [("notdrowsy", "open"), ("drowsy", "closed")]:
        class_folder = data_root / class_name
        if not class_folder.is_dir():
            continue
        for img_path in sorted(class_folder.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_img)

            if not res.face_landmarks:
                counts["no_face"] += 1
                continue

            landmarks = res.face_landmarks[0]
            h, w = img.shape[:2]
            l_pts = landmark_coords(landmarks, LEFT_EYE_EAR, w, h)
            r_pts = landmark_coords(landmarks, RIGHT_EYE_EAR, w, h)
            ear_l = eye_aspect_ratio(l_pts)
            ear_r = eye_aspect_ratio(r_pts)

            s_key = source_key(img_path, class_name)
            safe_source = s_key.replace(":", "_")

            for side, pts, ear in [("left", l_pts, ear_l), ("right", r_pts, ear_r)]:
                if ear is None:
                    continue
                crop = crop_eye_with_margin(img, pts)
                if crop is None:
                    continue

                if ear >= args.open_ear_min:
                    target_dir = open_dir
                    target_state = "open"
                elif ear <= args.closed_ear_max:
                    target_dir = closed_dir
                    target_state = "closed"
                else:
                    counts["ambiguous_skipped"] += 1
                    continue

                out_filename = f"webcam_{target_state}_capture_{safe_source}_{side}.png"
                cv2.imwrite(str(target_dir / out_filename), crop)
                counts[target_state] += 1

    landmarker.close()

    provenance = {
        "dataset": "Webcam RGB Eye Crops",
        "source": str(data_root),
        "modality": "rgb_webcam_crops",
        "labels": {"0": "open", "1": "closed"},
        "counts": counts,
        "ear_thresholds": {
            "open_ear_min": args.open_ear_min,
            "closed_ear_max": args.closed_ear_max,
        },
        "description": "High-confidence RGB eye crops extracted from local webcam stills via MediaPipe FaceLandmarker.",
    }
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(json.dumps(provenance, indent=2))
    print(f"Webcam eye crops saved: {counts['open']} open, {counts['closed']} closed.")


if __name__ == "__main__":
    main()
