"""Live detector: MediaPipe Tasks FaceLandmarker + calibrated temporal fusion."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vision.config import VisionConfig, load_vision_config
from vision.eye_state import EyeStateTracker
from vision.fusion import fuse
from vision.geometry import (
    LEFT_EYE_EAR,
    MOUTH_MAR,
    RIGHT_EYE_EAR,
    combined_ear,
    eye_aspect_ratio,
    face_bbox_from_landmarks,
    landmark_coords,
    classify_head_pose,
    looking_down_from_landmarks,
    looking_down_from_signals,
    mouth_aspect_ratio,
    pitch_from_matrix,
)
from vision.mouth_state import MouthStateTracker, YawnClassifierInterface
from vision.onnx_model import OnnxFaceScorer

try:
    from vision.eye_model import OnnxEyeScorer
except ImportError:  # Experimental scorer is optional; baseline vision must load.
    OnnxEyeScorer = None

logger = logging.getLogger("neuropulse.vision")

_APP_DIR = Path(__file__).resolve().parent.parent
VISION_OUTPUT_JSON = _APP_DIR / "vision_output.json"

HAS_MEDIAPIPE = False
mp = None
FaceLandmarker = None
FaceLandmarkerOptions = None
BaseOptions = None
RunningMode = None

try:
    import mediapipe as mp_pkg
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    mp = mp_pkg
    BaseOptions = mp_python.BaseOptions
    FaceLandmarker = mp_vision.FaceLandmarker
    FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
    RunningMode = mp_vision.RunningMode
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError) as exc:
    logger.warning("MediaPipe Tasks unavailable: %s", exc)


def resolve_fatigue_onnx_path(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
    else:
        env = os.environ.get("FATIGUE_ONNX_PATH", "").strip()
        p = Path(env).expanduser() if env else (_APP_DIR / "fatigue_model.onnx")
    if not p.is_absolute():
        p = (_APP_DIR / p).resolve()
    else:
        p = p.resolve()
    return p


class FatigueDetector:
    def __init__(
        self,
        onnx_model_path: str | None = None,
        config: VisionConfig | None = None,
        config_path: Path | None = None,
    ):
        self.config = config or load_vision_config(config_path or (_APP_DIR / "vision_runtime_config.json"))
        experiment_env = os.environ.get("NEUROPULSE_EYE_MODEL_EXPERIMENT")
        self.eye_model_experiment_enabled = (
            experiment_env.strip().lower() in {"1", "true", "yes", "on"}
            if experiment_env is not None
            else self.config.eye_model_experimental_enabled
        )
        self.eye_model = (
            OnnxEyeScorer(
                _APP_DIR / "models" / "vision" / "eye_resnet18_candidate.onnx",
                _APP_DIR / "models" / "vision" / "best_resnet18_eyes_transfer.json",
                enabled=self.eye_model_experiment_enabled,
                allow_experimental=self.eye_model_experiment_enabled,
            )
            if OnnxEyeScorer is not None
            else None
        )
        if self.eye_model_experiment_enabled:
            logger.warning(
                "Experimental eye-crop model enabled; output is advisory and not deployment-approved (available=%s, error=%s)",
                bool(self.eye_model and self.eye_model.available),
                None if self.eye_model is None else self.eye_model.last_error,
            )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        self.face_mesh = None
        self._last_ts_ms = 0

        if HAS_MEDIAPIPE:
            task_path = _APP_DIR / "models" / "mediapipe" / "face_landmarker.task"
            if task_path.is_file():
                try:
                    options = FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(task_path)),
                        running_mode=RunningMode.VIDEO,
                        num_faces=1,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=True,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.face_mesh = FaceLandmarker.create_from_options(options)
                    logger.info("MediaPipe Face Landmarker loaded: %s", task_path)
                    print(f"MediaPipe Face Landmarker loaded: {task_path}")
                except Exception as exc:
                    logger.warning("MediaPipe initialization failed: %s", exc)
                    print(f"MediaPipe initialization failed: {exc}")
            else:
                print(f"Face Landmarker model not found: {task_path}")

        self._onnx_file = resolve_fatigue_onnx_path(onnx_model_path)
        self.onnx = OnnxFaceScorer(
            self._onnx_file,
            [
                self._onnx_file.parent / "vision_model_config.json",
                _APP_DIR / "vision_model_config.json",
            ],
        )
        self.has_onnx = self.onnx.available
        self.session = self.onnx.session
        if self.has_onnx:
            print(f"ONNX model loaded: {self._onnx_file}")
            print(
                f"   Vision preprocess: {self.onnx.input_size}x{self.onnx.input_size} "
                f"({self.onnx.backbone}) class_names={self.onnx.class_names} "
                f"P(drowsy)=logit[{self.onnx.drowsy_idx}]"
            )
        else:
            print(f"ONNX model not available: {self.onnx.last_error}")

        self.eyes = EyeStateTracker(
            self.config,
            calibration_path=_APP_DIR / "data" / "eye_calibration.json",
            now_fn=time.monotonic,
        )
        self.mouth = MouthStateTracker(self.config, now_fn=time.monotonic)
        self.yawn_model = YawnClassifierInterface()
        self.frame_index = 0
        self._down_started_at = None

    def close(self) -> None:
        mesh = self.face_mesh
        self.face_mesh = None
        if mesh is not None:
            closer = getattr(mesh, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.debug("FaceLandmarker.close() failed", exc_info=True)

    def recalibrate(self) -> None:
        self.eyes.start_calibration()
        print("Eye calibration reset — look at the camera with eyes naturally open.")

    def cascade_ear_continuous(self, face_roi_gray, face_w, face_h) -> Optional[float]:
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        face_area = max(face_w * face_h, 1)
        if len(eyes) == 0:
            return None
        total = sum(float(ew * eh) for (_, _, ew, eh) in eyes)
        ratio = total / face_area
        return float(np.clip(0.12 + ratio * 3.2, 0.05, 0.55))

    def cascade_mar_continuous(self, mouths, face_roi_gray) -> Optional[float]:
        fh, fw = face_roi_gray.shape[:2]
        face_area = max(fh * fw, 1)
        if len(mouths) > 0:
            ratios = [(mw * mh) / face_area for (_, _, mw, mh) in mouths]
            r = max(ratios)
            return float(np.clip(0.15 + r * 2.8, 0.05, 0.95))
        lower = face_roi_gray[int(fh * 0.55) :, :]
        if lower.size == 0:
            return None
        blur = cv2.GaussianBlur(lower, (5, 5), 0)
        std = float(np.std(blur))
        t = np.clip((std - 8.0) / 28.0, 0.0, 1.0)
        return float(np.clip(0.08 + 0.22 * t, 0.05, 0.38))

    @staticmethod
    def _eye_crop(frame, points):
        if not points or len(points) < 6:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width <= 4 or height <= 2:
            return None
        size = max(width, height) * 1.5
        center_x = (min(xs) + max(xs)) * 0.5
        center_y = (min(ys) + max(ys)) * 0.5
        x1 = max(0, int(center_x - size * 0.5))
        y1 = max(0, int(center_y - size * 0.5))
        x2 = min(frame.shape[1], int(center_x + size * 0.5))
        y2 = min(frame.shape[0], int(center_y + size * 0.5))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    def detect(self, frame):
        self.frame_index += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.monotonic()
        left_ear = right_ear = ear = mar = None
        left_eye_points = right_eye_points = None
        eye_model_closed_probability = None
        eye_source = "none"
        face_bbox = (0, 0, w, h)
        face_detected = False
        pitch_deg = None
        look_ratio = None

        if self.face_mesh is not None:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = int(now * 1000)
            if timestamp_ms <= self._last_ts_ms:
                timestamp_ms = self._last_ts_ms + 1
            self._last_ts_ms = timestamp_ms
            results = self.face_mesh.detect_for_video(mp_image, timestamp_ms)
            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                left = landmark_coords(landmarks, LEFT_EYE_EAR, w, h)
                right = landmark_coords(landmarks, RIGHT_EYE_EAR, w, h)
                left_eye_points, right_eye_points = left, right
                mouth_pts = landmark_coords(landmarks, MOUTH_MAR, w, h)
                left_ear = eye_aspect_ratio(left)
                right_ear = eye_aspect_ratio(right)
                ear = combined_ear(left_ear, right_ear)
                mar = mouth_aspect_ratio(mouth_pts)
                eye_source = "MediaPipe Tasks"
                face_detected = True
                face_bbox = face_bbox_from_landmarks(landmarks, w, h)
                look_ratio = looking_down_from_landmarks(landmarks, w, h)
                matrices = getattr(results, "facial_transformation_matrixes", None) or []
                if matrices:
                    pitch_deg = pitch_from_matrix(matrices[0])
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                face_roi = gray[fy : fy + fh, fx : fx + fw]
                mouths = self.mouth_cascade.detectMultiScale(
                    face_roi, scaleFactor=1.3, minNeighbors=8, minSize=(20, 20)
                )
                ear = self.cascade_ear_continuous(face_roi, fw, fh)
                left_ear = right_ear = ear
                mar = self.cascade_mar_continuous(mouths, face_roi)
                eye_source = "Cascade"
                face_bbox = (fx, fy, fx + fw, fy + fh)
                face_detected = True

        looking_down = looking_down_from_signals(
            pitch_deg,
            look_ratio,
            self.config.looking_down_pitch_deg,
            self.config.looking_down_landmark_ratio,
        )
        if face_detected and looking_down:
            if self._down_started_at is None:
                self._down_started_at = now
            looking_down_duration_s = now - self._down_started_at
        else:
            self._down_started_at = None
            looking_down_duration_s = 0.0
        head_pose_status = classify_head_pose(
            face_detected,
            looking_down,
            looking_down_duration_s,
            self.config.head_drop_min_s,
        )

        if self.eye_model_experiment_enabled and self.eye_model is not None and face_detected and eye_source == "MediaPipe Tasks":
            probabilities = []
            for points in (left_eye_points, right_eye_points):
                crop = self._eye_crop(frame, points)
                if crop is not None:
                    score = self.eye_model.score_closed(crop)
                    if score is not None:
                        probabilities.append(score)
            if probabilities:
                eye_model_closed_probability = float(np.mean(probabilities))

        eye_obs = self.eyes.update(
            left_ear=left_ear,
            right_ear=right_ear,
            combined_ear=ear,
            face_detected=face_detected,
            looking_down=looking_down,
            pitch_deg=pitch_deg,
            timestamp=now,
            eye_model_closed_probability=eye_model_closed_probability,
            head_pose_status=head_pose_status,
            looking_down_duration_s=looking_down_duration_s,
        )
        mouth_obs = self.mouth.update(mar, timestamp=now)

        model_score = None
        if face_detected:
            model_score = self.onnx.score(frame, face_bbox)

        fused = fuse(
            eye_obs,
            mouth_obs,
            model_score,
            config=self.config,
            model_ok=self.onnx.available and model_score is not None,
        )

        if not face_detected:
            status = "NO FACE"
            index = 0.0
        else:
            status = fused.state
            index = fused.index

        confidence = index * 100.0
        features = {
            "ear": ear if ear is not None else float("nan"),
            "mar": mar if mar is not None else float("nan"),
            "left_ear": left_ear if left_ear is not None else float("nan"),
            "right_ear": right_ear if right_ear is not None else float("nan"),
            "eye_state": eye_obs.eye_state,
            "eye_state_estimate": eye_obs.eye_state_estimate,
            "activity_label": fused.activity_label,
            "mouth_class": mouth_obs.mouth_class,
            "looking_down": looking_down,
            "pitch_deg": pitch_deg,
            "head_pose_status": head_pose_status,
            "looking_down_duration_s": looking_down_duration_s,
            "perclos": eye_obs.perclos,
            "closed_duration_s": eye_obs.closed_duration_s,
            "eye_event": eye_obs.event,
            "eye_model_experiment_enabled": self.eye_model_experiment_enabled,
            "eye_model_available": bool(self.eye_model and self.eye_model.available),
            "eye_model_closed_probability": eye_model_closed_probability,
            "eye_model_assisted_temporal_state": eye_obs.eye_model_assisted,
            "temporal_eye_state": eye_obs.eye_state,
            "calibrating": eye_obs.calibrating,
            "open_ear_baseline": eye_obs.open_ear_baseline,
            "closed_threshold": eye_obs.closed_threshold,
            "open_threshold": eye_obs.open_threshold,
            "model_score": model_score,
            "onnx_error": None if model_score is not None else self.onnx.last_error,
            "onnx_drowsy_idx": self.onnx.drowsy_idx,
            "fatigue_score": index,
            "fatigue_score_raw": index,
            "face_bbox": face_bbox,
            "eye_source": eye_source,
            "reasons": fused.reasons,
            "yawn_model": self.yawn_model.predict(None)["missing"],
        }

        n = max(1, self.config.debug_every_n_frames)
        if self.frame_index % n == 0:
            logger.info(
                "frame=%s state=%s index=%.2f EAR=%s MAR=%s ear_estimate=%s temporal_eye=%s closed_s=%.2f blink_event=%s eye_model=%s model_closed=%s assisted=%s mouth=%s",
                self.frame_index,
                status,
                index,
                ear,
                mar,
                eye_obs.eye_state_estimate,
                eye_obs.eye_state,
                eye_obs.closed_duration_s,
                eye_obs.event,
                self.eye_model_experiment_enabled,
                eye_model_closed_probability,
                eye_obs.eye_model_assisted,
                mouth_obs.mouth_class,
            )

        return status, confidence, features
