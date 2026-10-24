"""
Continuous fatigue scoring: EAR + MAR + ONNX (weighted), with recalibration,
strong-drowsy boosts, alert dampening, and a short temporal average for stability.
"""

from collections import deque
import json
import os
import sys
import time
from pathlib import Path

import cv2
import math
import numpy as np
import onnxruntime as ort

# Audio alert (macOS afplay); cooldown in main loop
last_beep_time = 0.0

_APP_DIR = Path(__file__).resolve().parent

# Streamlit Vision Mode: live bridge file (project root, next to app.py)
VISION_OUTPUT_JSON = _APP_DIR / "vision_output.json"


def resolve_fatigue_onnx_path(explicit: str | None = None) -> Path:
    """
    Default: ``<app_dir>/fatigue_model.onnx`` (from ``python export_resnet_onnx.py``).
    Override with ``FATIGUE_ONNX_PATH`` or pass ``explicit``. Relative paths are resolved under ``_APP_DIR``.
    """
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


def beep():
    """Short non-blocking system sound (macOS)."""
    if sys.platform == "darwin":
        os.system("afplay /System/Library/Sounds/Glass.aiff &")


def beep_strong():
    """Optional stronger alert for very low confidence (macOS)."""
    if sys.platform == "darwin":
        os.system("afplay /System/Library/Sounds/Sosumi.aiff &")

# --- MediaPipe (landmarks) when available ---

# --- MediaPipe Tasks API ---
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
    print(f"MediaPipe Tasks unavailable: {exc}")

if not HAS_MEDIAPIPE:
    print("MediaPipe unavailable. Using cascade-based EAR/MAR proxies.")


# ==================== CONFIG ====================

# EAR → eye_score = (0.5 - ear) / (0.5 - 0.18), clamped [0, 1]
EAR_OPEN_REF = 0.5
EAR_CLOSED_REF = 0.15  # still used in cascade proxy clipping
EYE_SCORE_DENOM = EAR_OPEN_REF - 0.18  # 0.32 — stronger sensitivity vs closed eyes

# MAR → mouth_score = mar / MAR_SCALE
MAR_SCALE = 0.5

# Temporal smoothing of final fatigue (post recalibration)
SCORE_BUFFER_MAXLEN = 5

# Weighted fusion (EAR fixed to standard MediaPipe indices — eye channel is trustworthy again)
W_EYE = 0.50
W_MOUTH = 0.20
W_MODEL = 0.30

# Display uses `confidence` = fatigue_score×100 (after optional invert). DROWSY + red when
# this percentage is *below* the first value; ALERT + green otherwise.
DISPLAY_DROWSY_BELOW_PCT = 40.0
BEEP_BELOW_PCT = 25.0


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# Rarely needed: FATIGUE_SCORE_INVERT=1 if fused score polarity is still backwards.
FATIGUE_SCORE_INVERT = _env_flag("FATIGUE_SCORE_INVERT", "0")

# STEP 8 debug lines; set to 1 for every frame
DEBUG_PRINT_EVERY_N_FRAMES = 15

COLOR_ALERT = (0, 0, 255)
COLOR_DROWSY = (0, 255, 0)

# Vision ONNX: after ``train_resnet.py``, run ``export_resnet_onnx.py`` and keep
# ``fatigue_model.onnx`` + ``vision_model_config.json`` next to this file (or next to the ONNX if using FATIGUE_ONNX_PATH).

#  Face Mesh: six points per eye in order for EAR = (|p1-p5|+|p2-p4|) / (2|p0-p3|)
# (standard tutorial indices — not the first six of the full eyelid polygon).
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
# MAR: vertical (lip opening) / horizontal (mouth width) — inner lip / corners
MOUTH_MAR = [13, 14, 78, 308]
# Loose hull for face rectangle (debug / bbox) when drawing
FACE_HULL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


# ==================== DETECTOR ====================


class FatigueDetector:
    def __init__(self, onnx_model_path: str | None = None):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


        self.face_mesh = None

        if HAS_MEDIAPIPE:
            task_path = _APP_DIR / "models" / "mediapipe" / "face_landmarker.task"

            if task_path.is_file():
                try:
                    options = FaceLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(task_path)
                        ),
                        running_mode=RunningMode.VIDEO,
                        num_faces=1,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.face_mesh = FaceLandmarker.create_from_options(options)
                    print(f"MediaPipe Face Landmarker loaded: {task_path}")
                except Exception as exc:
                    print(f"MediaPipe initialization failed: {exc}")
            else:
                print(f"Face Landmarker model not found: {task_path}")

        self._onnx_file = resolve_fatigue_onnx_path(onnx_model_path)

        try:
            self.session = ort.InferenceSession(
                str(self._onnx_file),
                providers=["CPUExecutionProvider"],
            )
            self.has_onnx = True
            self._onnx_input_name = self.session.get_inputs()[0].name
            print(f"✅ ONNX model loaded: {self._onnx_file}")
        except Exception as e:
            print(f"⚠️  ONNX model not available: {e}")
            self.session = None
            self.has_onnx = False
            self._onnx_input_name = "image"

        # Optional: ResNet / ImageNet preprocessing (written by train_resnet.py)
        self._onnx_input_size = 64
        self._onnx_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._onnx_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._onnx_drowsy_idx = int(os.environ.get("FATIGUE_ONNX_DROWSY_CLASS", "1"))
        _cfg_loaded = False
        for _cfg_path in (
            self._onnx_file.parent / "vision_model_config.json",
            _APP_DIR / "vision_model_config.json",
        ):
            if not _cfg_path.is_file():
                continue
            try:
                with open(_cfg_path, encoding="utf-8") as cf:
                    _vc = json.load(cf)
                self._onnx_input_size = int(_vc.get("input_size", 64))
                self._onnx_mean = np.array(_vc["mean"], dtype=np.float32).reshape(1, 1, 3)
                self._onnx_std = np.array(_vc["std"], dtype=np.float32).reshape(1, 1, 3)
                if "onnx_drowsy_class_index" in _vc:
                    self._onnx_drowsy_idx = int(_vc["onnx_drowsy_class_index"])
                print(
                    f"   Vision preprocess: {self._onnx_input_size}×{self._onnx_input_size} "
                    f"({_vc.get('backbone', 'custom')})  ← {_cfg_path.name}"
                )
                _cfg_loaded = True
                break
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
        if self.has_onnx and not _cfg_loaded:
            print(
                "   Vision preprocess: 64×64 (0.5 norm) — add vision_model_config.json for ResNet/ImageNet"
            )

        _mode = os.environ.get("FATIGUE_ONNX_INPUT", "face").strip().lower()
        if _mode not in ("full", "face"):
            _mode = "face"
        self._onnx_input_mode = _mode
        self._onnx_tta = os.environ.get("FATIGUE_ONNX_TTA", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if self.has_onnx:
            print(
                f"   ONNX: region={self._onnx_input_mode}, P(drowsy)=logit[{self._onnx_drowsy_idx}]"
                + (" + flip TTA" if self._onnx_tta else "")
            )

        self.frame_index = 0
        self.score_buffer = deque(maxlen=SCORE_BUFFER_MAXLEN)

    def compute_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_landmark_coords(self, landmarks, indices, w, h):
        coords = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                coords.append((lm.x * w, lm.y * h))
        return coords

    def compute_ear_from_coords(self, eye_coords):
        """EAR from six landmarks in MediaPipe order: (|p1-p5|+|p2-p4|) / (2|p0-p3|)."""
        if len(eye_coords) < 6:
            return EAR_OPEN_REF
        p = eye_coords
        a = self.compute_distance(p[1], p[5])
        b = self.compute_distance(p[2], p[4])
        c = self.compute_distance(p[0], p[3])
        if c <= 1e-6:
            return EAR_OPEN_REF
        return (a + b) / (2.0 * c)

    def compute_mar_mediapipe(self, mouth_coords):
        """MAR = vertical lip opening / mouth width (indices 13-14 vs 78-308)."""
        if len(mouth_coords) < 4:
            return 0.0
        v = self.compute_distance(mouth_coords[0], mouth_coords[1])
        h = self.compute_distance(mouth_coords[2], mouth_coords[3])
        if h <= 1e-6:
            return 0.0
        return v / h

    @staticmethod
    def _expand_bbox(x1, y1, x2, y2, fw, fh, scale=1.22):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1) * scale
        bh = (y2 - y1) * scale
        nx1 = int(cx - bw * 0.5)
        ny1 = int(cy - bh * 0.5)
        nx2 = int(cx + bw * 0.5)
        ny2 = int(cy + bh * 0.5)
        return max(0, nx1), max(0, ny1), min(fw, nx2), min(fh, ny2)

    def cascade_ear_continuous(self, face_roi_gray, face_w, face_h):
        """Smooth EAR-like proxy from eye boxes vs face area (no landmarks)."""
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        face_area = max(face_w * face_h, 1)
        if len(eyes) == 0:
            return EAR_CLOSED_REF
        total = sum(float(ew * eh) for (_, _, ew, eh) in eyes)
        ratio = total / face_area
        # Map area ratio into ~[EAR_CLOSED_REF, EAR_OPEN_REF]
        ear = float(np.clip(0.12 + ratio * 3.2, EAR_CLOSED_REF, 0.55))
        return ear

    def cascade_mar_continuous(self, mouths, face_roi_gray):
        """MAR-like openness in a scale compatible with mouth_score = mar / 0.4."""
        fh, fw = face_roi_gray.shape[:2]
        face_area = max(fh * fw, 1)
        if len(mouths) > 0:
            ratios = [(mw * mh) / face_area for (_, _, mw, mh) in mouths]
            r = max(ratios)
            return float(np.clip(0.15 + r * 2.8, 0.05, 0.95))
        lower = face_roi_gray[int(fh * 0.55) :, :]
        if lower.size == 0:
            return 0.12
        blur = cv2.GaussianBlur(lower, (5, 5), 0)
        std = float(np.std(blur))
        t = np.clip((std - 8.0) / 28.0, 0.0, 1.0)
        return float(np.clip(0.08 + 0.22 * t, 0.05, 0.38))

    def preprocess_frame_onnx(self, frame):
        """Whole frame → square tensor (matches val: Resize on full image from capture.py)."""
        if frame is None or frame.size == 0:
            return None
        img = cv2.resize(frame, (self._onnx_input_size, self._onnx_input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self._onnx_mean) / self._onnx_std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img.astype(np.float32)

    def preprocess_face_onnx(self, frame, x1, y1, x2, y2):
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
        face_img = cv2.resize(face_img, (self._onnx_input_size, self._onnx_input_size))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32) / 255.0
        face_img = (face_img - self._onnx_mean) / self._onnx_std
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, 0)
        return face_img.astype(np.float32)

    def _onnx_prob_drowsy(self, tensor_nchw: np.ndarray) -> float:
        outputs = self.session.run(None, {self._onnx_input_name: tensor_nchw})
        logits = np.asarray(outputs[0][0], dtype=np.float64).reshape(-1)
        if logits.size < 2:
            return 0.0
        z = logits - np.max(logits)
        e = np.exp(z)
        probs = e / np.sum(e)
        k = self._onnx_drowsy_idx
        if k < 0 or k >= probs.size:
            k = min(1, probs.size - 1)
        return float(probs[k])

    def get_model_score(self, frame, face_bbox):
        if not self.has_onnx:
            return 0.0
        x1, y1, x2, y2 = face_bbox
        try:
            if self._onnx_input_mode == "face":
                fh, fw = frame.shape[:2]
                x1, y1, x2, y2 = self._expand_bbox(x1, y1, x2, y2, fw, fh, scale=1.22)
                batch = self.preprocess_face_onnx(frame, x1, y1, x2, y2)
            else:
                batch = self.preprocess_frame_onnx(frame)
            if batch is None:
                return 0.0
            p = self._onnx_prob_drowsy(batch)
            if self._onnx_tta:
                flipped = np.ascontiguousarray(batch[:, :, :, ::-1])
                p = 0.5 * (p + self._onnx_prob_drowsy(flipped))
            return p
        except Exception:
            return 0.0

    def detect(self, frame):
        self.frame_index += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- MediaPipe path ----------
        
        # ---------- MediaPipe Tasks path ----------
        if self.face_mesh is not None:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = int(time.monotonic() * 1000)

            results = self.face_mesh.detect_for_video(
                mp_image, timestamp_ms
            )

            if not results.face_landmarks:
                self.score_buffer.clear()
                return "NO FACE", 0.0, {}

            landmarks = results.face_landmarks[0]

            left = self.get_landmark_coords(
                landmarks, LEFT_EYE_EAR, w, h
            )
            right = self.get_landmark_coords(
                landmarks, RIGHT_EYE_EAR, w, h
            )
            mouth_pts = self.get_landmark_coords(
                landmarks, MOUTH_MAR, w, h
            )

            left_ear = self.compute_ear_from_coords(left)
            right_ear = self.compute_ear_from_coords(right)
            ear = (left_ear + right_ear) / 2.0
            eye_state = (
    "EYES CLOSED" if ear < 0.20
    else "EYES OPEN" if ear > 0.25
    else "EYES UNCERTAIN"
)
            mar = self.compute_mar_mediapipe(mouth_pts)
            eye_source = "MediaPipe Tasks"

            xs = [
                landmarks[i].x * w
                for i in FACE_HULL_IDX
                if i < len(landmarks)
            ]
            ys = [
                landmarks[i].y * h
                for i in FACE_HULL_IDX
                if i < len(landmarks)
            ]

            if not xs:
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]

            pad = 12
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)
            face_bbox = (x1, y1, x2, y2)

        else:
            # ---------- Cascade fallback ----------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
            )
            if len(faces) == 0:
                self.score_buffer.clear()
                return "NO FACE", 0.0, {}

            fx, fy, fw, fh = faces[0]
            face_roi = gray[fy : fy + fh, fx : fx + fw]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            mouths = self.mouth_cascade.detectMultiScale(
                face_roi, scaleFactor=1.3, minNeighbors=8, minSize=(20, 20)
            )

            ear = self.cascade_ear_continuous(face_roi, fw, fh)
            mar = self.cascade_mar_continuous(mouths, face_roi)
            eye_source = "Cascade"
            face_bbox = (fx, fy, fx + fw, fy + fh)

        # ---------- Scores (continuous) ----------
        eye_score = (EAR_OPEN_REF - ear) / EYE_SCORE_DENOM
        eye_score = float(max(0.0, min(1.0, eye_score)))

        mouth_score = mar / MAR_SCALE
        mouth_score = float(max(0.0, min(1.0, mouth_score)))

        model_score = self.get_model_score(frame, face_bbox)
        model_score = float(max(0.0, min(1.0, model_score)))

        fatigue_score = W_EYE * eye_score + W_MOUTH * mouth_score + W_MODEL * model_score
        print(
    f"EAR={ear:.3f} | eye_score={eye_score:.3f} | "
    f"MAR={mar:.3f} | mouth_score={mouth_score:.3f} | "
    f"model_score={model_score:.3f} | combined={fatigue_score:.3f}"
)

        # Strong drowsy boosts (eyes very closed / yawning mouth)
        if ear < 0.22:
            fatigue_score += 0.25
        if ear < 0.18:
            fatigue_score += 0.35
        if mar > 0.35:
            fatigue_score += 0.2

        fatigue_score = float(max(0.0, min(1.0, fatigue_score)))

        # Pull down clearly alert frames (wide eyes, closed mouth) — tuned for standard EAR/MAR
        if ear > 0.33 and mar < 0.12:
            fatigue_score *= 0.42

        fatigue_score = float(max(0.0, min(1.0, fatigue_score)))

        # Temporal smoothing (reduces frame-to-frame flicker)
        self.score_buffer.append(fatigue_score)
        smoothed_score = sum(self.score_buffer) / len(self.score_buffer)
        fatigue_score_raw = fatigue_score
        fatigue_score = float(smoothed_score)
        # User-facing drowsiness: some camera/model setups yield an "alertness-like" blend
        # (high when you feel drowsy). Invert so label + % match subjective state.
        if FATIGUE_SCORE_INVERT:
            fatigue_score = 1.0 - fatigue_score
        fatigue_score = float(max(0.0, min(1.0, fatigue_score)))

        confidence = fatigue_score * 100.0

# Higher fatigue score means more signs of drowsiness.
        if confidence >= DISPLAY_DROWSY_BELOW_PCT:
           status = "DROWSY"
        else:
           status = "ALERT"

        features = {
            "ear": ear,
            "mar": mar,
            "eye_score": eye_score,
            "mouth_score": mouth_score,
            "model_score": model_score,
            "fatigue_score": fatigue_score,
            "fatigue_score_raw": fatigue_score_raw,
            "face_bbox": face_bbox,
            "eye_source": eye_source,
            "left_ear": left_ear,
"right_ear": right_ear,
"eye_state": eye_state,
        }

        if self.frame_index % max(1, DEBUG_PRINT_EVERY_N_FRAMES) == 0:
            print(f"EAR: {ear:.3f} | MAR: {mar:.3f} | Score: {fatigue_score:.2f}")

        return status, confidence, features


# ==================== VISUALIZATION ====================


def draw_status(frame, status, confidence, features):
    display_label = status

    if display_label == "ALERT":
        color = (0, 255, 0)
    elif display_label == "DROWSY":
        color = (0, 0, 255)
    else:
        color = COLOR_ALERT

    if "face_bbox" in features:
        x1, y1, x2, y2 = features["face_bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"{display_label} ({confidence:.0f}%)"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    ear = features.get("ear", 0.0)
    mar = features.get("mar", 0.0)
    es = features.get("eye_score", 0.0)
    ms = features.get("mouth_score", 0.0)
    mscore = features.get("model_score", 0.0)
    fs = features.get("fatigue_score", 0.0)
    src = features.get("eye_source", "")

    cv2.putText(
        frame,
        f"EAR {ear:.3f} eye_s {es:.2f} [{src}]",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )
    cv2.putText(
        frame,
        f"MAR {mar:.3f} mouth_s {ms:.2f} | model {mscore:.2f} | fatigue {fs:.2f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )
  
    left_ear = features.get("left_ear", 0.0)
    right_ear = features.get("right_ear", 0.0)
    eye_state = features.get("eye_state", "EYES UNCERTAIN")

    eye_color = (
        (0, 0, 255) if eye_state == "EYES CLOSED"
        else (0, 255, 0) if eye_state == "EYES OPEN"
        else (0, 165, 255)
    )

    cv2.putText(
        frame,
        f"{eye_state} | L: {left_ear:.3f} R: {right_ear:.3f}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        eye_color,
        2,
    )



# ==================== MAIN ====================


def main():
    global last_beep_time

    print("\n" + "=" * 75)
    print("Continuous fatigue scoring (EAR + MAR + model)")
    print("=" * 75)
    print(f"Weights: eye={W_EYE} mouth={W_MOUTH} model={W_MODEL}")
    print(
        f"Label: DROWSY (red) if display % < {DISPLAY_DROWSY_BELOW_PCT}; "
        f"ALERT (green) otherwise. Beep if % < {BEEP_BELOW_PCT}."
    )
    print(f"Score invert (only if needed): {FATIGUE_SCORE_INVERT}  — set FATIGUE_SCORE_INVERT=1 if labels still reversed")
    print("=" * 75 + "\n")

    cap = cv2.VideoCapture(0)
    detector = FatigueDetector()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        status, confidence, features = detector.detect(frame)
        draw_status(frame, status, confidence, features)

        fatigue_score = float(
            features.get("fatigue_score", confidence / 100.0)
            if features
            else confidence / 100.0
        )
        vision_data = {
            "fatigue_score": fatigue_score,
            "label": status,
            "confidence_pct": float(confidence),
        }
        try:
            with open(VISION_OUTPUT_JSON, "w", encoding="utf-8") as vf:
                json.dump(vision_data, vf)
        except OSError:
            pass

        current_time = time.time()
        if confidence >= 75.0 and (current_time - last_beep_time > 3):
            beep()
            last_beep_time = current_time

        cv2.imshow("Fatigue Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Detection stopped")


if __name__ == "__main__":
    main()
