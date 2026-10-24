"""ONNX face scorer. Failures return None instead of a silent 0."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


class OnnxFaceScorer:
    def __init__(self, onnx_path: Path, config_paths: list[Path]):
        self.path = onnx_path
        self.available = False
        self.last_error: Optional[str] = None
        self.session = None
        self.input_name = "image"
        self.input_size = 64
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.drowsy_idx = int(os.environ.get("FATIGUE_ONNX_DROWSY_CLASS", "1"))
        self.class_names: list[str] = ["notdrowsy", "drowsy"]
        self.backbone = "unknown"
        self.input_mode = os.environ.get("FATIGUE_ONNX_INPUT", "face").strip().lower()
        if self.input_mode not in ("full", "face"):
            self.input_mode = "face"
        self.tta = os.environ.get("FATIGUE_ONNX_TTA", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._load_config(config_paths)
        self._load_session()

    def _load_config(self, config_paths: list[Path]) -> None:
        for cfg_path in config_paths:
            if not cfg_path.is_file():
                continue
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    vc = json.load(f)
                self.input_size = int(vc.get("input_size", 64))
                self.mean = np.array(vc["mean"], dtype=np.float32).reshape(1, 1, 3)
                self.std = np.array(vc["std"], dtype=np.float32).reshape(1, 1, 3)
                if "onnx_drowsy_class_index" in vc:
                    self.drowsy_idx = int(vc["onnx_drowsy_class_index"])
                self.class_names = list(vc.get("class_names", self.class_names))
                self.backbone = str(vc.get("backbone", self.backbone))
                return
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                self.last_error = f"config {cfg_path.name}: {exc}"

    def _load_session(self) -> None:
        if ort is None:
            self.last_error = "onnxruntime is not installed"
            return
        if not self.path.is_file():
            self.last_error = f"missing ONNX file: {self.path}"
            return
        try:
            self.session = ort.InferenceSession(
                str(self.path),
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.available = True
            self.last_error = None
        except Exception as exc:
            self.available = False
            self.session = None
            self.last_error = str(exc)

    @staticmethod
    def expand_bbox(x1, y1, x2, y2, fw, fh, scale=1.22):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1) * scale
        bh = (y2 - y1) * scale
        nx1 = int(cx - bw * 0.5)
        ny1 = int(cy - bh * 0.5)
        nx2 = int(cx + bw * 0.5)
        ny2 = int(cy + bh * 0.5)
        return max(0, nx1), max(0, ny1), min(fw, nx2), min(fh, ny2)

    def preprocess(self, bgr) -> Optional[np.ndarray]:
        if bgr is None or getattr(bgr, "size", 0) == 0:
            self.last_error = "empty image for ONNX"
            return None
        import cv2

        img = cv2.resize(bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, 0).astype(np.float32)

    def _prob_drowsy(self, tensor_nchw: np.ndarray) -> Optional[float]:
        if self.session is None:
            return None
        try:
            outputs = self.session.run(None, {self.input_name: tensor_nchw})
        except Exception as exc:
            self.last_error = f"onnx_run: {exc}"
            return None
        logits = np.asarray(outputs[0][0], dtype=np.float64).reshape(-1)
        if logits.size < 2 or not np.all(np.isfinite(logits)):
            self.last_error = f"invalid ONNX logits: {logits}"
            return None
        z = logits - np.max(logits)
        e = np.exp(z)
        probs = e / np.sum(e)
        k = self.drowsy_idx
        if k < 0 or k >= probs.size:
            self.last_error = f"drowsy class index {k} out of range for {probs.size} outputs"
            return None
        self.last_error = None
        return float(probs[k])

    def score(self, frame, face_bbox) -> Optional[float]:
        if not self.available:
            return None
        x1, y1, x2, y2 = face_bbox
        try:
            if self.input_mode == "face":
                fh, fw = frame.shape[:2]
                x1, y1, x2, y2 = self.expand_bbox(x1, y1, x2, y2, fw, fh, scale=1.22)
                crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
                batch = self.preprocess(crop)
            else:
                batch = self.preprocess(frame)
        except Exception as exc:
            self.last_error = f"preprocess: {exc}"
            return None
        if batch is None:
            return None
        p = self._prob_drowsy(batch)
        if p is None:
            return None
        if self.tta:
            flipped = np.ascontiguousarray(batch[:, :, :, ::-1])
            p2 = self._prob_drowsy(flipped)
            if p2 is None:
                return None
            p = 0.5 * (p + p2)
        return float(max(0.0, min(1.0, p)))
