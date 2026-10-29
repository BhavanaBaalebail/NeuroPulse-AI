"""Calibrated eye-state with hysteresis, blink vs prolonged closure, unknowns."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vision.config import VisionConfig


@dataclass
class EyeObservation:
    left_ear: Optional[float]
    right_ear: Optional[float]
    combined_ear: Optional[float]
    face_detected: bool
    looking_down: bool
    pitch_deg: Optional[float]
    eye_state: str  # OPEN | CLOSED | UNKNOWN
    event: str  # none | blink | prolonged_closure
    closed_duration_s: float
    perclos: float
    calibrating: bool
    calibrated: bool
    open_ear_baseline: Optional[float]
    closed_threshold: float
    open_threshold: float
    reasons: list[str] = field(default_factory=list)
    eye_model_closed_probability: Optional[float] = None
    eye_model_assisted: bool = False
    eye_state_estimate: Optional[str] = None
    head_pose_status: str = "level"
    looking_down_duration_s: float = 0.0


class EyeStateTracker:
    def __init__(
        self,
        config: VisionConfig | None = None,
        calibration_path: Path | None = None,
        now_fn=None,
    ):
        self.config = config or VisionConfig()
        self.calibration_path = calibration_path
        self._now = now_fn or (lambda: None)
        self._samples: list[float] = []
        self._state = "UNKNOWN"
        self._closed_started_at: Optional[float] = None
        self._last_event = "none"
        self._window: deque[tuple[float, str]] = deque()
        self.open_ear_baseline: Optional[float] = None
        if calibration_path and calibration_path.is_file():
            self._load(calibration_path)

    def _clock(self, timestamp: Optional[float]) -> float:
        if timestamp is not None:
            return float(timestamp)
        value = self._now()
        if value is None:
            raise ValueError("timestamp is required when now_fn is not set")
        return float(value)

    def start_calibration(self) -> None:
        self._samples = []
        self.open_ear_baseline = None

    def is_calibrating(self) -> bool:
        return self.open_ear_baseline is None and len(self._samples) < self.config.calibration_frames

    def _thresholds(self) -> tuple[float, float]:
        baseline = self.open_ear_baseline or self.config.default_open_ear
        closed_t = baseline * self.config.closed_ratio
        open_t = baseline * self.config.open_ratio
        if open_t <= closed_t:
            open_t = closed_t + 0.02
        return closed_t, open_t

    def _classify(self, ear: Optional[float]) -> str:
        if ear is None:
            return "UNKNOWN"
        closed_t, open_t = self._thresholds()
        if self._state == "CLOSED":
            if ear > open_t:
                return "OPEN"
            if ear < closed_t:
                return "CLOSED"
            return "CLOSED"
        if self._state == "OPEN":
            if ear < closed_t:
                return "CLOSED"
            if ear > open_t:
                return "OPEN"
            return "OPEN"
        if ear < closed_t:
            return "CLOSED"
        if ear > open_t:
            return "OPEN"
        return "UNKNOWN"

    def update(
        self,
        left_ear: Optional[float],
        right_ear: Optional[float],
        combined_ear: Optional[float],
        face_detected: bool,
        looking_down: bool = False,
        pitch_deg: Optional[float] = None,
        timestamp: Optional[float] = None,
        eye_model_closed_probability: Optional[float] = None,
        head_pose_status: str = "level",
        looking_down_duration_s: float = 0.0,
    ) -> EyeObservation:
        t = self._clock(timestamp)
        reasons: list[str] = []
        calibrating = False

        if not face_detected:
            combined_ear = None
            left_ear = None
            right_ear = None

        if combined_ear is not None and not looking_down and self.open_ear_baseline is None:
            # Ignore likely-closed samples so a blink does not drag the open-eye baseline down.
            if combined_ear >= self.config.default_open_ear * 0.85:
                self._samples.append(combined_ear)
            calibrating = True
            need = self.config.calibration_frames
            if len(self._samples) >= need:
                ordered = sorted(self._samples[-need:])
                mid = ordered[len(ordered) // 2]
                mid = min(self.config.max_open_ear, max(self.config.min_open_ear, mid))
                self.open_ear_baseline = float(mid)
                calibrating = False
                self._save()
            reasons.append("calibrating_open_eyes")

        new_state = self._classify(combined_ear if face_detected else None)
        eye_state_estimate = new_state
        model_assisted = False
        if (
            face_detected
            and combined_ear is not None
            and eye_model_closed_probability is not None
            and self._thresholds()[0] <= combined_ear <= self._thresholds()[1]
            and eye_model_closed_probability >= self.config.eye_model_closed_threshold
            and new_state != "CLOSED"
        ):
            new_state = "CLOSED"
            model_assisted = True
        event = "none"
        closed_duration = 0.0

        if new_state == "CLOSED":
            if self._closed_started_at is None:
                self._closed_started_at = t
            closed_duration = t - self._closed_started_at
        else:
            if self._state == "CLOSED" and self._closed_started_at is not None:
                dur = t - self._closed_started_at
                if dur <= self.config.blink_max_s:
                    event = "blink"
                elif dur >= self.config.prolonged_closure_s:
                    event = "prolonged_closure"
            self._closed_started_at = None

        if new_state == "CLOSED" and closed_duration >= self.config.prolonged_closure_s:
            event = "prolonged_closure"

        self._state = new_state
        self._last_event = event
        self._window.append((t, new_state))
        cutoff = t - self.config.perclos_window_s
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

        known = [s for _, s in self._window if s != "UNKNOWN"]
        closed_n = sum(1 for s in known if s == "CLOSED")
        perclos = (closed_n / len(known)) if known else 0.0

        closed_t, open_t = self._thresholds()
        return EyeObservation(
            left_ear=left_ear,
            right_ear=right_ear,
            combined_ear=combined_ear,
            face_detected=face_detected,
            looking_down=looking_down,
            pitch_deg=pitch_deg,
            eye_state=new_state,
            event=event,
            closed_duration_s=closed_duration,
            perclos=perclos,
            calibrating=calibrating or self.is_calibrating(),
            calibrated=self.open_ear_baseline is not None,
            open_ear_baseline=self.open_ear_baseline,
            closed_threshold=closed_t,
            open_threshold=open_t,
            reasons=reasons,
            eye_model_closed_probability=eye_model_closed_probability,
            eye_model_assisted=model_assisted,
            eye_state_estimate=eye_state_estimate,
            head_pose_status=head_pose_status,
            looking_down_duration_s=looking_down_duration_s,
        )

    def _save(self) -> None:
        if not self.calibration_path or self.open_ear_baseline is None:
            return
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "open_ear_baseline": self.open_ear_baseline,
            "closed_ratio": self.config.closed_ratio,
            "open_ratio": self.config.open_ratio,
        }
        with open(self.calibration_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            value = float(data["open_ear_baseline"])
            if self.config.min_open_ear <= value <= self.config.max_open_ear:
                self.open_ear_baseline = value
                self._samples = [value] * self.config.calibration_frames
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.open_ear_baseline = None
