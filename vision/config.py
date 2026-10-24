"""Configurable thresholds. Units are documented next to each field."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class VisionConfig:
    # --- EAR calibration (unitless ratio: vertical/horizontal eye opening) ---
    # Collect this many valid open-eye frames while the user looks at the camera.
    calibration_frames: int = 45
    # Relative to calibrated open-eye EAR. Closed if EAR < baseline * closed_ratio.
    closed_ratio: float = 0.62
    # Open if EAR > baseline * open_ratio. Band between the two is hysteresis.
    open_ratio: float = 0.80
    # Used only before calibration exists (typical webcam open EAR is ~0.32–0.38).
    default_open_ear: float = 0.34
    min_open_ear: float = 0.22
    max_open_ear: float = 0.48

    # --- Temporal eye logic (seconds) ---
    blink_max_s: float = 0.45
    prolonged_closure_s: float = 0.80
    drowsy_closure_s: float = 1.50
    drowsy_closure_looking_down_s: float = 2.50
    perclos_window_s: float = 30.0
    possible_fatigue_perclos: float = 0.25
    drowsy_perclos: float = 0.40

    # --- Head pose (degrees). Negative pitch = looking down toward keyboard. ---
    looking_down_pitch_deg: float = -12.0
    looking_down_landmark_ratio: float = 0.42
    # Downward pose shorter than this is treated as a glance; at/above it is head dropping.
    head_drop_min_s: float = 1.20

    # --- Mouth (MAR is unitless: inner-lip height / mouth width) ---
    # Mouth-open detection only — not a yawn by itself.
    mouth_open_mar: float = 0.28
    # Candidate yawn requires MAR at/above this for at least yawn_min_s.
    yawn_mar: float = 0.48
    yawn_min_s: float = 1.20
    # Short openings in this band are treated as speech / mouth movement.
    speech_mar_low: float = 0.16
    speech_mar_high: float = 0.42
    speech_max_open_s: float = 0.55
    mouth_history_s: float = 8.0

    # --- Fusion index (0–1 heuristic, NOT a calibrated probability) ---
    possible_fatigue_index: float = 0.40
    drowsy_index: float = 0.70
    yawn_index_boost: float = 0.18
    model_confirm_cap: float = 0.12

    # --- Alerts ---
    beep_cooldown_s: float = 4.0

    # --- Debug ---
    debug_every_n_frames: int = 30

    extra: dict = field(default_factory=dict)


def load_vision_config(path: Path | None = None) -> VisionConfig:
    cfg = VisionConfig()
    if path is None or not path.is_file():
        return cfg
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    allowed = {k for k in asdict(cfg) if k != "extra"}
    extra = {}
    for key, value in data.items():
        if key in allowed:
            setattr(cfg, key, type(getattr(cfg, key))(value))
        else:
            extra[key] = value
    cfg.extra = extra
    return cfg
