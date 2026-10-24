"""Explainable temporal fusion. The displayed index is not a calibrated probability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vision.config import VisionConfig
from vision.eye_state import EyeObservation
from vision.mouth_state import MouthObservation


@dataclass
class FusionResult:
    state: str  # ALERT | DROWSY  (only primary fatigue states)
    index: float  # 0–1 heuristic drowsiness index
    reasons: list[str] = field(default_factory=list)
    used_face_model: bool = False
    activity_label: str = ""


HEAD_ACTIVITY = {
    "glance_down": "Glancing down",
    "head_dropping": "Head dropping",
}


def describe_activity(eye: EyeObservation, mouth: MouthObservation, head_pose: str) -> str:
    if not eye.face_detected:
        return "No face"
    if eye.calibrating:
        return "Calibrating"
    if eye.eye_state == "UNKNOWN":
        return "Uncertain tracking"
    if head_pose in HEAD_ACTIVITY:
        return HEAD_ACTIVITY[head_pose]
    if eye.event == "blink":
        return "Blinking"
    if mouth.mouth_class == "speaking":
        return "Talking"
    if mouth.candidate_yawn or mouth.recent_yawn:
        return "Drowsy-possible yawn"
    return ""


def fuse(
    eye: EyeObservation,
    mouth: MouthObservation,
    model_score: Optional[float],
    config: VisionConfig | None = None,
    model_ok: bool = False,
) -> FusionResult:
    cfg = config or VisionConfig()
    reasons: list[str] = []

    if not eye.face_detected or eye.eye_state == "UNKNOWN":
        if not eye.face_detected:
            reasons.append("no_face_or_lost_tracking")
        else:
            reasons.append("eye_measurements_uncertain")
        activity = describe_activity(eye, mouth, "unknown")
        return FusionResult(state="ALERT", index=0.0, reasons=reasons, activity_label=activity)

    if eye.calibrating:
        reasons.append("calibrating_open_eye_baseline")
        return FusionResult(
            state="ALERT",
            index=0.05,
            reasons=reasons,
            activity_label="Calibrating",
        )

    index = 0.0
    looking_down = bool(eye.looking_down)
    down_s = float(getattr(eye, "looking_down_duration_s", 0.0) or 0.0)
    head_pose = getattr(eye, "head_pose_status", None) or "level"
    if looking_down:
        if head_pose == "head_dropping" or down_s >= cfg.head_drop_min_s:
            head_pose = "head_dropping"
        elif head_pose != "head_dropping":
            head_pose = "glance_down"
    elif head_pose not in ("unknown", "level"):
        head_pose = "level"

    is_glance = head_pose == "glance_down"
    is_drop = head_pose == "head_dropping"
    if is_glance:
        reasons.append("looking_down_brief_glance")
    elif is_drop:
        reasons.append("sustained_head_dropping")

    closure_limit = (
        cfg.drowsy_closure_looking_down_s if is_glance else cfg.drowsy_closure_s
    )
    sustained_close = eye.eye_state == "CLOSED" and eye.closed_duration_s >= closure_limit
    prolonged = eye.eye_state == "CLOSED" and eye.closed_duration_s >= cfg.prolonged_closure_s

    if eye.event == "blink":
        reasons.append("normal_blink")

    if sustained_close and not is_glance:
        index = max(index, 0.88)
        reasons.append(f"sustained_eye_closure_{eye.closed_duration_s:.2f}s")
    elif prolonged and is_glance:
        reasons.append(f"eyes_appear_closed_while_looking_down_{eye.closed_duration_s:.2f}s")
    elif prolonged:
        index = max(index, 0.55)
        reasons.append(f"prolonged_eye_closure_{eye.closed_duration_s:.2f}s")

    # PERCLOS is a rolling history; it must not keep DROWSY/POSSIBLE after the eyes reopen.
    if eye.eye_state == "CLOSED" and not is_glance:
        if eye.perclos >= cfg.drowsy_perclos:
            index = max(index, 0.82)
            reasons.append(f"high_eye_closure_fraction_{eye.perclos:.2f}_over_{cfg.perclos_window_s:.0f}s")
        elif eye.perclos >= cfg.possible_fatigue_perclos:
            index = max(index, 0.48)
            reasons.append(f"elevated_eye_closure_fraction_{eye.perclos:.2f}")

    if is_drop:
        reasons.append("head_dropping_not_counted_as_drowsiness_alone")

    if mouth.mouth_class == "speaking":
        reasons.append("possible_speech_ignored_for_fatigue")
    elif mouth.mouth_class == "open":
        reasons.append("mouth_open_not_a_yawn")
    elif mouth.candidate_yawn or mouth.recent_yawn:
        index = min(1.0, index + cfg.yawn_index_boost)
        reasons.append("candidate_yawn_from_sustained_mouth_opening")
        reasons.append("mar_cannot_confirm_speech_versus_yawn")

    used_model = False
    if model_ok and model_score is not None and index >= cfg.possible_fatigue_index:
        # Face CNN may only confirm an already elevated landmark state.
        contrib = min(cfg.model_confirm_cap, max(0.0, model_score) * cfg.model_confirm_cap)
        if contrib > 0.01:
            index = min(1.0, index + contrib)
            used_model = True
            reasons.append("face_model_used_as_confirmation_only")
    elif model_ok and model_score is not None:
        reasons.append("face_model_ignored_while_landmarks_are_alert")
    elif not model_ok:
        reasons.append("face_model_unavailable")

    if not reasons:
        reasons.append("eyes_open_stable")

    # Primary states are only ALERT / DROWSY. Short closures, glances, talking,
    # and head motion stay ALERT. DROWSY needs current sustained eye closure
    # (or high PERCLOS while the eyes are still closed), not head pose alone.
    drowsy_from_eyes = sustained_close and not is_glance
    drowsy_from_perclos = (
        eye.eye_state == "CLOSED"
        and not is_glance
        and eye.perclos >= cfg.drowsy_perclos
    )
    if drowsy_from_eyes or drowsy_from_perclos:
        state = "DROWSY"
        index = max(index, cfg.drowsy_index)
    else:
        state = "ALERT"
        index = min(index, 0.25)

    return FusionResult(
        state=state,
        index=float(max(0.0, min(1.0, index))),
        reasons=reasons,
        used_face_model=used_model,
        activity_label=describe_activity(eye, mouth, head_pose),
    )
