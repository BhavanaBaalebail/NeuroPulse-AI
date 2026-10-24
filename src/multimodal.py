"""Decision-level fusion of vision JSON and behavioural XGBoost outputs.

Does not merge feature vectors, average incompatible scores, or invent
confidence / recovery values. Vision labels stay ALERT/DROWSY; behaviour
labels stay Alert/Mild Fatigue/High Fatigue; the fused display labels are
Alert, Possible Fatigue, High Concern, or Unknown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

VISION_STALE_S = 2.0
FINAL_ALERT = "Alert"
FINAL_POSSIBLE = "Possible Fatigue"
FINAL_HIGH = "High Concern"
FINAL_UNKNOWN = "Unknown"

_BEHAVIOUR_ALERT = "Alert"
_BEHAVIOUR_MILD = "Mild Fatigue"
_BEHAVIOUR_HIGH = "High Fatigue"


@dataclass
class StreamReading:
    available: bool
    vote: Optional[str]  # vision: alert|drowsy; behaviour: alert|mild|high
    status: str
    reasons: list[str] = field(default_factory=list)
    quality: list[str] = field(default_factory=list)
    activity_label: str = ""
    eye_state: Optional[str] = None
    ear: Optional[float] = None
    mar: Optional[float] = None
    perclos: Optional[float] = None
    model_score: Optional[float] = None
    age_s: Optional[float] = None
    behaviour_label: Optional[str] = None
    probabilities: Optional[dict] = None


@dataclass
class FusedAssessment:
    final: str
    reasons: list[str]
    vision: StreamReading
    behaviour: StreamReading
    conflict: bool
    confidence: None = None  # never invented
    limitations: list[str] = field(default_factory=list)
    explanation: str = ""


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def parse_vision_payload(
    data: Optional[dict],
    *,
    now: float,
    process_running: Optional[bool] = None,
    stale_s: float = VISION_STALE_S,
) -> StreamReading:
    quality: list[str] = []
    reasons: list[str] = []
    if not data:
        quality.append("vision_json_missing")
        return StreamReading(
            available=False,
            vote=None,
            status="unavailable",
            reasons=["vision_output_missing"],
            quality=quality,
        )

    if process_running is False:
        quality.append("camera_stopped")
        return StreamReading(
            available=False,
            vote=None,
            status="stopped",
            reasons=["camera_process_not_running"],
            quality=quality,
            activity_label=str(data.get("activity_label") or ""),
            eye_state=data.get("eye_state"),
        )

    updated_at = _as_float(data.get("updated_at"))
    if updated_at is None:
        quality.append("missing_timestamp")
        return StreamReading(
            available=False,
            vote=None,
            status="stale",
            reasons=["vision_timestamp_missing_treated_as_stale"],
            quality=quality,
        )

    age_s = max(0.0, float(now) - updated_at)
    if age_s > stale_s:
        quality.append("stale_vision")
        return StreamReading(
            available=False,
            vote=None,
            status="stale",
            reasons=[f"vision_older_than_{stale_s:.1f}s"],
            quality=quality,
            age_s=age_s,
            activity_label=str(data.get("activity_label") or ""),
            eye_state=data.get("eye_state"),
        )

    raw_reasons = [str(r) for r in (data.get("reasons") or [])]
    activity = str(data.get("activity_label") or "").strip()
    eye_state = data.get("eye_state")
    label = str(data.get("label") or "").strip().upper()

    no_face = activity.lower() == "no face" or any("no_face" in r for r in raw_reasons)
    if no_face:
        quality.append("no_face")
        return StreamReading(
            available=False,
            vote=None,
            status="no_face",
            reasons=raw_reasons or ["no_face_or_lost_tracking"],
            quality=quality,
            activity_label=activity,
            eye_state=eye_state,
            age_s=age_s,
            ear=_as_float(data.get("ear")),
            mar=_as_float(data.get("mar")),
        )

    if activity.lower() == "calibrating" or any("calibrat" in r for r in raw_reasons):
        quality.append("calibrating")
        return StreamReading(
            available=False,
            vote=None,
            status="calibrating",
            reasons=raw_reasons or ["calibrating_open_eye_baseline"],
            quality=quality,
            activity_label=activity,
            eye_state=eye_state,
            age_s=age_s,
        )

    if str(eye_state or "").upper() == "UNKNOWN" or activity.lower() == "uncertain tracking":
        quality.append("uncertain_tracking")
        return StreamReading(
            available=False,
            vote=None,
            status="uncertain",
            reasons=raw_reasons or ["eye_measurements_uncertain"],
            quality=quality,
            activity_label=activity,
            eye_state=eye_state,
            age_s=age_s,
        )

    if label == "DROWSY":
        vote = "drowsy"
        status = "drowsy"
    elif label == "ALERT":
        vote = "alert"
        status = "alert"
    else:
        quality.append("unknown_vision_label")
        return StreamReading(
            available=False,
            vote=None,
            status="unavailable",
            reasons=raw_reasons + [f"unsupported_vision_label_{label}"],
            quality=quality,
            activity_label=activity,
            age_s=age_s,
        )

    quality.append("fresh")
    reasons = list(raw_reasons)
    if activity:
        reasons.append(f"activity:{activity}")
    return StreamReading(
        available=True,
        vote=vote,
        status=status,
        reasons=reasons,
        quality=quality,
        activity_label=activity,
        eye_state=eye_state,
        ear=_as_float(data.get("ear")),
        mar=_as_float(data.get("mar")),
        perclos=_as_float(data.get("perclos")),
        model_score=_as_float(data.get("model_score")),
        age_s=age_s,
    )


def read_vision_file(
    path: Path | str,
    *,
    now: float,
    process_running: Optional[bool] = None,
    stale_s: float = VISION_STALE_S,
) -> StreamReading:
    file_path = Path(path)
    if not file_path.is_file():
        return parse_vision_payload(None, now=now, process_running=process_running, stale_s=stale_s)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return StreamReading(
            available=False,
            vote=None,
            status="unavailable",
            reasons=["vision_json_unreadable"],
            quality=["vision_json_unreadable"],
        )
    if not isinstance(data, dict):
        return parse_vision_payload(None, now=now, process_running=process_running, stale_s=stale_s)
    return parse_vision_payload(
        data, now=now, process_running=process_running, stale_s=stale_s
    )


def behaviour_from_snapshot(
    snapshot: Optional[dict],
    prediction: Optional[dict] = None,
) -> StreamReading:
    if not snapshot:
        return StreamReading(
            available=False,
            vote=None,
            status="unavailable",
            reasons=["behaviour_snapshot_missing"],
            quality=["behaviour_missing"],
        )
    status = str(snapshot.get("status") or "unavailable")
    notes = [str(n) for n in (snapshot.get("notes") or [])]
    if not snapshot.get("ready_for_prediction"):
        quality = ["insufficient_behaviour" if status == "insufficient" else status]
        return StreamReading(
            available=False,
            vote=None,
            status=status,
            reasons=notes or [f"behaviour_{status}"],
            quality=quality,
        )
    if not prediction or "label" not in prediction:
        return StreamReading(
            available=False,
            vote=None,
            status="unavailable",
            reasons=["behaviour_prediction_missing"],
            quality=["behaviour_prediction_missing"],
        )
    label = str(prediction["label"])
    if label == _BEHAVIOUR_HIGH:
        vote = "high"
    elif label == _BEHAVIOUR_MILD:
        vote = "mild"
    elif label == _BEHAVIOUR_ALERT:
        vote = "alert"
    else:
        return StreamReading(
            available=False,
            vote=None,
            status="unavailable",
            reasons=[f"unsupported_behaviour_label_{label}"],
            quality=["unsupported_behaviour_label"],
            behaviour_label=label,
        )
    return StreamReading(
        available=True,
        vote=vote,
        status=status,
        reasons=notes,
        quality=["ready"],
        behaviour_label=label,
        probabilities=prediction.get("probabilities"),
    )


def stream_display_label(reading: StreamReading, *, kind: str) -> str:
    """Visible stream title. Never substitute LIVE for a predicted class."""
    if kind == "behaviour":
        return reading.behaviour_label or reading.status
    if reading.vote == "drowsy":
        return "DROWSY"
    if reading.vote == "alert":
        return "ALERT"
    return reading.status


def explain_fusion(fused: FusedAssessment) -> str:
    """One short sentence for the dashboard. Does not invent scores."""
    reasons = fused.reasons
    if fused.final == FINAL_UNKNOWN:
        return "Monitoring unavailable — check sensor status."
    if "both_streams_alert" in reasons:
        return "Both signals indicate normal alertness."
    if "vision_drowsy_not_cancelled_by_behaviour_alert" in reasons:
        return "Sustained eye closure detected despite normal behavioural activity."
    if "behaviour_high_with_vision_alert_conflict" in reasons:
        return "Behaviour suggests fatigue despite an alert vision signal."
    if "behaviour_mild_with_vision_alert_conflict" in reasons:
        return "Behaviour suggests fatigue despite an alert vision signal."
    if "both_streams_indicate_concern" in reasons:
        return "Both vision and behaviour indicate elevated fatigue risk."
    if "vision_only_drowsy_behaviour_unavailable" in reasons:
        return "Sustained eye closure detected. Behavioural data is not ready yet."
    if "behaviour_only_high_vision_unavailable" in reasons:
        return "Behaviour indicates high fatigue. Vision is unavailable."
    if "behaviour_only_mild_vision_unavailable" in reasons:
        return "Behaviour suggests fatigue. Vision is unavailable."
    if "vision_alert_only" in reasons:
        return "Vision indicates alertness. Behavioural capture is still collecting."
    if "behaviour_alert_only" in reasons:
        return "Behaviour indicates alertness. Vision is unavailable."
    if fused.final == FINAL_HIGH:
        return "High concern from available sensor evidence."
    if fused.final == FINAL_POSSIBLE:
        return "Behaviour suggests fatigue."
    return "Both signals indicate normal alertness."


def _finish(fused: FusedAssessment) -> FusedAssessment:
    fused.explanation = explain_fusion(fused)
    return fused


def combine(vision: StreamReading, behaviour: StreamReading) -> FusedAssessment:
    limitations = [
        "Decision-level fusion only; XGBoost and vision CNN/landmarks are not jointly trained.",
        "Fused output is not a calibrated probability and is not a medical assessment.",
        "Vision DROWSY is landmark-temporal (sustained closure / PERCLOS while closed); "
        "brief blinks, talking, and looking down stay ALERT in the vision pipeline.",
        "Behavioural Mild/High can fire from keyboard/mouse proxies even when eyes are open.",
    ]
    reasons: list[str] = []
    conflict = False

    if not vision.available and not behaviour.available:
        reasons.extend(vision.reasons[:3])
        reasons.extend(behaviour.reasons[:3])
        reasons.append("both_streams_unavailable")
        return _finish(FusedAssessment(
            final=FINAL_UNKNOWN,
            reasons=reasons,
            vision=vision,
            behaviour=behaviour,
            conflict=False,
            limitations=limitations,
        ))

    if vision.available and vision.vote == "drowsy":
        reasons.append("vision_drowsy")
        reasons.extend(r for r in vision.reasons if "sustained" in r or "perclos" in r or "closure" in r)
        if behaviour.available and behaviour.vote == "alert":
            conflict = True
            reasons.append("vision_drowsy_not_cancelled_by_behaviour_alert")
        elif behaviour.available and behaviour.vote in {"mild", "high"}:
            reasons.append("both_streams_indicate_concern")
        elif not behaviour.available:
            reasons.append("vision_only_drowsy_behaviour_unavailable")
        return _finish(FusedAssessment(
            final=FINAL_HIGH,
            reasons=reasons,
            vision=vision,
            behaviour=behaviour,
            conflict=conflict,
            limitations=limitations,
        ))

    if behaviour.available and behaviour.vote == "high":
        reasons.append("behaviour_high_fatigue")
        if vision.available and vision.vote == "alert":
            conflict = True
            reasons.append("behaviour_high_with_vision_alert_conflict")
        elif not vision.available:
            reasons.append("behaviour_only_high_vision_unavailable")
        return _finish(FusedAssessment(
            final=FINAL_HIGH,
            reasons=reasons,
            vision=vision,
            behaviour=behaviour,
            conflict=conflict,
            limitations=limitations,
        ))

    if behaviour.available and behaviour.vote == "mild":
        reasons.append("behaviour_mild_fatigue")
        if vision.available and vision.vote == "alert":
            conflict = True
            reasons.append("behaviour_mild_with_vision_alert_conflict")
        elif not vision.available:
            reasons.append("behaviour_only_mild_vision_unavailable")
        return _finish(FusedAssessment(
            final=FINAL_POSSIBLE,
            reasons=reasons,
            vision=vision,
            behaviour=behaviour,
            conflict=conflict,
            limitations=limitations,
        ))

    if vision.available and vision.vote == "alert" and behaviour.available and behaviour.vote == "alert":
        reasons.append("both_streams_alert")
        if vision.activity_label:
            reasons.append(f"vision_activity:{vision.activity_label}")
        return _finish(FusedAssessment(
            final=FINAL_ALERT,
            reasons=reasons,
            vision=vision,
            behaviour=behaviour,
            conflict=False,
            limitations=limitations,
        ))

    if vision.available and vision.vote == "alert":
        reasons.append("vision_alert_only")
        if vision.activity_label:
            reasons.append(f"vision_activity:{vision.activity_label}")
        reasons.append("behaviour_unavailable")
        return _finish(FusedAssessment(
            final=FINAL_ALERT,
            reasons=reasons,
            vision=vision,
            behaviour=behaviour,
            conflict=False,
            limitations=limitations,
        ))

    if behaviour.available and behaviour.vote == "alert":
        reasons.append("behaviour_alert_only")
        reasons.append("vision_unavailable")
        return _finish(FusedAssessment(
            final=FINAL_ALERT,
            reasons=reasons,
            vision=vision,
            behaviour=behaviour,
            conflict=False,
            limitations=limitations,
        ))

    reasons.append("unhandled_combination_unknown")
    return _finish(FusedAssessment(
        final=FINAL_UNKNOWN,
        reasons=reasons,
        vision=vision,
        behaviour=behaviour,
        conflict=False,
        limitations=limitations,
    ))


def stop_vision_process(proc, *, wait_s: float = 1.0) -> None:
    if proc is None:
        return
    poll = getattr(proc, "poll", None)
    if callable(poll) and poll() is not None:
        return
    terminate = getattr(proc, "terminate", None)
    if callable(terminate):
        terminate()
    wait = getattr(proc, "wait", None)
    if callable(wait):
        try:
            wait(timeout=wait_s)
        except Exception:
            kill = getattr(proc, "kill", None)
            if callable(kill):
                kill()


def sync_vision_process_lifetime(proc, *, wanted: bool):
    """Stop the OpenCV app.py process when the mode no longer needs the camera."""
    if wanted:
        return proc
    stop_vision_process(proc)
    return None


def vision_process_wanted(app_mode: str) -> bool:
    return app_mode in {"Vision Mode", "Multimodal Mode"}


def vision_process_alive(proc) -> bool:
    if proc is None:
        return False
    poll = getattr(proc, "poll", None)
    if not callable(poll):
        return False
    return poll() is None


def start_vision_subprocess(
    proc,
    *,
    executable: str,
    app_path: Path | str,
    cwd: Path | str,
    popen=None,
) -> tuple[Any, dict]:
    """Reuse a live app.py process; never spawn a second copy."""
    if vision_process_alive(proc):
        return proc, {"ok": True, "already_running": True, "error": None}
    path = Path(app_path)
    if not path.is_file():
        return None, {
            "ok": False,
            "already_running": False,
            "error": f"Could not find app.py at {path}",
        }
    launcher = popen
    if launcher is None:
        import subprocess

        launcher = subprocess.Popen
    try:
        new_proc = launcher([executable, str(path)], cwd=str(cwd))
    except Exception as exc:
        return None, {"ok": False, "already_running": False, "error": str(exc)}
    return new_proc, {"ok": True, "already_running": False, "error": None}


def start_multimodal_monitoring(
    monitor,
    proc,
    *,
    executable: str,
    app_path: Path | str,
    cwd: Path | str,
    popen=None,
    enable_listeners: bool = True,
) -> dict:
    new_proc, vision = start_vision_subprocess(
        proc, executable=executable, app_path=app_path, cwd=cwd, popen=popen
    )
    behaviour = monitor.start(enable_listeners=enable_listeners)
    vision_ok = bool(vision.get("ok"))
    behaviour_ok = bool(behaviour.get("ok"))
    return {
        "proc": new_proc,
        "vision": vision,
        "behaviour": behaviour,
        "ok": vision_ok and behaviour_ok,
        "partial": vision_ok != behaviour_ok,
    }


def stop_multimodal_monitoring(monitor, proc, *, wait_s: float = 1.0):
    if monitor is not None:
        monitor.stop()
    stop_vision_process(proc, wait_s=wait_s)
    return None


CHART_LEVEL = {
    FINAL_ALERT: 0,
    FINAL_POSSIBLE: 1,
    FINAL_HIGH: 2,
}


def append_state_log(log: list[dict], *, now: float, fused: FusedAssessment, max_rows: int = 40) -> list[dict]:
    entry = {
        "t": now,
        "final": fused.final,
        "vision": fused.vision.status,
        "behaviour": fused.behaviour.behaviour_label or fused.behaviour.status,
        "explanation": fused.explanation,
    }
    if log:
        prev = log[-1]
        if (
            prev.get("final") == entry["final"]
            and prev.get("vision") == entry["vision"]
            and prev.get("behaviour") == entry["behaviour"]
        ):
            return log
    return (list(log) + [entry])[-max_rows:]


def append_chart_sample(
    samples: list[dict], *, now: float, fused: FusedAssessment, max_n: int = 120
) -> list[dict]:
    level = CHART_LEVEL.get(fused.final)
    if level is None:
        return samples
    return (list(samples) + [{"t": now, "level": level, "label": fused.final}])[-max_n:]
