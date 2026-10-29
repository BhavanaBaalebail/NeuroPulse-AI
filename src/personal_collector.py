"""Personal behaviour session store — aggregate features only.

Reuses LiveBehaviourMonitor snapshots. Never stores characters, clipboard,
passwords, or raw key codes. Does not train or overwrite production models.
"""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from logger import FEATURE_NAMES, LIVE_MIN_CLICKS, LIVE_MIN_KEYS, LiveBehaviourMonitor

ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = ROOT / "data" / "personal_behaviour_sessions.csv"
SYNTHETIC_CSV = ROOT / "data" / "personal_behaviour_synthetic.csv"

PERSONAL_WINDOW_SECONDS = 8 * 3600.0
PERSONAL_WARMUP_SECONDS = 20.0
MIN_REAL_FOR_SYNTHETIC = 8
SLEEPINESS_MIN = 1
SLEEPINESS_MAX = 9

REAL_COLUMNS = [
    "session_id",
    "saved_at",
    "started_at",
    *FEATURE_NAMES,
    "quality_status",
    "quality_ready",
    "quality_notes",
    "key_count",
    "backspace_count",
    "mouse_move_count",
    "click_count",
    "sleepiness_rating",
    "is_synthetic",
    "source_session_id",
]


def make_personal_monitor(now_fn=None) -> LiveBehaviourMonitor:
    """Same feature formulas as live capture; window covers a full recording session."""
    return LiveBehaviourMonitor(
        window_seconds=PERSONAL_WINDOW_SECONDS,
        warmup_seconds=PERSONAL_WARMUP_SECONDS,
        min_keys=LIVE_MIN_KEYS,
        min_clicks=LIVE_MIN_CLICKS,
        now_fn=now_fn,
    )


def sync_personal_monitor_lifetime(monitor, *, app_mode: str) -> None:
    """Stop personal listeners unless Personal Collector is the active mode."""
    if app_mode != "Personal Collector":
        monitor.stop()


def _notes_text(snapshot: dict) -> str:
    return " ".join(str(n) for n in (snapshot.get("notes") or []))


def sanitize_features(snapshot: dict) -> dict[str, Any]:
    """Map unmeasured live sentinels to None. Does not invent zeros."""
    raw = dict(snapshot.get("features") or {})
    notes = _notes_text(snapshot)
    keys = int(snapshot.get("key_count") or 0)
    out: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        value = raw.get(name)
        out[name] = value

    if snapshot.get("status") == "stopped":
        return {name: None for name in FEATURE_NAMES}

    if keys < 1:
        for name in (
            "typing_speed_kpm",
            "backspace_rate",
            "mean_key_interval_ms",
            "std_key_interval_ms",
        ):
            out[name] = None
    elif keys < 2:
        out["mean_key_interval_ms"] = None
        out["std_key_interval_ms"] = None
    elif keys < 3:
        out["std_key_interval_ms"] = None

    event_count = (
        keys
        + int(snapshot.get("mouse_move_count") or 0)
        + int(snapshot.get("click_count") or 0)
    )
    if event_count < 2:
        out["idle_time_sec"] = None

    if "mouse speed is unmeasured" in notes:
        out["mouse_speed_px_per_sec"] = None
        out["mouse_distance_px"] = None
    if "click rate is unmeasured" in notes:
        out["click_rate_per_min"] = None
    if "click latency is unmeasured" in notes or out.get("mean_click_latency_ms") is None:
        out["mean_click_latency_ms"] = None
    return out


def session_is_empty(snapshot: dict) -> bool:
    keys = int(snapshot.get("key_count") or 0)
    clicks = int(snapshot.get("click_count") or 0)
    moves = int(snapshot.get("mouse_move_count") or 0)
    return keys == 0 and clicks == 0 and moves == 0


def build_session_row(
    snapshot: dict,
    *,
    sleepiness_rating: Optional[int] = None,
    saved_at: Optional[str] = None,
    started_at: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    if session_is_empty(snapshot):
        raise ValueError("Session has no keyboard or clustered mouse activity; not saved.")
    stamp = saved_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rating = sleepiness_rating
    if rating is not None:
        rating = int(rating)
        if rating < SLEEPINESS_MIN or rating > SLEEPINESS_MAX:
            raise ValueError(
                f"sleepiness_rating must be {SLEEPINESS_MIN}-{SLEEPINESS_MAX} or omitted."
            )
    features = sanitize_features(snapshot)
    return {
        "session_id": session_id or f"real-{uuid.uuid4().hex[:12]}",
        "saved_at": stamp,
        "started_at": started_at or stamp,
        **{name: features[name] for name in FEATURE_NAMES},
        "quality_status": (
            snapshot.get("status")
            if snapshot.get("ready_for_prediction")
            else "insufficient"
        ),
        "quality_ready": bool(snapshot.get("ready_for_prediction")),
        "quality_notes": "; ".join(str(n) for n in (snapshot.get("notes") or [])),
        "key_count": int(snapshot.get("key_count") or 0),
        "backspace_count": int(snapshot.get("backspace_count") or 0),
        "mouse_move_count": int(snapshot.get("mouse_move_count") or 0),
        "click_count": int(snapshot.get("click_count") or 0),
        "sleepiness_rating": rating,
        "is_synthetic": False,
        "source_session_id": "",
    }


def append_session_row(row: dict[str, Any], path: Path = REAL_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REAL_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in REAL_COLUMNS})
    return path


def save_snapshot(
    snapshot: dict,
    *,
    sleepiness_rating: Optional[int] = None,
    started_at: Optional[str] = None,
    path: Path = REAL_CSV,
) -> dict[str, Any]:
    row = build_session_row(
        snapshot,
        sleepiness_rating=sleepiness_rating,
        started_at=started_at,
    )
    append_session_row(row, path=path)
    return row


def load_sessions(path: Path = REAL_CSV) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=REAL_COLUMNS)
    frame = pd.read_csv(path)
    if "is_synthetic" in frame.columns:
        frame["is_synthetic"] = frame["is_synthetic"].astype(str).str.lower().isin({"true", "1"})
    return frame


def ready_real_sessions(path: Path = REAL_CSV) -> pd.DataFrame:
    frame = load_sessions(path)
    if frame.empty:
        return frame
    real = frame[~frame["is_synthetic"].fillna(False)]
    if "quality_ready" in real.columns:
        ready_flag = real["quality_ready"].astype(str).str.lower().isin({"true", "1"})
        return real[ready_flag].reset_index(drop=True)
    return real.reset_index(drop=True)


def _complete_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    return numeric.dropna(axis=0, how="any")


def generate_synthetic_rows(
    real: pd.DataFrame,
    *,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Draw from the empirical feature distribution. Not fatigue ground truth."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    ready = real
    if "quality_ready" in real.columns:
        ready = real[real["quality_ready"].astype(str).str.lower().isin({"true", "1"})]
    complete = _complete_feature_matrix(ready)
    if len(ready) < MIN_REAL_FOR_SYNTHETIC:
        raise ValueError(
            f"Need at least {MIN_REAL_FOR_SYNTHETIC} ready real sessions; have {len(ready)}."
        )
    generator = rng or np.random.default_rng()
    ids = ready["session_id"].astype(str).tolist() if "session_id" in ready.columns else ["unknown"]

    if len(complete) < 3:
        raise ValueError(
            "Need at least 3 complete ready real sessions to preserve feature correlations."
        )
    values = complete.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    cov = np.cov(values, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    cov = np.atleast_2d(cov)
    ridge = 1e-6 * np.eye(cov.shape[0])
    drawn = generator.multivariate_normal(mean, cov + ridge, size=n_samples)
    lows = complete.min(axis=0).to_numpy(dtype=float)
    highs = complete.max(axis=0).to_numpy(dtype=float)
    drawn = np.clip(drawn, lows, highs)
    samples = pd.DataFrame(drawn, columns=list(FEATURE_NAMES))
    method = "correlated_multivariate_normal_clipped_to_observed_range"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for i in range(n_samples):
        source = str(generator.choice(ids))
        row = {name: samples.iloc[i][name] for name in FEATURE_NAMES}
        for name in FEATURE_NAMES:
            value = row[name]
            if value is not None and pd.notna(value):
                row[name] = float(value)
            else:
                row[name] = None
        rows.append(
            {
                "session_id": f"syn-{uuid.uuid4().hex[:12]}",
                "saved_at": now,
                "started_at": now,
                **row,
                "quality_status": "synthetic",
                "quality_ready": False,
                "quality_notes": (
                    f"{method}; not fatigue ground truth; derived from real session {source}"
                ),
                "key_count": "",
                "click_count": "",
                "sleepiness_rating": None,
                "is_synthetic": True,
                "source_session_id": source,
            }
        )
    return pd.DataFrame(rows, columns=REAL_COLUMNS)


def write_synthetic(frame: pd.DataFrame, path: Path = SYNTHETIC_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def export_payload(real_path: Path = REAL_CSV, synthetic_path: Path = SYNTHETIC_CSV) -> str:
    payload = {
        "disclaimer": (
            "Personal behaviour aggregates only. No key characters. "
            "Synthetic rows are not fatigue ground truth."
        ),
        "real": load_sessions(real_path).to_dict(orient="records"),
        "synthetic": load_sessions(synthetic_path).to_dict(orient="records")
        if synthetic_path.is_file()
        else [],
    }
    return json.dumps(payload, indent=2, default=str)
