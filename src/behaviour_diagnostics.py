"""Developer diagnostics for live vs training behaviour features.

Does not change model weights, thresholds, labels, or live feature formulas.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from predict import FEATURE_NAMES, BASE_DIR, label_map, model, predict_fatigue, scaler

DATA_PATH = os.path.join(BASE_DIR, "data", "behavior_final_dataset.csv")
REPORT_PATH = os.path.join(BASE_DIR, "data", "live_behaviour_diagnosis.json")
VECTOR_LOG_PATH = os.path.join(BASE_DIR, "data", "live_behaviour_vectors.jsonl")
DEBUG_ENV = "NEUROPULSE_BEHAVIOUR_DEBUG"

FEATURE_UNITS = {
    "typing_speed_kpm": "Monkeytype WPM stored under a kpm name (5 chars/word), not raw keys/min",
    "backspace_rate": "corrections / keys (training floor ~0.01)",
    "mean_key_interval_ms": "60000 / WPM (word-period ms), not mean inter-key ms",
    "std_key_interval_ms": "training: (100 - Monkeytype consistency) * noise; live: pstdev of burst inter-key ms",
    "mouse_distance_px": "training: synthetic speed * 60 * 0.6–1.6; live: clustered speed * 60",
    "mouse_speed_px_per_sec": "training: uniform draw by fatigue_label; live: clustered cursor px/s",
    "click_rate_per_min": "training: uniform draw by fatigue_label; live: clicks / clustered click span * 60",
    "mean_click_latency_ms": "training: synthetic reaction-time draw by label; live: mean move→click ≤420ms",
    "idle_time_sec": "training: uniform draw by label; live: max in-window inter-event gap, capped 30s",
    "session_minutes": "training: uniform draw by label; live: wall-clock minutes since Start",
}

DEMO_PRESETS = {
    "Focused Coder": {
        "typing_speed_kpm": 230,
        "backspace_rate": 0.02,
        "mean_key_interval_ms": 115,
        "std_key_interval_ms": 28,
        "mouse_distance_px": 9500,
        "mouse_speed_px_per_sec": 160,
        "click_rate_per_min": 16,
        "mean_click_latency_ms": 90,
        "idle_time_sec": 1.5,
        "session_minutes": 40,
    },
    "Mild Fatigue": {
        "typing_speed_kpm": 135,
        "backspace_rate": 0.10,
        "mean_key_interval_ms": 320,
        "std_key_interval_ms": 120,
        "mouse_distance_px": 4200,
        "mouse_speed_px_per_sec": 75,
        "click_rate_per_min": 9,
        "mean_click_latency_ms": 340,
        "idle_time_sec": 8,
        "session_minutes": 150,
    },
    "High Fatigue": {
        "typing_speed_kpm": 72,
        "backspace_rate": 0.17,
        "mean_key_interval_ms": 510,
        "std_key_interval_ms": 185,
        "mouse_distance_px": 1800,
        "mouse_speed_px_per_sec": 30,
        "click_rate_per_min": 4,
        "mean_click_latency_ms": 620,
        "idle_time_sec": 15,
        "session_minutes": 310,
    },
}

# Representative short live session (active typing/mouse, minutes not hours).
LIVE_LIKE_VECTORS = {
    "short_active": {
        "typing_speed_kpm": 80.0,
        "backspace_rate": 0.0,
        "mean_key_interval_ms": 60000.0 / 80.0,
        "std_key_interval_ms": 30.0,
        "mouse_distance_px": 133.0 * 60.0,
        "mouse_speed_px_per_sec": 133.0,
        "click_rate_per_min": 20.0,
        "mean_click_latency_ms": 80.0,
        "idle_time_sec": 0.2,
        "session_minutes": 0.6,
    },
    "active_5min": {
        "typing_speed_kpm": 70.0,
        "backspace_rate": 0.04,
        "mean_key_interval_ms": 60000.0 / 70.0,
        "std_key_interval_ms": 40.0,
        "mouse_distance_px": 120.0 * 60.0,
        "mouse_speed_px_per_sec": 120.0,
        "click_rate_per_min": 12.0,
        "mean_click_latency_ms": 140.0,
        "idle_time_sec": 1.5,
        "session_minutes": 5.0,
    },
}


def load_training_frame(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def class_feature_stats(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        by_class = {}
        for label in (0, 1, 2):
            series = df.loc[df["fatigue_label"] == label, name]
            by_class[str(label)] = {
                "n": int(len(series)),
                "min": float(series.min()),
                "p05": float(series.quantile(0.05)),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
                "mean": float(series.mean()),
            }
        overall = df[name]
        out[name] = {
            "unit": FEATURE_UNITS[name],
            "overall": {
                "min": float(overall.min()),
                "p50": float(overall.median()),
                "max": float(overall.max()),
                "mean": float(overall.mean()),
            },
            "by_class": by_class,
        }
    return out


def predict_vector(features: dict) -> dict[str, Any]:
    ordered = {name: float(features[name]) for name in FEATURE_NAMES}
    result = predict_fatigue(**ordered)
    return {
        "features": ordered,
        "class_id": result["class_id"],
        "label": result["label"],
        "probabilities": result["probabilities"],
    }


def compare_vector_to_training(features: dict, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for name in FEATURE_NAMES:
        value = float(features[name])
        col = df[name]
        percentile = float((col <= value).mean() * 100.0)
        rows.append(
            {
                "feature": name,
                "value": value,
                "unit": FEATURE_UNITS[name],
                "training_p50": float(col.median()),
                "training_min": float(col.min()),
                "training_max": float(col.max()),
                "overall_percentile": round(percentile, 2),
                "class0_p50": float(df.loc[df["fatigue_label"] == 0, name].median()),
                "class1_p50": float(df.loc[df["fatigue_label"] == 1, name].median()),
                "class2_p50": float(df.loc[df["fatigue_label"] == 2, name].median()),
            }
        )
    return rows


def live_like_training_overlap(df: pd.DataFrame) -> dict[str, Any]:
    mask = (
        (df["session_minutes"] < 10)
        & (df["idle_time_sec"] < 2)
        & (df["mean_click_latency_ms"] <= 240)
    )
    counts = df.loc[mask, "fatigue_label"].value_counts().to_dict()
    return {
        "definition": "session_minutes<10 and idle_time_sec<2 and mean_click_latency_ms<=240",
        "n": int(mask.sum()),
        "n_total": int(len(df)),
        "labels": {str(k): int(v) for k, v in counts.items()},
        "class1_session_lt_10_frac": float((df.loc[df["fatigue_label"] == 1, "session_minutes"] < 10).mean()),
        "class1_idle_lt_2_frac": float((df.loc[df["fatigue_label"] == 1, "idle_time_sec"] < 2).mean()),
        "class2_latency_gt_420_frac": float(
            (df.loc[df["fatigue_label"] == 2, "mean_click_latency_ms"] > 420).mean()
        ),
    }


def swap_one_feature_predictions(base: dict, donor: dict) -> dict[str, Any]:
    out = {}
    for name in FEATURE_NAMES:
        trial = dict(base)
        trial[name] = donor[name]
        predicted = predict_vector(trial)
        out[name] = {"label": predicted["label"], **predicted["probabilities"]}
    return out


def build_report(live_features: Optional[dict] = None) -> dict[str, Any]:
    df = load_training_frame()
    importances = {
        name: float(score)
        for name, score in sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda item: -item[1],
        )
    }
    demo_preds = {name: predict_vector(vec) for name, vec in DEMO_PRESETS.items()}
    live_preds = {name: predict_vector(vec) for name, vec in LIVE_LIKE_VECTORS.items()}
    short = LIVE_LIKE_VECTORS["short_active"]
    class_medians = {
        str(label): {name: float(df.loc[df["fatigue_label"] == label, name].median()) for name in FEATURE_NAMES}
        for label in (0, 1, 2)
    }
    report: dict[str, Any] = {
        "label_map": label_map,
        "model_classes": [int(c) for c in model.classes_],
        "scaler_feature_names": list(getattr(scaler, "feature_names_in_", FEATURE_NAMES)),
        "scaler_mean": {name: float(v) for name, v in zip(FEATURE_NAMES, scaler.mean_)},
        "scaler_scale": {name: float(v) for name, v in zip(FEATURE_NAMES, scaler.scale_)},
        "gain_importances": importances,
        "training_rows": int(len(df)),
        "training_class_counts": {str(k): int(v) for k, v in df["fatigue_label"].value_counts().sort_index().items()},
        "feature_units_and_ranges": class_feature_stats(df),
        "demo_predictions": demo_preds,
        "live_like_predictions": live_preds,
        "short_active_vs_training": compare_vector_to_training(short, df),
        "live_like_training_overlap": live_like_training_overlap(df),
        "short_active_swap_to_class1_median": swap_one_feature_predictions(short, class_medians["1"]),
        "notes": [
            "Labels 0/1/2 were assigned from Monkeytype risk rules, then mouse/click/idle/session were sampled from the label.",
            "XGBoost gain is dominated by those synthetic columns: idle_time_sec, session_minutes, mean_click_latency_ms.",
            "A short live session with clustered motion and gesture latency sits in a region that is almost only class 0 in training.",
            "Readiness withholds prediction until keys/mouse/clicks/latency exist; it does not remap classes.",
            "Demo sliders can set long session_minutes, high idle, and >420ms latency that live capture rarely produces.",
        ],
    }
    if live_features:
        report["supplied_live_vector"] = predict_vector(live_features)
        report["supplied_live_vs_training"] = compare_vector_to_training(live_features, df)
    return report


def write_report(path: str = REPORT_PATH, live_features: Optional[dict] = None) -> str:
    payload = build_report(live_features=live_features)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def debug_enabled() -> bool:
    return os.environ.get(DEBUG_ENV, "").strip() in {"1", "true", "True", "yes"}


def log_live_vector(
    features: dict,
    prediction: Optional[dict],
    *,
    status: str,
    path: str = VECTOR_LOG_PATH,
) -> None:
    if not debug_enabled():
        return
    row = {
        "status": status,
        "features": {name: features.get(name) for name in FEATURE_NAMES},
        "prediction": prediction,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def format_debug_caption(features: dict, prediction: dict) -> str:
    parts = []
    for name in FEATURE_NAMES:
        value = features.get(name)
        if isinstance(value, (int, float)):
            parts.append(f"{name}={float(value):.3f}")
        else:
            parts.append(f"{name}={value}")
    probs = prediction.get("probabilities") or {}
    return (
        "DEV live vector: "
        + " | ".join(parts)
        + f" → {prediction.get('label')} "
        + f"(p0={probs.get('Alert')} p1={probs.get('Mild Fatigue')} p2={probs.get('High Fatigue')})"
    )
