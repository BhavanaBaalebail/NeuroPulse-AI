import os
import joblib
import pandas as pd

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "behavior_xgb.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "xgb_scaler.pkl")

# ---------------------------------------------------
# LOAD ONCE
# ---------------------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------------------------------
# LABEL MAP
# ---------------------------------------------------

label_map = {
    0: "Alert",
    1: "Mild Fatigue",
    2: "High Fatigue"
}

FEATURE_NAMES = [
    "typing_speed_kpm",
    "backspace_rate",
    "mean_key_interval_ms",
    "std_key_interval_ms",
    "mouse_distance_px",
    "mouse_speed_px_per_sec",
    "click_rate_per_min",
    "mean_click_latency_ms",
    "idle_time_sec",
    "session_minutes",
]

# ---------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------

def predict_fatigue(
    typing_speed_kpm,
    backspace_rate,
    mean_key_interval_ms,
    std_key_interval_ms,
    mouse_distance_px,
    mouse_speed_px_per_sec,
    click_rate_per_min,
    mean_click_latency_ms,
    idle_time_sec,
    session_minutes
):

    values = {
        "typing_speed_kpm": typing_speed_kpm,
        "backspace_rate": backspace_rate,
        "mean_key_interval_ms": mean_key_interval_ms,
        "std_key_interval_ms": std_key_interval_ms,
        "mouse_distance_px": mouse_distance_px,
        "mouse_speed_px_per_sec": mouse_speed_px_per_sec,
        "click_rate_per_min": click_rate_per_min,
        "mean_click_latency_ms": mean_click_latency_ms,
        "idle_time_sec": idle_time_sec,
        "session_minutes": session_minutes,
    }
    row = pd.DataFrame([{name: values[name] for name in FEATURE_NAMES}])

    row_scaled = scaler.transform(row)

    pred = model.predict(row_scaled)[0]
    probs = model.predict_proba(row_scaled)[0]

    confidence = float(round(float(max(probs)) * 100, 2))

    return {
        "class_id": int(pred),
        "label": label_map[int(pred)],
        "confidence": confidence,
        "probabilities": {
            "Alert": float(round(float(probs[0]) * 100, 2)),
            "Mild Fatigue": float(round(float(probs[1]) * 100, 2)),
            "High Fatigue": float(round(float(probs[2]) * 100, 2))
        }
    }

# ---------------------------------------------------
# TEST RUN
# ---------------------------------------------------

if __name__ == "__main__":

    # result = predict_fatigue(
    #     typing_speed_kpm=78,
    #     backspace_rate=0.14,
    #     mean_key_interval_ms=420,
    #     std_key_interval_ms=145,
    #     mouse_distance_px=1800,
    #     mouse_speed_px_per_sec=34,
    #     click_rate_per_min=4,
    #     mean_click_latency_ms=510,
    #     idle_time_sec=14,
    #     session_minutes=210
    
    result = predict_fatigue(
    typing_speed_kpm=95,           # Fast, consistent typing
    backspace_rate=0.04,           # Very few errors
    mean_key_interval_ms=310,      # Quick transitions
    std_key_interval_ms=45,        # High rhythmic consistency (Low jitter)
    mouse_distance_px=2400,        # Active movement
    mouse_speed_px_per_sec=85,     # Sharp, fast cursor movement
    click_rate_per_min=12,         # Highly engaged
    mean_click_latency_ms=280,     # Fast reaction time
    idle_time_sec=5,               # Minimal distraction
    session_minutes=20             # Just started
    )

    print(result)