# logger.py
# -----------------------------------------------------------
# Final Corrected Behavioral Biometrics Logger
# Real keyboard + mouse telemetry
# Rolling 10-second windows
# Metrics scaled correctly for model inference
#
# Run:
# python logger.py
#
# Output:
# live_metrics.csv
# -----------------------------------------------------------

import atexit
import time
import math
import threading
import statistics
from datetime import datetime

import pandas as pd
from pynput import keyboard, mouse

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

WINDOW_SECONDS = 3600
CSV_FILE = "live_metrics.csv"

# -----------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------

lock = threading.Lock()

# keyboard
total_keys = 0
backspace_count = 0
key_timestamps = []

# mouse
mouse_distance = 0.0
click_count = 0
click_latencies = []

# previous mouse position
prev_mouse_x = None
prev_mouse_y = None

# activity tracking
last_activity_time = time.time()
last_mouse_move_time = None

# -----------------------------------------------------------
# KEYBOARD LISTENER
# -----------------------------------------------------------

def on_key_press(key):
    global total_keys, backspace_count
    global key_timestamps, last_activity_time

    now = time.time()

    with lock:
        total_keys += 1
        key_timestamps.append(now)
        last_activity_time = now

        if key == keyboard.Key.backspace:
            backspace_count += 1


# -----------------------------------------------------------
# MOUSE MOVE LISTENER
# -----------------------------------------------------------

def on_move(x, y):
    global prev_mouse_x, prev_mouse_y
    global mouse_distance
    global last_activity_time, last_mouse_move_time

    now = time.time()

    with lock:
        if prev_mouse_x is not None and prev_mouse_y is not None:
            dx = x - prev_mouse_x
            dy = y - prev_mouse_y
            dist = math.sqrt(dx * dx + dy * dy)
            mouse_distance += dist

        prev_mouse_x = x
        prev_mouse_y = y

        last_activity_time = now
        last_mouse_move_time = now


# -----------------------------------------------------------
# MOUSE CLICK LISTENER
# -----------------------------------------------------------

def on_click(x, y, button, pressed):
    global click_count, click_latencies
    global last_activity_time

    if not pressed:
        return

    now = time.time()

    with lock:
        click_count += 1
        last_activity_time = now

        if last_mouse_move_time is not None:
            latency_ms = (now - last_mouse_move_time) * 1000

            if latency_ms >= 0:
                click_latencies.append(latency_ms)


# -----------------------------------------------------------
# COMPUTE METRICS
# -----------------------------------------------------------

def compute_metrics():
    global total_keys, backspace_count
    global key_timestamps
    global mouse_distance, click_count, click_latencies
    global last_activity_time

    with lock:

        now = time.time()

        # ---------------------------------------------------
        # Keyboard timing intervals
        # ---------------------------------------------------

        key_intervals_ms = []

        if len(key_timestamps) >= 2:
            for i in range(1, len(key_timestamps)):
                diff = (key_timestamps[i] - key_timestamps[i - 1]) * 1000
                key_intervals_ms.append(diff)

        mean_key_interval = (
            statistics.mean(key_intervals_ms)
            if key_intervals_ms else 0.0
        )

        std_key_interval = (
            statistics.pstdev(key_intervals_ms)
            if len(key_intervals_ms) >= 2 else 0.0
        )

        # ---------------------------------------------------
        # Scale metrics to per-minute units
        # ---------------------------------------------------

        scale_factor = 60 / WINDOW_SECONDS

        typing_speed_kpm = total_keys * scale_factor

        click_rate = click_count * scale_factor

        backspace_rate = (
            backspace_count / total_keys
            if total_keys > 0 else 0.0
        )

        mouse_speed = mouse_distance / WINDOW_SECONDS

        mean_click_latency = (
            statistics.mean(click_latencies)
            if click_latencies else 0.0
        )

        idle_time = now - last_activity_time

        # ---------------------------------------------------
        # Output Row
        # ---------------------------------------------------

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "window_seconds": WINDOW_SECONDS,
            "total_keys": total_keys,
            "typing_speed_kpm": round(typing_speed_kpm, 2),
            "backspace_count": backspace_count,
            "backspace_rate": round(backspace_rate, 4),
            "mean_key_interval_ms": round(mean_key_interval, 2),
            "std_key_interval_ms": round(std_key_interval, 2),
            "mouse_distance_px": round(mouse_distance, 2),
            "mouse_speed_px_per_sec": round(mouse_speed, 2),
            "click_count": click_count,
            "click_rate_per_min": round(click_rate, 2),
            "mean_click_latency_ms": round(mean_click_latency, 2),
            "idle_time_sec": round(idle_time, 2),
        }

        reset_window_state()

        return row


# -----------------------------------------------------------
# RESET STATE
# -----------------------------------------------------------

def reset_window_state():
    global total_keys, backspace_count
    global key_timestamps
    global mouse_distance, click_count
    global click_latencies
    global prev_mouse_x, prev_mouse_y

    total_keys = 0
    backspace_count = 0
    key_timestamps = []

    mouse_distance = 0.0
    click_count = 0
    click_latencies = []

    prev_mouse_x = None
    prev_mouse_y = None


# -----------------------------------------------------------
# SAVE CSV
# -----------------------------------------------------------

def append_to_csv(row):
    try:
        df = pd.DataFrame([row])

        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            header_needed = f.tell() == 0
            df.to_csv(f, header=header_needed, index=False)

    except Exception as e:
        print("CSV write error:", e)


# -----------------------------------------------------------
# LOOP
# -----------------------------------------------------------

def logger_loop():
    print(f"Logging every {WINDOW_SECONDS} seconds...")
    print("Press CTRL + C to stop.\n")

    while True:
        time.sleep(WINDOW_SECONDS)

        row = compute_metrics()
        append_to_csv(row)

        print("Logged:", row)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():

    kb_listener = keyboard.Listener(
        on_press=on_key_press
    )

    ms_listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click
    )

    kb_listener.start()
    ms_listener.start()

    logger_loop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped logger.")


# -----------------------------------------------------------
# Live dashboard monitor (timings and counts only)
# -----------------------------------------------------------

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

LIVE_WINDOW_SECONDS = 60.0
LIVE_WARMUP_SECONDS = 20.0
LIVE_MIN_KEYS = 25
LIVE_MIN_CLICKS = 2
LIVE_MIN_MOUSE_EVENTS = 2
LIVE_IDLE_CAP_SEC = 30.0
# Gaps longer than this are pauses (idle), not typing rhythm. Training std_key_interval
# comes from Monkeytype consistency during active typing, not AFK.
TYPING_PAUSE_MS = 2000.0
# Treat longer gaps as idle, not slow pointing/clicking (training rates assume activity).
MOUSE_ACTIVE_S = 0.5
CLICK_ACTIVE_S = 10.0
MIN_MOVE_PX = 3.0
# Training mean_click_latency_ms is a synthetic reaction-time draw (alert 70–240ms,
# mild up to 420ms). Live time-since-last-move after typing is not that quantity.
CLICK_GESTURE_MS = 420.0


def _active_motion(samples: list[tuple[float, float]], max_gap_s: float) -> tuple[float, float]:
    """Distance and duration from consecutive samples that are not idle gaps."""
    distance = 0.0
    duration = 0.0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        if 0.0 < dt <= max_gap_s:
            duration += dt
            distance += samples[i][1]
    return distance, duration


def _active_duration(times: list[float], max_gap_s: float) -> float:
    total = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if 0.0 < dt <= max_gap_s:
            total += dt
    return total
# Monkeytype (training source) WPM uses 5 characters per word.
# build_hybrid_dataset.py stores that WPM in typing_speed_kpm and
# mean_key_interval_ms = 60000 / wpm.
CHARS_PER_WORD = 5.0


def _halt_listeners(*listeners) -> None:
    for listener in listeners:
        if listener is None:
            continue
        try:
            listener.stop()
        except Exception:
            pass


def sync_live_monitor_lifetime(monitor, *, app_mode: str, live_mode: bool) -> None:
    """Stop capture unless Live Behaviour or Multimodal Mode needs the listeners."""
    capture_active = (app_mode == "Neural Dashboard" and live_mode) or (
        app_mode == "Multimodal Mode"
    )
    if not capture_active:
        monitor.stop()


class LiveBehaviourMonitor:
    """Rolling-window keyboard/mouse timings for the existing 10-feature schema.

    Stores event timestamps and counts only — never key characters, clipboard,
    or raw event payloads.
    """

    def __init__(
        self,
        window_seconds: float = LIVE_WINDOW_SECONDS,
        warmup_seconds: float = LIVE_WARMUP_SECONDS,
        min_keys: int = LIVE_MIN_KEYS,
        min_clicks: int = LIVE_MIN_CLICKS,
        idle_cap_sec: float = LIVE_IDLE_CAP_SEC,
        now_fn=None,
    ):
        self.window_seconds = float(window_seconds)
        self.warmup_seconds = float(warmup_seconds)
        self.min_keys = int(min_keys)
        self.min_clicks = int(min_clicks)
        self.min_mouse_events = LIVE_MIN_MOUSE_EVENTS
        self.idle_cap_sec = float(idle_cap_sec)
        self._now = now_fn or time.time
        self._lock = threading.Lock()
        self._running = False
        self._kb_listener = None
        self._ms_listener = None
        self._last_error = None
        self._reset_session_unlocked()
        atexit.register(self.stop)

    def _reset_session_unlocked(self) -> None:
        now = self._now()
        self._session_started_at = now
        self._last_activity_at = now
        self._last_mouse_move_at = None
        self._prev_mouse = None
        self._keys = []
        self._move_deltas = []
        self._clicks = []

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def start(self, enable_listeners: bool = True) -> dict:
        with self._lock:
            if self._running:
                return {"ok": True, "already_running": True, "error": None}
            self._last_error = None
            self._reset_session_unlocked()
            self._running = True

        if not enable_listeners:
            return {"ok": True, "already_running": False, "error": None}

        kb_listener = None
        ms_listener = None
        try:
            from pynput import keyboard as kb
            from pynput import mouse as ms
        except ImportError:
            with self._lock:
                self._running = False
                self._last_error = (
                    "pynput is not installed. Install it with `pip install pynput`."
                )
                error = self._last_error
            return {"ok": False, "already_running": False, "error": error}

        try:
            kb_listener = kb.Listener(on_press=self._on_key_press)
            ms_listener = ms.Listener(on_move=self._on_move, on_click=self._on_click)
            kb_listener.daemon = True
            ms_listener.daemon = True
            kb_listener.start()
            ms_listener.start()
        except Exception as exc:
            _halt_listeners(kb_listener, ms_listener)
            with self._lock:
                self._running = False
                self._kb_listener = None
                self._ms_listener = None
                self._last_error = (
                    "Could not start keyboard/mouse listeners. On macOS, enable "
                    "Accessibility (and Input Monitoring if prompted) for the "
                    f"terminal or Streamlit app. ({exc})"
                )
                error = self._last_error
            return {"ok": False, "already_running": False, "error": error}

        with self._lock:
            if not self._running:
                kb_to_stop, ms_to_stop = kb_listener, ms_listener
            else:
                self._kb_listener = kb_listener
                self._ms_listener = ms_listener
                return {"ok": True, "already_running": False, "error": None}
        _halt_listeners(kb_to_stop, ms_to_stop)
        return {"ok": True, "already_running": False, "error": None}

    def stop(self) -> None:
        with self._lock:
            self._running = False
            kb_listener = self._kb_listener
            ms_listener = self._ms_listener
            self._kb_listener = None
            self._ms_listener = None
            self._reset_session_unlocked()
        _halt_listeners(kb_listener, ms_listener)

    def _on_key_press(self, key) -> None:
        is_backspace = False
        try:
            from pynput import keyboard as kb

            is_backspace = key == kb.Key.backspace
        except Exception:
            is_backspace = str(key) == "Key.backspace"
        self.record_key(is_backspace=is_backspace)

    def _on_move(self, x, y) -> None:
        self.record_move(float(x), float(y))

    def _on_click(self, x, y, button, pressed) -> None:
        if pressed:
            self.record_click()

    def record_key(self, is_backspace: bool = False, timestamp: float | None = None) -> None:
        with self._lock:
            if not self._running:
                return
            now = self._now() if timestamp is None else float(timestamp)
            self._keys.append((now, bool(is_backspace)))
            self._last_activity_at = now
            self._prune_unlocked(now)

    def record_move(self, x: float, y: float, timestamp: float | None = None) -> None:
        with self._lock:
            if not self._running:
                return
            now = self._now() if timestamp is None else float(timestamp)
            dist = 0.0
            if self._prev_mouse is not None:
                dx = x - self._prev_mouse[0]
                dy = y - self._prev_mouse[1]
                dist = math.sqrt(dx * dx + dy * dy)
            self._prev_mouse = (x, y)
            self._move_deltas.append((now, dist))
            self._last_mouse_move_at = now
            self._last_activity_at = now
            self._prune_unlocked(now)

    def record_click(self, timestamp: float | None = None) -> None:
        with self._lock:
            if not self._running:
                return
            now = self._now() if timestamp is None else float(timestamp)
            latency = None
            if self._last_mouse_move_at is not None:
                latency = max(0.0, (now - self._last_mouse_move_at) * 1000.0)
            self._clicks.append((now, latency))
            self._last_activity_at = now
            self._prune_unlocked(now)

    def _prune_unlocked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._keys = [row for row in self._keys if row[0] >= cutoff]
        self._move_deltas = [row for row in self._move_deltas if row[0] >= cutoff]
        self._clicks = [row for row in self._clicks if row[0] >= cutoff]

    def snapshot(self) -> dict:
        with self._lock:
            now = self._now()
            self._prune_unlocked(now)
            elapsed = max(0.0, now - self._session_started_at)
            window_elapsed = min(self.window_seconds, elapsed) if self._running else 0.0

            key_count = len(self._keys)
            backspaces = sum(1 for _, is_bs in self._keys if is_bs)
            intervals = [
                (self._keys[i][0] - self._keys[i - 1][0]) * 1000.0
                for i in range(1, len(self._keys))
            ]
            burst_intervals = [gap for gap in intervals if gap <= TYPING_PAUSE_MS]
            mouse_distance = sum(delta for _, delta in self._move_deltas)
            click_count = len(self._clicks)
            latencies = [
                lat
                for _, lat in self._clicks
                if lat is not None and lat <= CLICK_GESTURE_MS
            ]
            event_times = sorted(
                [t for t, _ in self._keys]
                + [t for t, _ in self._move_deltas]
                + [t for t, _ in self._clicks]
            )
            internal_gaps = [
                event_times[i] - event_times[i - 1]
                for i in range(1, len(event_times))
            ]
            # Trailing time after the last event is dashboard viewing / Streamlit
            # refresh, not Monkeytype afkDuration. Use the largest in-window pause.
            idle_raw = max(internal_gaps) if internal_gaps else 0.0

            notes = []
            # WPM uses burst typing time (Monkeytype test duration), not idle between bursts.
            typing_span = sum(burst_intervals) / 1000.0
            if typing_span < 1e-3:
                typing_span = max(window_elapsed, 1e-6)
            typing_speed = (key_count / CHARS_PER_WORD) * 60.0 / typing_span
            mean_interval = 60000.0 / max(typing_speed, 1.0)
            # Training std_key_interval_ms is a Monkeytype-consistency proxy (alert
            # median ~30ms), not 5× inter-key std. Do not multiply by CHARS_PER_WORD.
            std_interval = statistics.pstdev(burst_intervals) if len(burst_intervals) >= 2 else 0.0

            significant_moves = [(t, dist) for t, dist in self._move_deltas if dist >= MIN_MOVE_PX]
            significant_distance, mouse_active = _active_motion(significant_moves, MOUSE_ACTIVE_S)
            mouse_ok = (
                len(significant_moves) >= self.min_mouse_events
                and mouse_active > 1e-3
                and significant_distance > 0.0
            )
            if mouse_ok:
                mouse_speed = significant_distance / mouse_active
            else:
                mouse_speed = 0.0
                notes.append(
                    "mouse speed is unmeasured without clustered cursor motion; "
                    "idle-span trackpad jitter is not treated as slow movement."
                )
            mouse_distance_minute = mouse_speed * 60.0

            click_times = [t for t, _ in self._clicks]
            click_active = _active_duration(click_times, CLICK_ACTIVE_S)
            clicks_ok = click_count >= self.min_clicks and click_active > 1e-3
            if clicks_ok:
                click_rate = click_count * 60.0 / click_active
            else:
                click_rate = 0.0
                notes.append(
                    "click rate is unmeasured when clicks are too sparse to estimate a per-minute rate."
                )

            latency_unavailable = len(latencies) == 0
            click_latency = statistics.mean(latencies) if latencies else None

            if key_count < self.min_keys:
                notes.append(
                    f"need at least {self.min_keys} key timings so a single correction cannot dominate backspace_rate."
                )
            if latency_unavailable:
                notes.append(
                    "click latency is unmeasured until a click follows a cursor move "
                    f"within {int(CLICK_GESTURE_MS)}ms (training reaction-time range); "
                    "prediction is withheld."
                )
            notes.append(
                "typing_speed_kpm is WPM using 5 keys/word over burst typing time "
                "(Monkeytype convention used in training)."
            )
            notes.append(
                "mouse_distance_px is one-minute equivalent (speed × 60), matching training construction."
            )
            notes.append("session_minutes is wall-clock time since Start, not a rolling-window rate.")

            features = {
                "typing_speed_kpm": typing_speed,
                "backspace_rate": (backspaces / key_count) if key_count else 0.0,
                "mean_key_interval_ms": mean_interval,
                "std_key_interval_ms": std_interval,
                "mouse_distance_px": mouse_distance_minute,
                "mouse_speed_px_per_sec": mouse_speed,
                "click_rate_per_min": click_rate,
                "mean_click_latency_ms": click_latency,
                "idle_time_sec": min(idle_raw, self.idle_cap_sec),
                "session_minutes": elapsed / 60.0,
            }
            ordered = {name: features[name] for name in FEATURE_NAMES}

            if not self._running:
                status = "stopped"
                status_label = "Stopped"
                ordered = {name: 0.0 for name in FEATURE_NAMES}
            elif self._last_error:
                status = "error"
                status_label = "Listener error"
            elif elapsed < self.warmup_seconds:
                status = "collecting"
                status_label = "Collecting data"
            elif (
                key_count < self.min_keys
                or not clicks_ok
                or not mouse_ok
                or latency_unavailable
            ):
                status = "insufficient"
                status_label = "Insufficient data"
            else:
                status = "ready"
                status_label = "Live features ready"

            return {
                "status": status,
                "status_label": status_label,
                "error": self._last_error,
                "running": self._running,
                "features": ordered,
                "notes": notes,
                "session_seconds": elapsed,
                "window_elapsed_s": window_elapsed,
                "key_count": key_count,
                "backspace_count": backspaces,
                "mouse_move_count": len(self._move_deltas),
                "click_count": click_count,
                "ready_for_prediction": status == "ready",
            }