import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from logger import CHARS_PER_WORD
from logger import FEATURE_NAMES as LIVE_FEATURE_NAMES
from logger import LiveBehaviourMonitor, sync_live_monitor_lifetime
from predict import FEATURE_NAMES, predict_fatigue


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _pointer_sample(mon, clock):
    # Clustered pointing at alert-like speed (~133 px/s), with an idle gap between gestures.
    mon.record_move(0, 0)
    clock.advance(0.15)
    mon.record_move(20, 0)
    clock.advance(0.15)
    mon.record_move(40, 0)
    clock.advance(0.05)
    mon.record_click()
    clock.advance(2.0)
    mon.record_move(45, 5)
    clock.advance(0.15)
    mon.record_move(65, 5)
    clock.advance(0.15)
    mon.record_move(85, 5)
    clock.advance(0.05)
    mon.record_click()


def _type_burst(mon, clock, n=40, interval=0.1, backspace_at=None):
    for i in range(n):
        clock.advance(interval)
        mon.record_key(is_backspace=(backspace_at is not None and i == backspace_at))


class LiveBehaviourTests(unittest.TestCase):
    def test_feature_order_matches_predictor(self):
        self.assertEqual(list(LIVE_FEATURE_NAMES), list(FEATURE_NAMES))
        self.assertEqual(
            FEATURE_NAMES,
            [
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
            ],
        )

    def test_demo_predictor_still_accepts_slider_vector(self):
        result = predict_fatigue(
            230, 0.02, 115, 28, 9500, 160, 16, 90, 1.5, 40
        )
        self.assertIn(result["label"], {"Alert", "Mild Fatigue", "High Fatigue"})
        self.assertEqual(set(result["probabilities"]), {"Alert", "Mild Fatigue", "High Fatigue"})

    def test_warmup_is_collecting_even_with_keys(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=20, min_keys=5, now_fn=clock
        )
        mon.start(enable_listeners=False)
        for i in range(8):
            clock.advance(0.2)
            mon.record_key(is_backspace=False)
        snap = mon.snapshot()
        self.assertEqual(snap["status"], "collecting")
        self.assertEqual(snap["status_label"], "Collecting data")
        self.assertFalse(snap["ready_for_prediction"])
        mon.stop()

    def test_empty_window_after_warmup_is_insufficient_not_fatigue(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=20, min_keys=5, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(25)
        snap = mon.snapshot()
        self.assertEqual(snap["status"], "insufficient")
        self.assertEqual(snap["status_label"], "Insufficient data")
        self.assertFalse(snap["ready_for_prediction"])
        self.assertGreaterEqual(snap["features"]["session_minutes"], 25 / 60)
        mon.stop()

    def test_ready_snapshot_uses_schema_and_predicts(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=5, min_keys=5, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(5.1)
        for i in range(8):
            clock.advance(0.15)
            mon.record_key(is_backspace=(i == 3))
        _pointer_sample(mon, clock)
        snap = mon.snapshot()
        self.assertEqual(snap["status"], "ready")
        self.assertTrue(snap["ready_for_prediction"])
        self.assertEqual(list(snap["features"]), list(FEATURE_NAMES))
        result = predict_fatigue(**snap["features"])
        self.assertIn(result["label"], {"Alert", "Mild Fatigue", "High Fatigue"})
        self.assertGreater(snap["features"]["typing_speed_kpm"], 0)
        self.assertAlmostEqual(snap["features"]["backspace_rate"], 1 / 8)
        self.assertIsNotNone(snap["features"]["mean_click_latency_ms"])
        self.assertAlmostEqual(
            snap["features"]["mean_key_interval_ms"],
            60000.0 / max(snap["features"]["typing_speed_kpm"], 1.0),
            places=4,
        )
        self.assertAlmostEqual(
            snap["features"]["mouse_distance_px"],
            snap["features"]["mouse_speed_px_per_sec"] * 60.0,
            places=4,
        )
        mon.stop()

    def test_session_duration_is_not_the_rolling_window(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=10, warmup_seconds=2, min_keys=1, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(120)
        mon.record_key()
        snap = mon.snapshot()
        self.assertAlmostEqual(snap["features"]["session_minutes"], 2.0, places=5)
        self.assertLessEqual(snap["window_elapsed_s"], 10.0001)
        mon.stop()

    def test_start_stop_without_duplicate_listeners(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(now_fn=clock)
        first = mon.start(enable_listeners=False)
        second = mon.start(enable_listeners=False)
        self.assertTrue(first["ok"])
        self.assertFalse(first["already_running"])
        self.assertTrue(second["already_running"])
        self.assertTrue(mon.is_running())
        mon.record_key()
        mon.stop()
        self.assertFalse(mon.is_running())
        mon.record_key()
        snap = mon.snapshot()
        self.assertEqual(snap["status"], "stopped")
        self.assertEqual(snap["key_count"], 0)

    def test_listener_errors_are_reported(self):
        mon = LiveBehaviourMonitor(now_fn=Clock())
        try:
            import pynput.keyboard  # noqa: F401
        except ImportError:
            result = mon.start(enable_listeners=True)
            self.assertFalse(result["ok"])
            self.assertIn("pynput", result["error"].lower())
            self.assertFalse(mon.is_running())
            return
        with patch("pynput.keyboard.Listener", side_effect=OSError("accessibility denied")):
            result = mon.start(enable_listeners=True)
        self.assertFalse(result["ok"])
        self.assertIn("Accessibility", result["error"])
        self.assertFalse(mon.is_running())

    def test_features_match_training_formulas(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=1, min_keys=5, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(1.0)
        for _ in range(300):
            clock.advance(0.2)
            mon.record_key()
        # 1 + 300*0.2 = 61s elapsed; window keeps last 60s of keys.
        _pointer_sample(mon, clock)
        snap = mon.snapshot()
        keys = snap["key_count"]
        typing_span = max(0.2 * (keys - 1), 1e-3)
        expected_wpm = (keys / CHARS_PER_WORD) * 60.0 / typing_span
        self.assertAlmostEqual(snap["features"]["typing_speed_kpm"], expected_wpm, places=5)
        self.assertAlmostEqual(
            snap["features"]["mean_key_interval_ms"],
            60000.0 / max(expected_wpm, 1.0),
            places=5,
        )
        self.assertAlmostEqual(
            snap["features"]["mouse_distance_px"],
            snap["features"]["mouse_speed_px_per_sec"] * 60.0,
            places=5,
        )
        self.assertTrue(snap["ready_for_prediction"])
        mon.stop()

    def test_zero_click_window_is_not_ready(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=5, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        for _ in range(8):
            clock.advance(0.1)
            mon.record_key()
        snap = mon.snapshot()
        self.assertEqual(snap["status"], "insufficient")
        self.assertFalse(snap["ready_for_prediction"])
        self.assertIsNone(snap["features"]["mean_click_latency_ms"])
        mon.stop()

    def test_click_without_move_does_not_invent_latency(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=5, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        for _ in range(8):
            clock.advance(0.1)
            mon.record_key()
        mon.record_click()
        snap = mon.snapshot()
        self.assertFalse(snap["ready_for_prediction"])
        self.assertIsNone(snap["features"]["mean_click_latency_ms"])
        mon.stop()

    def test_mode_switch_helper_stops_vision_and_demo(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(now_fn=clock)
        mon.start(enable_listeners=False)
        sync_live_monitor_lifetime(mon, app_mode="Vision Mode", live_mode=True)
        self.assertFalse(mon.is_running())
        mon.start(enable_listeners=False)
        sync_live_monitor_lifetime(mon, app_mode="Neural Dashboard", live_mode=False)
        self.assertFalse(mon.is_running())
        mon.start(enable_listeners=False)
        sync_live_monitor_lifetime(mon, app_mode="Neural Dashboard", live_mode=True)
        self.assertTrue(mon.is_running())
        mon.start(enable_listeners=False)
        sync_live_monitor_lifetime(mon, app_mode="Multimodal Mode", live_mode=False)
        self.assertTrue(mon.is_running())
        sync_live_monitor_lifetime(mon, app_mode="Vision Mode", live_mode=True)
        self.assertFalse(mon.is_running())
        mon.stop()

    def test_stop_does_not_deadlock_when_listener_joins_callback(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(now_fn=clock)

        class JoiningListener:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                return None

            def stop(self):
                finished = threading.Event()

                def callback():
                    mon.record_key()
                    finished.set()

                worker = threading.Thread(target=callback)
                worker.start()
                worker.join(timeout=2.0)
                if worker.is_alive():
                    raise TimeoutError("deadlock: stop() held the callback lock")
                if not finished.wait(timeout=0.1):
                    raise TimeoutError("callback did not finish")

        with patch("pynput.keyboard.Listener", JoiningListener), patch(
            "pynput.mouse.Listener", JoiningListener
        ):
            self.assertTrue(mon.start(enable_listeners=True)["ok"])
            mon.stop()
        self.assertFalse(mon.is_running())

    def test_start_does_not_deadlock_if_listener_emits_during_start(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(now_fn=clock)

        class StartingListener:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                mon.record_key()

            def stop(self):
                return None

        done = threading.Event()
        error = []

        def run_start():
            try:
                with patch("pynput.keyboard.Listener", StartingListener), patch(
                    "pynput.mouse.Listener", StartingListener
                ):
                    mon.start(enable_listeners=True)
            except Exception as exc:
                error.append(exc)
            finally:
                done.set()

        worker = threading.Thread(target=run_start)
        worker.start()
        worker.join(timeout=2.0)
        self.assertTrue(done.is_set(), "deadlock: start() held the callback lock")
        self.assertEqual(error, [])
        mon.stop()

    def test_ordinary_typing_does_not_predict_fatigue(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=25, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        _type_burst(mon, clock, n=40, interval=0.1)
        _pointer_sample(mon, clock)
        snap = mon.snapshot()
        self.assertTrue(snap["ready_for_prediction"])
        self.assertGreater(snap["features"]["typing_speed_kpm"], 42)
        self.assertLess(snap["features"]["backspace_rate"], 0.05)
        result = predict_fatigue(**snap["features"])
        self.assertEqual(result["label"], "Alert")
        mon.stop()

    def test_isolated_backspace_does_not_score_fatigue(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=25, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        mon.record_key(is_backspace=True)
        _pointer_sample(mon, clock)
        snap = mon.snapshot()
        self.assertFalse(snap["ready_for_prediction"])
        self.assertEqual(snap["status"], "insufficient")
        mon.stop()

    def test_single_correction_in_ordinary_burst_stays_alert(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=25, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        _type_burst(mon, clock, n=40, interval=0.1, backspace_at=12)
        _pointer_sample(mon, clock)
        snap = mon.snapshot()
        self.assertTrue(snap["ready_for_prediction"])
        self.assertAlmostEqual(snap["features"]["backspace_rate"], 1 / 40)
        result = predict_fatigue(**snap["features"])
        self.assertEqual(result["label"], "Alert")
        mon.stop()

    def _ready_typing_session(self, n, interval, backspace_at=None, trailing_idle=0.0):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=25, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        _type_burst(mon, clock, n=n, interval=interval, backspace_at=backspace_at)
        _pointer_sample(mon, clock)
        clock.advance(trailing_idle)
        snap = mon.snapshot()
        self.assertEqual(snap["status"], "ready", snap)
        result = predict_fatigue(**snap["features"])
        return mon, snap, result

    def test_fast_normal_typing_zero_backspaces_is_alert(self):
        mon, snap, result = self._ready_typing_session(n=80, interval=0.12)
        feats = snap["features"]
        self.assertEqual(feats["backspace_rate"], 0.0)
        self.assertGreater(feats["typing_speed_kpm"], 70)
        self.assertLess(feats["std_key_interval_ms"], 70)
        self.assertEqual(result["label"], "Alert")
        mon.stop()

    def test_slow_normal_typing_zero_backspaces_is_alert(self):
        mon, snap, result = self._ready_typing_session(n=40, interval=0.28)
        feats = snap["features"]
        self.assertEqual(feats["backspace_rate"], 0.0)
        self.assertGreater(feats["typing_speed_kpm"], 38)
        self.assertLess(feats["typing_speed_kpm"], 55)
        self.assertEqual(result["label"], "Alert")
        mon.stop()

    def test_one_correction_in_fast_typing_is_alert(self):
        mon, snap, result = self._ready_typing_session(
            n=80, interval=0.12, backspace_at=20
        )
        self.assertAlmostEqual(snap["features"]["backspace_rate"], 1 / 80)
        self.assertEqual(result["label"], "Alert")
        mon.stop()

    def test_trailing_idle_after_typing_is_not_session_afk(self):
        mon, snap, result = self._ready_typing_session(
            n=80, interval=0.12, trailing_idle=12.0
        )
        self.assertLess(snap["features"]["idle_time_sec"], 6.0)
        self.assertEqual(result["label"], "Alert")
        mon.stop()

    def test_stale_click_latency_is_not_a_reaction_time(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=25, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        _type_burst(mon, clock, n=40, interval=0.12)
        mon.record_move(0, 0)
        clock.advance(0.15)
        mon.record_move(20, 0)
        clock.advance(1.0)
        mon.record_click()
        clock.advance(0.15)
        mon.record_move(40, 0)
        clock.advance(1.0)
        mon.record_click()
        snap = mon.snapshot()
        self.assertFalse(snap["ready_for_prediction"])
        self.assertEqual(snap["status"], "insufficient")
        self.assertIsNone(snap["features"]["mean_click_latency_ms"])
        mon.stop()

    def test_idle_mouse_span_during_typing_is_not_scored_as_fatigue(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=25, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        mon.record_move(0, 0)
        clock.advance(0.05)
        mon.record_click()
        _type_burst(mon, clock, n=300, interval=0.1)
        mon.record_move(1, 0)
        clock.advance(0.05)
        mon.record_click()
        snap = mon.snapshot()
        self.assertFalse(snap["ready_for_prediction"])
        self.assertEqual(snap["status"], "insufficient")
        mon.stop()

    def test_mouse_speed_uses_clustered_motion_not_idle_span(self):
        clock = Clock()
        mon = LiveBehaviourMonitor(
            window_seconds=60, warmup_seconds=2, min_keys=25, now_fn=clock
        )
        mon.start(enable_listeners=False)
        clock.advance(2.1)
        _type_burst(mon, clock, n=40, interval=0.1)
        _pointer_sample(mon, clock)
        snap = mon.snapshot()
        speed = snap["features"]["mouse_speed_px_per_sec"]
        self.assertGreater(speed, 65)
        self.assertLess(speed, 220)
        mon.stop()

    def test_behaviour_diagnostics_do_not_force_fatigue_on_short_live_like(self):
        from behaviour_diagnostics import (
            DEMO_PRESETS,
            LIVE_LIKE_VECTORS,
            build_report,
            predict_vector,
        )

        report = build_report()
        self.assertEqual(report["model_classes"], [0, 1, 2])
        self.assertEqual(report["scaler_feature_names"], list(FEATURE_NAMES))
        self.assertEqual(predict_vector(DEMO_PRESETS["Mild Fatigue"])["label"], "Mild Fatigue")
        self.assertEqual(predict_vector(DEMO_PRESETS["High Fatigue"])["label"], "High Fatigue")
        live = predict_vector(LIVE_LIKE_VECTORS["short_active"])
        self.assertEqual(live["label"], "Alert")
        self.assertGreater(live["probabilities"]["Alert"], 90)
        overlap = report["live_like_training_overlap"]
        self.assertGreaterEqual(overlap["labels"].get("0", 0), overlap["n"] - 2)
        self.assertLess(overlap["class1_session_lt_10_frac"], 0.02)
        top = list(report["gain_importances"])[:3]
        self.assertEqual(top, ["idle_time_sec", "session_minutes", "mean_click_latency_ms"])


if __name__ == "__main__":
    unittest.main()
