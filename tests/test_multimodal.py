import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logger import LiveBehaviourMonitor, sync_live_monitor_lifetime
from multimodal import (
    FINAL_ALERT,
    FINAL_HIGH,
    FINAL_POSSIBLE,
    FINAL_UNKNOWN,
    VISION_STALE_S,
    append_state_log,
    behaviour_from_snapshot,
    combine,
    explain_fusion,
    parse_vision_payload,
    read_vision_file,
    start_multimodal_monitoring,
    stop_multimodal_monitoring,
    stop_vision_process,
    stream_display_label,
    sync_vision_process_lifetime,
    vision_process_wanted,
)
from vision.eye_state import EyeObservation
from vision.fusion import fuse
from vision.mouth_state import MouthStateTracker
from vision.config import VisionConfig


def _vision(
    label="ALERT",
    *,
    now=1000.0,
    updated_at=999.5,
    process_running=True,
    activity="",
    eye_state="OPEN",
    reasons=None,
    **extra,
):
    payload = {
        "label": label,
        "updated_at": updated_at,
        "activity_label": activity,
        "eye_state": eye_state,
        "reasons": reasons or [],
        "index_is_probability": False,
    }
    payload.update(extra)
    return parse_vision_payload(payload, now=now, process_running=process_running)


def _behaviour(label=None, status="ready", ready=True, notes=None, probs=None):
    snap = {
        "status": status,
        "ready_for_prediction": ready,
        "notes": notes or [],
    }
    pred = None
    if ready and label:
        pred = {"label": label, "probabilities": probs or {}}
    return behaviour_from_snapshot(snap, pred)


def _eye(**kwargs):
    values = dict(
        left_ear=0.34,
        right_ear=0.34,
        combined_ear=0.34,
        face_detected=True,
        looking_down=False,
        pitch_deg=0.0,
        eye_state="OPEN",
        event="none",
        closed_duration_s=0.0,
        perclos=0.05,
        calibrating=False,
        calibrated=True,
        open_ear_baseline=0.34,
        closed_threshold=0.21,
        open_threshold=0.27,
        reasons=[],
    )
    values.update(kwargs)
    return EyeObservation(**values)


class MultimodalFusionTests(unittest.TestCase):
    def test_sustained_eye_closure_is_high_concern_without_yawn_or_typing(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=1.0)
        vision_state = fuse(
            _eye(
                eye_state="CLOSED",
                closed_duration_s=1.6,
                perclos=0.4,
                event="prolonged_closure",
            ),
            mouth,
            model_score=None,
            model_ok=False,
        )
        self.assertEqual(vision_state.state, "DROWSY")
        fused = combine(
            _vision("DROWSY", reasons=vision_state.reasons, activity=""),
            _behaviour(status="insufficient", ready=False),
        )
        self.assertEqual(fused.final, FINAL_HIGH)
        self.assertTrue(any("vision_drowsy" in r for r in fused.reasons))
        self.assertIsNone(fused.confidence)

    def test_brief_blink_talking_looking_down_are_not_drowsy(self):
        blink = fuse(
            _eye(event="blink", eye_state="OPEN"),
            MouthStateTracker(VisionConfig()).update(0.08, timestamp=0.1),
            model_score=0.9,
            model_ok=True,
        )
        talk = fuse(
            _eye(),
            MouthStateTracker(VisionConfig()).update(0.30, timestamp=0.2),
            model_score=0.4,
            model_ok=True,
        )
        glance = fuse(
            _eye(
                looking_down=True,
                eye_state="CLOSED",
                closed_duration_s=0.5,
                perclos=0.2,
                head_pose_status="glance_down",
                looking_down_duration_s=0.3,
            ),
            MouthStateTracker(VisionConfig()).update(0.08, timestamp=1.0),
            model_score=0.6,
            model_ok=True,
        )
        for result, activity in (
            (blink, "Blinking"),
            (talk, "Talking"),
            (glance, "Glancing down"),
        ):
            self.assertEqual(result.state, "ALERT", activity)
            fused = combine(
                _vision("ALERT", activity=activity, reasons=result.reasons),
                _behaviour("Alert"),
            )
            self.assertEqual(fused.final, FINAL_ALERT)

    def test_vision_drowsy_not_cancelled_by_behaviour_alert(self):
        fused = combine(_vision("DROWSY", reasons=["sustained_eye_closure_1.50s"]), _behaviour("Alert"))
        self.assertEqual(fused.final, FINAL_HIGH)
        self.assertTrue(fused.conflict)
        self.assertIn("vision_drowsy_not_cancelled_by_behaviour_alert", fused.reasons)

    def test_behaviour_fatigue_with_vision_alert_is_conflict_not_alert(self):
        mild = combine(_vision("ALERT"), _behaviour("Mild Fatigue"))
        self.assertEqual(mild.final, FINAL_POSSIBLE)
        self.assertTrue(mild.conflict)
        self.assertIn("behaviour_mild_with_vision_alert_conflict", mild.reasons)
        high = combine(_vision("ALERT"), _behaviour("High Fatigue"))
        self.assertEqual(high.final, FINAL_HIGH)
        self.assertTrue(high.conflict)
        self.assertIn("behaviour_high_with_vision_alert_conflict", high.reasons)

    def test_both_streams_indicate_concern(self):
        fused = combine(
            _vision("DROWSY", reasons=["high_eye_closure_fraction_0.40_over_30s"]),
            _behaviour("High Fatigue"),
        )
        self.assertEqual(fused.final, FINAL_HIGH)
        self.assertFalse(fused.conflict)
        self.assertIn("both_streams_indicate_concern", fused.reasons)

    def test_recovery_after_eyes_reopen(self):
        drowsy = combine(_vision("DROWSY", reasons=["sustained_eye_closure_1.50s"]), _behaviour("Alert"))
        self.assertEqual(drowsy.final, FINAL_HIGH)
        recovered = combine(_vision("ALERT", eye_state="OPEN", reasons=["eyes_open_stable"]), _behaviour("Alert"))
        self.assertEqual(recovered.final, FINAL_ALERT)
        self.assertFalse(recovered.conflict)

    def test_no_face_is_unknown_not_alert_vote(self):
        vision = _vision(
            "ALERT",
            activity="No face",
            reasons=["no_face_or_lost_tracking"],
            eye_state="UNKNOWN",
        )
        self.assertFalse(vision.available)
        fused = combine(vision, _behaviour(status="insufficient", ready=False))
        self.assertEqual(fused.final, FINAL_UNKNOWN)

    def test_stale_and_stopped_camera(self):
        stale = _vision("DROWSY", updated_at=1000.0 - VISION_STALE_S - 0.5, now=1000.0)
        self.assertEqual(stale.status, "stale")
        self.assertFalse(stale.available)
        stopped = _vision("DROWSY", process_running=False)
        self.assertEqual(stopped.status, "stopped")
        missing_ts = parse_vision_payload(
            {"label": "DROWSY", "reasons": []}, now=1.0, process_running=True
        )
        self.assertFalse(missing_ts.available)
        self.assertIn("missing_timestamp", missing_ts.quality)

    def test_insufficient_behaviour_with_alert_vision_is_vision_only_alert(self):
        fused = combine(_vision("ALERT"), _behaviour(status="insufficient", ready=False))
        self.assertEqual(fused.final, FINAL_ALERT)
        self.assertIn("vision_alert_only", fused.reasons)

    def test_both_streams_unavailable_is_unknown(self):
        fused = combine(
            parse_vision_payload(None, now=1.0),
            _behaviour(status="stopped", ready=False),
        )
        self.assertEqual(fused.final, FINAL_UNKNOWN)
        self.assertIn("both_streams_unavailable", fused.reasons)

    def test_read_vision_file_and_bridge_timestamp(self):
        import app

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vision_output.json"
            with patch.object(app, "VISION_OUTPUT_JSON", path):
                app.write_vision_bridge(
                    "ALERT",
                    10.0,
                    {
                        "fatigue_score": 0.1,
                        "reasons": ["eyes_open_stable"],
                        "eye_state": "OPEN",
                        "ear": 0.33,
                        "mar": 0.2,
                        "perclos": 0.04,
                        "model_score": 0.1,
                    },
                )
            data = json.loads(path.read_text())
            self.assertIn("updated_at", data)
            reading = read_vision_file(path, now=data["updated_at"] + 0.2, process_running=True)
            self.assertTrue(reading.available)
            self.assertEqual(reading.vote, "alert")
            self.assertAlmostEqual(reading.ear, 0.33)

    def test_does_not_average_incompatible_scores(self):
        fused = combine(_vision("ALERT"), _behaviour("Mild Fatigue"))
        self.assertIsNone(fused.confidence)
        self.assertEqual(fused.final, FINAL_POSSIBLE)

    def test_freshness_threshold(self):
        just_ok = _vision("ALERT", updated_at=1000.0 - VISION_STALE_S, now=1000.0)
        self.assertTrue(just_ok.available)
        too_old = _vision("ALERT", updated_at=1000.0 - VISION_STALE_S - 0.01, now=1000.0)
        self.assertFalse(too_old.available)

    def test_mode_switch_listener_and_process_cleanup(self):
        mon = LiveBehaviourMonitor(now_fn=lambda: 1.0)
        mon.start(enable_listeners=False)
        sync_live_monitor_lifetime(mon, app_mode="Multimodal Mode", live_mode=False)
        self.assertTrue(mon.is_running())
        sync_live_monitor_lifetime(mon, app_mode="Neural Dashboard", live_mode=False)
        self.assertFalse(mon.is_running())

        proc = MagicMock()
        proc.poll.return_value = None
        self.assertTrue(vision_process_wanted("Multimodal Mode"))
        self.assertTrue(vision_process_wanted("Vision Mode"))
        self.assertFalse(vision_process_wanted("Neural Dashboard"))
        self.assertIs(sync_vision_process_lifetime(proc, wanted=True), proc)
        proc.terminate.assert_not_called()
        self.assertIsNone(sync_vision_process_lifetime(proc, wanted=False))
        proc.terminate.assert_called_once()
        stop_vision_process(None)

    def test_calibrating_and_unknown_eyes_are_not_alert_votes(self):
        cal = _vision("ALERT", activity="Calibrating", reasons=["calibrating_open_eye_baseline"])
        self.assertFalse(cal.available)
        unk = _vision("ALERT", eye_state="UNKNOWN", activity="Uncertain tracking")
        self.assertFalse(unk.available)
        fused = combine(cal, _behaviour(status="collecting", ready=False))
        self.assertEqual(fused.final, FINAL_UNKNOWN)

    def test_fusion_explanations_are_human_readable(self):
        both = combine(_vision("ALERT"), _behaviour("Alert"))
        self.assertEqual(both.explanation, "Both signals indicate normal alertness.")
        drowsy = combine(
            _vision("DROWSY", reasons=["sustained_eye_closure_1.50s"]),
            _behaviour("Alert"),
        )
        self.assertEqual(
            drowsy.explanation,
            "Sustained eye closure detected despite normal behavioural activity.",
        )
        conflict = combine(_vision("ALERT"), _behaviour("Mild Fatigue"))
        self.assertEqual(
            conflict.explanation,
            "Behaviour suggests fatigue despite an alert vision signal.",
        )
        missing = combine(
            parse_vision_payload(None, now=1.0),
            _behaviour(status="insufficient", ready=False),
        )
        self.assertEqual(missing.explanation, "Monitoring unavailable — check sensor status.")
        self.assertEqual(explain_fusion(missing), missing.explanation)

    def test_start_monitoring_does_not_duplicate_process(self):
        mon = LiveBehaviourMonitor(now_fn=lambda: 1.0)
        launched = []

        def fake_popen(*args, **kwargs):
            proc = MagicMock()
            proc.poll.return_value = None
            launched.append(args)
            return proc

        first = start_multimodal_monitoring(
            mon,
            None,
            executable="python3",
            app_path=ROOT / "app.py",
            cwd=ROOT,
            popen=fake_popen,
            enable_listeners=False,
        )
        self.assertTrue(first["ok"])
        self.assertFalse(first["vision"]["already_running"])
        self.assertEqual(len(launched), 1)
        second = start_multimodal_monitoring(
            mon,
            first["proc"],
            executable="python3",
            app_path=ROOT / "app.py",
            cwd=ROOT,
            popen=fake_popen,
            enable_listeners=False,
        )
        self.assertTrue(second["vision"]["already_running"])
        self.assertTrue(second["behaviour"]["already_running"])
        self.assertEqual(len(launched), 1)
        stop_multimodal_monitoring(mon, first["proc"])
        self.assertFalse(mon.is_running())
        first["proc"].terminate.assert_called()

    def test_partial_startup_failure_is_reported(self):
        mon = LiveBehaviourMonitor(now_fn=lambda: 1.0)
        result = start_multimodal_monitoring(
            mon,
            None,
            executable="python3",
            app_path=ROOT / "does_not_exist_app.py",
            cwd=ROOT,
            popen=lambda *a, **k: MagicMock(),
            enable_listeners=False,
        )
        self.assertTrue(result["partial"])
        self.assertFalse(result["ok"])
        self.assertTrue(result["behaviour"]["ok"])
        self.assertFalse(result["vision"]["ok"])
        stop_multimodal_monitoring(mon, result["proc"])

    def test_state_log_records_changes_only(self):
        a = combine(_vision("ALERT"), _behaviour("Alert"))
        b = combine(_vision("DROWSY", reasons=["sustained_eye_closure_1.50s"]), _behaviour("Alert"))
        log = append_state_log([], now=1.0, fused=a)
        log = append_state_log(log, now=2.0, fused=a)
        self.assertEqual(len(log), 1)
        log = append_state_log(log, now=3.0, fused=b)
        self.assertEqual(len(log), 2)
        self.assertEqual(log[-1]["final"], FINAL_HIGH)

    def test_stream_display_label_uses_predicted_class_not_live(self):
        self.assertEqual(
            stream_display_label(_behaviour("Mild Fatigue"), kind="behaviour"),
            "Mild Fatigue",
        )
        self.assertEqual(
            stream_display_label(_behaviour("High Fatigue"), kind="behaviour"),
            "High Fatigue",
        )
        self.assertEqual(
            stream_display_label(_behaviour("Alert"), kind="behaviour"),
            "Alert",
        )
        waiting = behaviour_from_snapshot(
            {"status": "insufficient", "ready_for_prediction": False, "notes": []},
            None,
        )
        self.assertEqual(stream_display_label(waiting, kind="behaviour"), "insufficient")
        self.assertEqual(stream_display_label(_vision("DROWSY"), kind="vision"), "DROWSY")
        self.assertEqual(stream_display_label(_vision("ALERT"), kind="vision"), "ALERT")


if __name__ == "__main__":
    unittest.main()
