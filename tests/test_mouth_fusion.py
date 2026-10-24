import unittest

from vision.config import VisionConfig
from vision.eye_state import EyeObservation
from vision.fusion import describe_activity, fuse
from vision.mouth_state import MouthStateTracker
from vision.overlay import format_status_headline


def _eye(**kwargs):
    baseline = kwargs.pop("open_ear_baseline", 0.34)
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
        open_ear_baseline=baseline,
        closed_threshold=0.21,
        open_threshold=0.27,
        reasons=[],
    )
    values.update(kwargs)
    return EyeObservation(**values)


class MouthFusionTests(unittest.TestCase):
    def test_short_mouth_opening_is_speech_not_yawn(self):
        tr = MouthStateTracker(VisionConfig())
        obs = tr.update(0.32, timestamp=0.0)
        obs = tr.update(0.32, timestamp=0.2)
        self.assertEqual(obs.mouth_class, "speaking")
        self.assertFalse(obs.candidate_yawn)

    def test_single_high_mar_frame_is_not_yawn(self):
        tr = MouthStateTracker(VisionConfig())
        obs = tr.update(0.55, timestamp=0.0)
        self.assertNotEqual(obs.mouth_class, "candidate_yawn")
        self.assertFalse(obs.candidate_yawn)

    def test_sustained_wide_open_is_candidate_yawn(self):
        tr = MouthStateTracker(VisionConfig(yawn_min_s=1.2, yawn_mar=0.48))
        obs = None
        for i in range(15):
            obs = tr.update(0.55, timestamp=i * 0.1)
        self.assertTrue(obs.candidate_yawn)
        self.assertEqual(obs.mouth_class, "candidate_yawn")

    def test_talking_does_not_make_alert_person_drowsy(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.30, timestamp=0.1)
        result = fuse(_eye(), mouth, model_score=0.4, model_ok=True)
        self.assertEqual(result.state, "ALERT")
        self.assertTrue(any("speech" in r for r in result.reasons))

    def test_looking_down_does_not_by_itself_trigger_drowsy(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=1.0)
        result = fuse(
            _eye(
                looking_down=True,
                eye_state="CLOSED",
                closed_duration_s=0.5,
                perclos=0.2,
                head_pose_status="glance_down",
                looking_down_duration_s=0.3,
            ),
            mouth,
            model_score=0.6,
            model_ok=True,
        )
        self.assertEqual(result.state, "ALERT")
        self.assertEqual(result.activity_label, "Glancing down")
        self.assertEqual(format_status_headline("ALERT", result.activity_label), "ALERT - Glancing down")

    def test_head_dropping_alone_is_not_drowsy(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=1.0)
        result = fuse(
            _eye(
                looking_down=True,
                eye_state="OPEN",
                head_pose_status="head_dropping",
                looking_down_duration_s=1.8,
            ),
            mouth,
            model_score=None,
            model_ok=False,
        )
        self.assertEqual(result.state, "ALERT")
        self.assertEqual(result.activity_label, "Head dropping")

    def test_head_dropping_with_prolonged_closure_is_combined(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=2.0)
        result = fuse(
            _eye(
                looking_down=True,
                eye_state="CLOSED",
                closed_duration_s=1.6,
                event="prolonged_closure",
                perclos=0.2,
                head_pose_status="head_dropping",
                looking_down_duration_s=1.8,
            ),
            mouth,
            model_score=None,
            model_ok=False,
        )
        self.assertEqual(result.state, "DROWSY")
        self.assertTrue(any("combined" in r or "sustained_eye_closure" in r for r in result.reasons))

    def test_high_perclos_does_not_stick_after_eyes_reopen(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=3.0)
        result = fuse(
            _eye(eye_state="OPEN", closed_duration_s=0.0, perclos=0.55),
            mouth,
            model_score=None,
            model_ok=False,
        )
        self.assertEqual(result.state, "ALERT")

    def test_normal_blink_stays_alert(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=1.0)
        result = fuse(_eye(event="blink", perclos=0.06), mouth, model_score=None, model_ok=False)
        self.assertEqual(result.state, "ALERT")

    def test_sustained_closure_is_drowsy(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=2.0)
        result = fuse(
            _eye(eye_state="CLOSED", closed_duration_s=1.8, perclos=0.3, event="prolonged_closure"),
            mouth,
            model_score=None,
            model_ok=False,
        )
        self.assertEqual(result.state, "DROWSY")
        self.assertTrue(any("sustained_eye_closure" in r for r in result.reasons))

    def test_missing_face_uncertain(self):
        mouth = MouthStateTracker(VisionConfig()).update(None, timestamp=0.0)
        result = fuse(_eye(face_detected=False, eye_state="UNKNOWN"), mouth, None, model_ok=False)
        self.assertEqual(result.state, "ALERT")
        self.assertEqual(result.activity_label, "No face")

    def test_model_cannot_create_drowsiness_alone(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=1.0)
        result = fuse(_eye(perclos=0.04), mouth, model_score=0.99, model_ok=True)
        self.assertEqual(result.state, "ALERT")

    def test_temporal_state_transitions(self):
        cfg = VisionConfig(prolonged_closure_s=0.8, drowsy_closure_s=1.5)
        mouth = MouthStateTracker(cfg).update(0.08, timestamp=0.0)

        # Baseline: open eyes -> ALERT
        res = fuse(_eye(eye_state="OPEN", closed_duration_s=0.0, perclos=0.0), mouth, None, config=cfg)
        self.assertEqual(res.state, "ALERT")

        # Brief blink: closed for 0.2s, event='blink' -> ALERT
        res = fuse(_eye(eye_state="CLOSED", closed_duration_s=0.2, event="blink", perclos=0.05), mouth, None, config=cfg)
        self.assertEqual(res.state, "ALERT")

        # Short/prolonged closure under the drowsy threshold stays ALERT
        res = fuse(_eye(eye_state="CLOSED", closed_duration_s=0.9, perclos=0.1), mouth, None, config=cfg)
        self.assertEqual(res.state, "ALERT")

        # Sustained closure: 1.6s -> DROWSY
        res = fuse(_eye(eye_state="CLOSED", closed_duration_s=1.6, perclos=0.2), mouth, None, config=cfg)
        self.assertEqual(res.state, "DROWSY")

        # Recovery: eyes reopen -> ALERT
        res = fuse(_eye(eye_state="OPEN", closed_duration_s=0.0, perclos=0.1), mouth, None, config=cfg)
        self.assertEqual(res.state, "ALERT")

    def test_candidate_yawn_alone_does_not_cause_drowsy(self):
        cfg = VisionConfig()
        mouth = MouthStateTracker(cfg)
        # Simulate sustained wide open mouth for 1.3s
        obs = None
        for i in range(14):
            obs = mouth.update(0.55, timestamp=i * 0.1)
        self.assertTrue(obs.candidate_yawn)
        res = fuse(_eye(eye_state="OPEN", closed_duration_s=0.0, perclos=0.0), obs, None, config=cfg)
        self.assertNotEqual(res.state, "DROWSY")
        self.assertTrue(any("candidate_yawn" in r for r in res.reasons))
        self.assertEqual(res.activity_label, "Drowsy-possible yawn")
        self.assertEqual(res.state, "ALERT")
        self.assertEqual(
            format_status_headline("ALERT", res.activity_label),
            "ALERT - Drowsy-possible yawn",
        )

    def test_activity_label_clears_when_head_is_level(self):
        mouth = MouthStateTracker(VisionConfig()).update(0.08, timestamp=1.0)
        down = fuse(
            _eye(looking_down=True, head_pose_status="glance_down", looking_down_duration_s=0.3),
            mouth,
            None,
            model_ok=False,
        )
        up = fuse(
            _eye(looking_down=False, head_pose_status="level", looking_down_duration_s=0.0),
            mouth,
            None,
            model_ok=False,
        )
        self.assertEqual(down.activity_label, "Glancing down")
        self.assertEqual(up.state, "ALERT")
        self.assertEqual(up.activity_label, "")
        self.assertEqual(format_status_headline("ALERT", up.activity_label), "ALERT")
        self.assertEqual(format_status_headline("ALERT?", "Talking?"), "ALERT - Talking")
        self.assertEqual(format_status_headline("DROWSY?", ""), "DROWSY")
        self.assertEqual(describe_activity(_eye(), mouth, "level"), "")


if __name__ == "__main__":
    unittest.main()
