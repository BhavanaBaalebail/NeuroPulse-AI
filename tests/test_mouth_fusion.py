import unittest

from vision.config import VisionConfig
from vision.eye_state import EyeObservation, EyeStateTracker
from vision.fusion import fuse
from vision.mouth_state import MouthStateTracker


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
            _eye(looking_down=True, eye_state="CLOSED", closed_duration_s=0.5, perclos=0.2),
            mouth,
            model_score=0.6,
            model_ok=True,
        )
        self.assertNotEqual(result.state, "DROWSY")
        self.assertTrue(any("looking_down" in r for r in result.reasons))

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

    def test_missing_face_is_alert_with_separate_activity(self):
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

        # Vision's primary states remain ALERT/DROWSY; activity is supplementary.
        res = fuse(_eye(eye_state="CLOSED", closed_duration_s=0.9, perclos=0.1), mouth, None, config=cfg)
        self.assertEqual(res.state, "ALERT")

        # Sustained closure: 1.6s -> DROWSY
        res = fuse(_eye(eye_state="CLOSED", closed_duration_s=1.6, perclos=0.2), mouth, None, config=cfg)
        self.assertEqual(res.state, "DROWSY")

        # Recovery: eyes reopen -> ALERT
        res = fuse(_eye(eye_state="OPEN", closed_duration_s=0.0, perclos=0.1), mouth, None, config=cfg)
        self.assertEqual(res.state, "ALERT")

    def test_experimental_eye_assistance_keeps_blinks_temporal(self):
        cfg = VisionConfig(calibration_frames=3, eye_model_experimental_enabled=True)
        tracker = EyeStateTracker(cfg, calibration_path=None)
        for i in range(3):
            eye = tracker.update(0.34, 0.34, 0.34, True, timestamp=i * 0.05)
        for i in range(1, 13):
            eye = tracker.update(0.34, 0.34, 0.34, True, timestamp=0.15 + i * 0.05)
        mouth = MouthStateTracker(cfg).update(0.08, timestamp=0.8)
        mid_ear = (eye.closed_threshold + eye.open_threshold) / 2

        blink = tracker.update(mid_ear, mid_ear, mid_ear, True, timestamp=0.85, eye_model_closed_probability=0.99)
        self.assertTrue(blink.eye_model_assisted)
        self.assertEqual(fuse(blink, mouth, None, config=cfg).state, "ALERT")
        reopened = tracker.update(0.34, 0.34, 0.34, True, timestamp=1.05, eye_model_closed_probability=0.0)
        self.assertEqual(reopened.event, "blink")
        self.assertEqual(fuse(reopened, mouth, None, config=cfg).state, "ALERT")

        sustained = None
        for i in range(18):
            sustained = tracker.update(
                mid_ear,
                mid_ear,
                mid_ear,
                True,
                timestamp=1.15 + i * 0.1,
                eye_model_closed_probability=0.99,
            )
        self.assertEqual(fuse(sustained, mouth, None, config=cfg).state, "DROWSY")

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


if __name__ == "__main__":
    unittest.main()
