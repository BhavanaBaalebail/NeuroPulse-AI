import unittest

from vision.config import VisionConfig
from vision.eye_state import EyeStateTracker


class EyeStateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = VisionConfig(calibration_frames=5, default_open_ear=0.34, closed_ratio=0.62, open_ratio=0.80)
        self.tr = EyeStateTracker(self.cfg, calibration_path=None)

    def _calibrate(self, ear=0.34):
        t = 0.0
        last = None
        for i in range(8):
            last = self.tr.update(ear, ear, ear, True, timestamp=t)
            t += 0.05
        self.assertTrue(last.calibrated)
        return t

    def test_calibration_uses_observed_open_ear(self):
        t = 0.0
        last = None
        for _ in range(8):
            last = self.tr.update(0.36, 0.36, 0.36, True, timestamp=t)
            t += 0.03
        self.assertAlmostEqual(last.open_ear_baseline, 0.36, places=2)
        self.assertLess(last.closed_threshold, 0.28)
        self.assertGreater(last.open_threshold, last.closed_threshold)

    def test_hysteresis_avoids_flicker(self):
        t = self._calibrate(0.34)
        closed_t = self.tr.open_ear_baseline * self.cfg.closed_ratio
        open_t = self.tr.open_ear_baseline * self.cfg.open_ratio
        mid = (closed_t + open_t) / 2
        obs = self.tr.update(mid, mid, mid, True, timestamp=t)
        self.assertEqual(obs.eye_state, "OPEN")
        low = closed_t - 0.02
        obs = self.tr.update(low, low, low, True, timestamp=t + 0.05)
        self.assertEqual(obs.eye_state, "CLOSED")
        obs = self.tr.update(mid, mid, mid, True, timestamp=t + 0.10)
        self.assertEqual(obs.eye_state, "CLOSED")
        high = open_t + 0.02
        obs = self.tr.update(high, high, high, True, timestamp=t + 0.15)
        self.assertEqual(obs.eye_state, "OPEN")

    def test_brief_blink_is_not_prolonged_closure(self):
        t = self._calibrate()
        closed = 0.12
        open_e = 0.34
        self.tr.update(closed, closed, closed, True, timestamp=t)
        mid = self.tr.update(closed, closed, closed, True, timestamp=t + 0.12)
        self.assertEqual(mid.eye_state, "CLOSED")
        done = self.tr.update(open_e, open_e, open_e, True, timestamp=t + 0.20)
        self.assertEqual(done.event, "blink")
        self.assertEqual(done.eye_state, "OPEN")

    def test_sustained_closure_event(self):
        t = self._calibrate()
        closed = 0.12
        obs = None
        for i in range(20):
            obs = self.tr.update(closed, closed, closed, True, timestamp=t + i * 0.1)
        self.assertEqual(obs.event, "prolonged_closure")
        self.assertGreaterEqual(obs.closed_duration_s, 0.8)

    def test_missing_face_is_unknown(self):
        t = self._calibrate()
        obs = self.tr.update(0.1, 0.1, 0.1, False, timestamp=t)
        self.assertEqual(obs.eye_state, "UNKNOWN")
        self.assertFalse(obs.face_detected)

    def test_none_ear_is_unknown_not_closed(self):
        t = self._calibrate()
        obs = self.tr.update(None, None, None, True, timestamp=t)
        self.assertEqual(obs.eye_state, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
