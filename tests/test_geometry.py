import math
import unittest

from vision.geometry import (
    classify_head_pose,
    combined_ear,
    eye_aspect_ratio,
    looking_down_from_signals,
    mouth_aspect_ratio,
    pitch_from_matrix,
)


class _LM:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class GeometryTests(unittest.TestCase):
    def test_open_eye_ear_typical_range(self):
        # Wide-ish eye: vertical 6+6, horizontal 40 → EAR 0.15? Let's construct ~0.35
        # (a+b)/(2c)=0.35 → a+b = 0.70 c. If c=20, a+b=14, a=b=7
        coords = [(0, 0), (6, -3.5), (14, -3.5), (20, 0), (14, 3.5), (6, 3.5)]
        ear = eye_aspect_ratio(coords)
        self.assertIsNotNone(ear)
        self.assertAlmostEqual(ear, 0.35, places=2)

    def test_invalid_landmarks_are_unknown(self):
        self.assertIsNone(eye_aspect_ratio(None))
        self.assertIsNone(eye_aspect_ratio([(0, 0)] * 5))
        collapsed = [(0, 0), (1, 1), (2, 2), (0, 0), (3, 3), (4, 4)]
        self.assertIsNone(eye_aspect_ratio(collapsed))
        self.assertIsNone(mouth_aspect_ratio(None))
        self.assertIsNone(combined_ear(None, None))

    def test_combined_ear_uses_available_side(self):
        self.assertEqual(combined_ear(None, 0.3), 0.3)
        self.assertEqual(combined_ear(0.4, None), 0.4)
        self.assertAlmostEqual(combined_ear(0.2, 0.4), 0.3)

    def test_mar_vertical_over_horizontal(self):
        mouth = [(0, 0), (0, 10), (0, 5), (20, 5)]
        self.assertAlmostEqual(mouth_aspect_ratio(mouth), 0.5)

    def test_pitch_from_identity_near_zero(self):
        ident = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        pitch = pitch_from_matrix(ident)
        self.assertIsNotNone(pitch)
        self.assertTrue(abs(pitch) < 1.0)

    def test_brief_glance_is_not_head_dropping(self):
        self.assertEqual(classify_head_pose(True, True, 0.3, 1.2), "glance_down")
        self.assertEqual(classify_head_pose(True, True, 1.2, 1.2), "head_dropping")
        self.assertEqual(classify_head_pose(True, False, 2.0, 1.2), "level")
        self.assertEqual(classify_head_pose(False, True, 2.0, 1.2), "unknown")

    def test_looking_down_uses_pitch_or_landmark_ratio(self):
        self.assertFalse(looking_down_from_signals(-4.0, 0.30, -12.0, 0.42))
        self.assertTrue(looking_down_from_signals(-15.0, 0.30, -12.0, 0.42))
        self.assertTrue(looking_down_from_signals(-4.0, 0.50, -12.0, 0.42))
        self.assertFalse(looking_down_from_signals(None, None, -12.0, 0.42))


if __name__ == "__main__":
    unittest.main()
