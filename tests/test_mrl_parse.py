import unittest

from import_mrl_eyes import parse_eye_state


class MrlParseTests(unittest.TestCase):
    def test_closed_and_open(self):
        # subject_img_gender_glasses_state_reflections_lighting_sensor
        self.assertEqual(parse_eye_state("s0001_00001_0_0_0_0_0_01.png"), "closed")
        self.assertEqual(parse_eye_state("s0001_00002_0_0_1_0_1_01.png"), "open")

    def test_short_name_skipped(self):
        self.assertIsNone(parse_eye_state("img_12.jpg"))


if __name__ == "__main__":
    unittest.main()
