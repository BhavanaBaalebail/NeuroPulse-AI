import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from logger import FEATURE_NAMES
from prepare_real_behaviour_data import (
    BACKSPACE_KEYCODE,
    anonymize,
    build_windows,
    detect_separator,
    features_from_events,
    load_evtrack_info,
    load_evtrack_track,
    pair_keydown_keyup,
    prepare,
    split_episodes,
)


def _event(t, name, key=0, x=0.0, y=0.0):
    return {"t": float(t), "event": name, "key": key, "xpos": x, "ypos": y}


class PrepareRealBehaviourTests(unittest.TestCase):
    def test_detects_tab_separator_despite_csv_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            path.write_text("_id\tuser\tsession_id\n1\tU\t9\n", encoding="utf-8")
            self.assertEqual(detect_separator(path), "\t")

    def test_parses_tab_track_and_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = Path(tmp) / "EVTRACKINFO.csv"
            track = Path(tmp) / "EVTRACKTRACK.csv"
            info.write_text(
                "_id\tuser\tsession_id\tdate\tua\tscreen\tscreenh\twindoww\twindowh\tdocumentw\tdocumenth\n"
                "id1\tUser9\t42\t2020-02-17 09:35:49.089692\tMozilla/Test\t1920\t1080\t1920\t951\t1920\t951\n",
                encoding="utf-8",
            )
            track.write_text(
                "_id\tcursor\ttimestamp\txpos\typos\tkey\tevent\tuser\tsession_id\n"
                "a\t0\t2020-02-17 08:35:44.485\t10.0\t10.0\t0\tmousemove\tUser9\t42\n"
                "b\t0\t2020-02-17 08:35:44.585\t40.0\t10.0\t0\tmousemove\tUser9\t42\n"
                "c\t0\t2020-02-17 08:35:44.685\t0.0\t0.0\t87\tkeydown\tUser9\t42\n"
                "d\t0\t2020-02-17 08:35:44.785\t0.0\t0.0\t87\tkeyup\tUser9\t42\n",
                encoding="utf-8",
            )
            info_df = load_evtrack_info(info)
            track_df = load_evtrack_track(track)
            self.assertEqual(list(info_df.columns)[:11][:3], ["_id", "user", "session_id"])
            self.assertEqual(len(track_df), 4)
            self.assertEqual(track_df.iloc[2]["event"], "keydown")
            self.assertEqual(int(track_df.iloc[2]["key"]), 87)
            self.assertFalse(track_df["timestamp_parsed"].isna().any())
            windows = build_windows(track_df)
            self.assertTrue(
                str(windows.iloc[0]["window_start"]).startswith("2020-02-17 08:35:44")
            )

    def test_backspace_is_only_keycode_8(self):
        events = [
            _event(1.00, "keydown", key=87),
            _event(1.10, "keydown", key=8),
            _event(1.20, "keydown", key=67),
            _event(1.30, "keydown", key=14),
            _event(1.40, "keydown", key=0),
        ]
        feats = features_from_events(events, episode_start=1.0, window_end=2.0)
        self.assertEqual(feats["n_keydown"], 5)
        self.assertEqual(feats["n_backspace"], 1)
        self.assertEqual(BACKSPACE_KEYCODE, 8)
        self.assertAlmostEqual(feats["backspace_rate"], 0.2)

    def test_keyup_does_not_double_count_keys(self):
        events = [
            _event(1.00, "keydown", key=87),
            _event(1.08, "keyup", key=87),
            _event(1.20, "keydown", key=83),
            _event(1.28, "keyup", key=83),
        ]
        feats = features_from_events(events, episode_start=1.0, window_end=2.0)
        self.assertEqual(feats["n_keydown"], 2)
        pairing = pair_keydown_keyup(events)
        self.assertEqual(pairing["paired"], 2)
        self.assertEqual(pairing["unmatched_keydown"], 0)

    def test_mouse_distance_and_click_latency(self):
        events = [
            _event(1.00, "mousemove", x=0.0, y=0.0),
            _event(1.10, "mousemove", x=30.0, y=0.0),
            _event(1.12, "mousedown", x=30.0, y=0.0),
            _event(1.20, "mouseup", x=30.0, y=0.0),
            _event(1.30, "mousemove", x=60.0, y=0.0),
            _event(1.32, "mousedown", x=60.0, y=0.0),
        ]
        feats = features_from_events(events, episode_start=1.0, window_end=2.0)
        self.assertGreater(feats["mouse_speed_px_per_sec"], 0)
        self.assertAlmostEqual(feats["mouse_distance_px"], feats["mouse_speed_px_per_sec"] * 60.0)
        self.assertEqual(feats["n_mousedown"], 2)
        self.assertIsNotNone(feats["mean_click_latency_ms"])
        self.assertLessEqual(feats["mean_click_latency_ms"], 420)
        self.assertIsNotNone(feats["click_rate_per_min"])

    def test_stale_move_is_not_click_latency(self):
        events = [
            _event(1.00, "mousemove", x=0.0, y=0.0),
            _event(2.00, "mousedown", x=0.0, y=0.0),
        ]
        feats = features_from_events(events, episode_start=1.0, window_end=3.0)
        self.assertIsNone(feats["mean_click_latency_ms"])

    def test_idle_uses_max_internal_gap_not_trailing(self):
        events = [
            _event(1.0, "keydown", key=87),
            _event(1.1, "keydown", key=83),
            _event(4.1, "keydown", key=68),
        ]
        feats = features_from_events(events, episode_start=1.0, window_end=10.0)
        self.assertAlmostEqual(feats["idle_time_sec"], 3.0, places=5)

    def test_session_split_on_large_gap(self):
        events = [
            _event(1.0, "keydown", key=87),
            _event(2.0, "keydown", key=83),
            _event(400.0, "keydown", key=68),
        ]
        episodes = split_episodes(events, gap_s=300.0)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(len(episodes[0]), 2)
        self.assertEqual(len(episodes[1]), 1)

    def test_missing_features_are_none_not_fabricated(self):
        events = [_event(1.0, "mouseover", x=5.0, y=5.0)]
        feats = features_from_events(events, episode_start=1.0, window_end=2.0)
        self.assertIsNone(feats["typing_speed_kpm"])
        self.assertIsNone(feats["backspace_rate"])
        self.assertIsNone(feats["mouse_speed_px_per_sec"])
        self.assertIsNone(feats["mean_click_latency_ms"])
        self.assertIsNone(feats["click_rate_per_min"])

    def test_output_has_feature_order_no_text_no_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = Path(tmp) / "info.csv"
            track = Path(tmp) / "track.csv"
            out = Path(tmp) / "windows.csv"
            report_path = Path(tmp) / "quality.json"
            info.write_text(
                "_id\tuser\tsession_id\tdate\tua\tscreen\tscreenh\twindoww\twindowh\tdocumentw\tdocumenth\n"
                "id1\tUser1\t99\t2020-02-17 09:00:00.000000\tMozilla/Test\t800\t600\t800\t600\t800\t600\n",
                encoding="utf-8",
            )
            rows = [
                "_id\tcursor\ttimestamp\txpos\typos\tkey\tevent\tuser\tsession_id",
                "1\t0\t2020-02-17 08:00:00.000\t0.0\t0.0\t87\tkeydown\tUser1\t99",
                "2\t0\t2020-02-17 08:00:00.080\t0.0\t0.0\t87\tkeyup\tUser1\t99",
                "3\t0\t2020-02-17 08:00:00.160\t0.0\t0.0\t8\tkeydown\tUser1\t99",
                "4\t0\t2020-02-17 08:00:00.200\t10.0\t0.0\t0\tmousemove\tUser1\t99",
                "5\t0\t2020-02-17 08:00:00.280\t40.0\t0.0\t0\tmousemove\tUser1\t99",
                "6\t0\t2020-02-17 08:00:00.300\t40.0\t0.0\t0\tmousedown\tUser1\t99",
                "7\t0\t2020-02-17 08:00:00.400\t70.0\t0.0\t0\tmousemove\tUser1\t99",
                "8\t0\t2020-02-17 08:00:00.420\t70.0\t0.0\t0\tmousedown\tUser1\t99",
            ]
            # enough keydowns for a burst WPM
            extra = []
            for i in range(20):
                extra.append(
                    f"k{i}\t0\t2020-02-17 08:00:01.{i:03d}\t0.0\t0.0\t83\tkeydown\tUser1\t99"
                )
            track.write_text("\n".join(rows + extra) + "\n", encoding="utf-8")
            report = prepare(track, info, out, report_path)
            self.assertGreater(report["window_count"], 0)
            text = out.read_text(encoding="utf-8")
            self.assertNotIn("User1", text)
            self.assertNotIn("Mozilla", text)
            self.assertNotIn("fatigue_label", text)
            header = text.splitlines()[0].split(",")
            feat_start = header.index("typing_speed_kpm")
            self.assertEqual(header[feat_start : feat_start + 10], list(FEATURE_NAMES))
            self.assertIn("window_start", header)
            self.assertEqual(anonymize("User1"), anonymize("User1"))
            self.assertNotEqual(anonymize("User1"), anonymize("User2"))
            quality = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("feature_availability", quality)
            self.assertEqual(quality["inspection"]["event_types"]["keydown"], 22)

    def test_build_windows_does_not_keep_raw_user(self):
        import pandas as pd

        track = pd.DataFrame(
            [
                {
                    "_id": "a",
                    "cursor": 0,
                    "timestamp": "2020-02-17 08:00:00.000",
                    "xpos": 0.0,
                    "ypos": 0.0,
                    "key": 87,
                    "event": "keydown",
                    "user": "User3",
                    "session_id": 7,
                    "timestamp_parsed": pd.Timestamp("2020-02-17 08:00:00.000"),
                }
            ]
        )
        out = build_windows(track)
        self.assertNotIn("user", out.columns)
        self.assertNotIn("session_id", out.columns)
        self.assertEqual(out.iloc[0]["user_anon"], anonymize("User3"))


if __name__ == "__main__":
    unittest.main()
