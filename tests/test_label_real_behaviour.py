import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from label_real_behaviour_data import (
    DISCLAIMER,
    LABEL_ALERT,
    LABEL_HIGH,
    LABEL_MILD,
    LABEL_UNCERTAIN,
    LABEL_VERSION,
    label_windows,
    prepare,
)


def _row(
    user="u1",
    session="s1",
    *,
    typing=70.0,
    std=160.0,
    mouse=500.0,
    click_rate=30.0,
    latency=260.0,
    backspace=0.0,
    keys=40,
    idle=5.0,
    session_minutes=10.0,
    **extra,
):
    data = {
        "user_anon": user,
        "session_anon": session,
        "episode_index": 0,
        "window_start": "2020-02-17 08:00:00.000",
        "window_end": "2020-02-17 08:01:00.000",
        "episode_start": "2020-02-17 08:00:00.000",
        "window_seconds": 60.0,
        "typing_speed_kpm": typing,
        "backspace_rate": backspace,
        "mean_key_interval_ms": 60000.0 / typing if typing else None,
        "std_key_interval_ms": std,
        "mouse_distance_px": mouse * 60.0 if mouse is not None else None,
        "mouse_speed_px_per_sec": mouse,
        "click_rate_per_min": click_rate,
        "mean_click_latency_ms": latency,
        "idle_time_sec": idle,
        "session_minutes": session_minutes,
        "n_keydown": keys,
        "n_backspace": 0,
        "n_mousemove": 20,
        "n_mousedown": 4,
        "n_events": 50,
        "n_click_latencies": 2,
    }
    data.update(extra)
    return data


def _baseline_user(user="u1", n=6, **kwargs):
    return [_row(user=user, session=f"s{i}", **kwargs) for i in range(n)]


class LabelRealBehaviourTests(unittest.TestCase):
    def test_idle_and_session_alone_are_uncertain(self):
        rows = []
        for i in range(6):
            rows.append(
                _row(
                    typing=None,
                    std=None,
                    mouse=None,
                    click_rate=None,
                    latency=None,
                    backspace=None,
                    keys=0,
                    idle=30.0,
                    session_minutes=180.0,
                )
            )
        labelled = label_windows(pd.DataFrame(rows))
        self.assertEqual(len(labelled), 6)
        self.assertTrue((labelled["label"] == LABEL_UNCERTAIN).all())
        self.assertTrue(
            labelled["label_reason"].str.contains("idle_and_session_minutes_excluded_from_vote").all()
        )

    def test_missing_features_not_replaced_with_zero(self):
        rows = _baseline_user()
        rows.append(
            _row(
                session="probe",
                typing=None,
                std=None,
                mouse=None,
                click_rate=None,
                latency=None,
                backspace=None,
                keys=0,
            )
        )
        frame = pd.DataFrame(rows)
        labelled = label_windows(frame)
        probe = labelled.iloc[-1]
        self.assertTrue(pd.isna(probe["typing_speed_kpm"]))
        self.assertTrue(pd.isna(probe["mouse_speed_px_per_sec"]))
        self.assertNotEqual(probe["typing_speed_kpm"], 0)
        self.assertEqual(probe["label"], LABEL_UNCERTAIN)
        self.assertIn("missing_typing_speed_kpm", probe["label_reason"])

    def test_two_mild_deviations_are_mild_fatigue(self):
        rows = _baseline_user(typing=80.0, std=150.0, mouse=500.0, latency=250.0)
        rows.append(_row(session="slow", typing=60.0, std=210.0, mouse=500.0, latency=250.0))
        labelled = label_windows(pd.DataFrame(rows))
        probe = labelled.iloc[-1]
        self.assertEqual(probe["label"], LABEL_MILD)
        self.assertEqual(probe["label_version"], LABEL_VERSION)

    def test_two_strong_deviations_are_high_fatigue(self):
        rows = _baseline_user(typing=80.0, std=150.0, mouse=500.0, latency=250.0)
        rows.append(_row(session="very_slow", typing=40.0, std=280.0, mouse=500.0, latency=250.0))
        labelled = label_windows(pd.DataFrame(rows))
        self.assertEqual(labelled.iloc[-1]["label"], LABEL_HIGH)

    def test_typical_windows_are_alert(self):
        labelled = label_windows(pd.DataFrame(_baseline_user()))
        self.assertTrue((labelled["label"] == LABEL_ALERT).all())

    def test_single_deviation_stays_uncertain(self):
        rows = _baseline_user(typing=80.0, std=150.0, mouse=500.0)
        rows.append(_row(session="one", typing=50.0, std=150.0, mouse=500.0))
        labelled = label_windows(pd.DataFrame(rows))
        self.assertEqual(labelled.iloc[-1]["label"], LABEL_UNCERTAIN)

    def test_user_relative_baseline(self):
        slow_user = _baseline_user(user="slow", typing=40.0, std=150.0, mouse=500.0)
        fast_user = _baseline_user(user="fast", typing=100.0, std=150.0, mouse=500.0)
        # 40 WPM is typical for slow_user but a strong drop for fast_user when paired
        # with elevated jitter.
        slow_user.append(_row(user="slow", session="probe", typing=40.0, std=150.0, mouse=500.0))
        fast_user.append(_row(user="fast", session="probe", typing=40.0, std=270.0, mouse=500.0))
        labelled = label_windows(pd.DataFrame(slow_user + fast_user))
        slow_probe = labelled[(labelled.user_anon == "slow") & (labelled.session_anon == "probe")].iloc[0]
        fast_probe = labelled[(labelled.user_anon == "fast") & (labelled.session_anon == "probe")].iloc[0]
        self.assertEqual(slow_probe["label"], LABEL_ALERT)
        self.assertEqual(fast_probe["label"], LABEL_HIGH)

    def test_implausible_wpm_is_not_used_as_evidence(self):
        rows = _baseline_user(typing=70.0, mouse=500.0, std=150.0)
        rows.append(
            _row(
                session="repeat",
                typing=900.0,
                mouse=None,
                std=None,
                click_rate=None,
                latency=None,
                backspace=None,
                keys=40,
            )
        )
        labelled = label_windows(pd.DataFrame(rows))
        probe = labelled.iloc[-1]
        self.assertEqual(probe["label"], LABEL_UNCERTAIN)
        self.assertTrue(pd.isna(probe["mouse_speed_px_per_sec"]) or probe["n_mousedown"] >= 0)
        self.assertIn("missing_mouse_speed_px_per_sec", probe["label_reason"])

    def test_prepare_keeps_all_windows_and_writes_disclaimer(self):
        rows = _baseline_user() + [
            _row(
                session="empty",
                typing=None,
                std=None,
                mouse=None,
                click_rate=None,
                latency=None,
                backspace=None,
                keys=0,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "windows.csv"
            out = Path(tmp) / "labelled.csv"
            report_path = Path(tmp) / "report.json"
            pd.DataFrame(rows).to_csv(src, index=False)
            report = prepare(src, out, report_path)
            labelled = pd.read_csv(out)
            self.assertEqual(len(labelled), len(rows))
            self.assertEqual(report["windows_discarded"], 0)
            self.assertIn("label", labelled.columns)
            self.assertIn("label_reason", labelled.columns)
            self.assertIn("label_version", labelled.columns)
            self.assertIn("heuristic", report["disclaimer"].lower())
            self.assertIn("not actual fatigue", DISCLAIMER.lower())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["n_windows"], len(rows))
            self.assertIn("Uncertain", payload["label_distribution"])


if __name__ == "__main__":
    unittest.main()
