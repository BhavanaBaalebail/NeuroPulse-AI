import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from logger import FEATURE_NAMES, LiveBehaviourMonitor
from personal_collector import (
    REAL_CSV,
    REAL_COLUMNS,
    append_session_row,
    build_session_row,
    export_payload,
    generate_synthetic_rows,
    load_sessions,
    sanitize_features,
    save_snapshot,
    write_synthetic,
)


def sample_snapshot(**updates):
    features = {name: 1.0 for name in FEATURE_NAMES}
    features["mean_click_latency_ms"] = None
    snapshot = {
        "status": "insufficient",
        "ready_for_prediction": False,
        "notes": ["click latency is unmeasured"],
        "features": features,
        "key_count": 2,
        "backspace_count": 1,
        "mouse_move_count": 3,
        "click_count": 0,
        "session_seconds": 10,
    }
    snapshot.update(updates)
    return snapshot


class PersonalCollectorTests(unittest.TestCase):
    def test_empty_session_is_rejected_and_low_activity_is_flagged(self):
        with self.assertRaisesRegex(ValueError, "no keyboard or clustered mouse activity"):
            build_session_row(
                sample_snapshot(key_count=0, click_count=0, mouse_move_count=0)
            )

        row = build_session_row(sample_snapshot())
        self.assertEqual(row["quality_status"], "insufficient")
        self.assertFalse(row["quality_ready"])
        self.assertEqual(row["key_count"], 2)

    def test_unmeasured_features_remain_missing(self):
        snapshot = sample_snapshot()
        features = sanitize_features(snapshot)
        self.assertIsNone(features["mean_click_latency_ms"])

        snapshot["key_count"] = 0
        snapshot["mouse_move_count"] = 0
        snapshot["click_count"] = 0
        features = sanitize_features(snapshot)
        self.assertIsNone(features["typing_speed_kpm"])
        self.assertIsNone(features["backspace_rate"])
        self.assertIsNone(features["mean_key_interval_ms"])
        self.assertIsNone(features["idle_time_sec"])

    def test_saved_rows_contain_only_aggregates_and_export_separates_data(self):
        with tempfile.TemporaryDirectory() as directory:
            real_path = Path(directory) / "real.csv"
            synthetic_path = Path(directory) / "synthetic.csv"
            snapshot = sample_snapshot(raw_events=["typed-secret", "Key.backspace"])
            row = save_snapshot(snapshot, path=real_path)
            self.assertEqual(row["backspace_count"], 1)
            self.assertEqual(row["mouse_move_count"], 3)
            self.assertFalse(row["is_synthetic"])
            self.assertEqual(set(row), set(REAL_COLUMNS))

            frame = load_sessions(real_path)
            self.assertEqual(len(frame), 1)
            self.assertNotIn("typed-secret", frame.to_csv(index=False))
            self.assertNotIn("Key.backspace", frame.to_csv(index=False))
            payload = export_payload(real_path, synthetic_path)
            self.assertIn('"real"', payload)
            self.assertIn('"synthetic": []', payload)

    def test_completed_session_creates_csv_with_one_row(self):
        self.assertEqual(REAL_CSV, ROOT / "data" / "personal_behaviour_sessions.csv")
        with tempfile.TemporaryDirectory() as directory:
            real_path = Path(directory) / "personal_behaviour_sessions.csv"
            monitor = LiveBehaviourMonitor(
                window_seconds=60,
                warmup_seconds=0,
                now_fn=lambda: 100.0,
            )
            monitor.start(enable_listeners=False)
            monitor.record_key()
            final_snapshot = monitor.snapshot()

            row = save_snapshot(
                final_snapshot,
                started_at="2026-10-24 12:00:00",
                path=real_path,
            )
            monitor.stop()

            self.assertTrue(real_path.is_file())
            saved = load_sessions(real_path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["session_id"], row["session_id"])
            self.assertEqual(saved.iloc[0]["started_at"], "2026-10-24 12:00:00")

    def test_synthetic_rows_keep_feature_covariance_and_source_ids(self):
        rng = np.random.default_rng(44)
        size = 30
        base = np.linspace(1.0, 10.0, size)
        real = pd.DataFrame({
            name: base * (index + 1) for index, name in enumerate(FEATURE_NAMES)
        })
        real["session_id"] = [f"real-{index}" for index in range(size)]
        real["quality_ready"] = True

        generated = generate_synthetic_rows(real, n_samples=20, rng=rng)
        self.assertTrue(generated["is_synthetic"].all())
        self.assertTrue(generated["quality_status"].eq("synthetic").all())
        self.assertTrue(generated["sleepiness_rating"].isna().all())
        self.assertTrue(generated["source_session_id"].isin(real["session_id"]).all())
        self.assertGreater(generated[FEATURE_NAMES].corr().iloc[0, 1], 0.5)
        for name in FEATURE_NAMES:
            self.assertGreaterEqual(generated[name].min(), real[name].min())
            self.assertLessEqual(generated[name].max(), real[name].max())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            write_synthetic(generated, path)
            stored = load_sessions(path)
            self.assertEqual(len(stored), 20)
            self.assertTrue(stored["is_synthetic"].all())

    def test_synthetic_generation_rejects_incomplete_distribution(self):
        real = pd.DataFrame({name: [1.0] + [None] * 7 for name in FEATURE_NAMES})
        real["session_id"] = [f"real-{index}" for index in range(8)]
        real["quality_ready"] = True
        with self.assertRaisesRegex(ValueError, "3 complete ready real sessions"):
            generate_synthetic_rows(real, n_samples=2, rng=np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()