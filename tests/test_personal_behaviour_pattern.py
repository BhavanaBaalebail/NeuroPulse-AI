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

from personal_behaviour_pattern import PATTERN_FEATURES, build_pattern, prepare


class PersonalBehaviourPatternTests(unittest.TestCase):
    def test_current_data_uses_ratings_and_excludes_synthetic(self):
        model, report = build_pattern(
            ROOT / "data" / "personal_behaviour_sessions.csv",
            ROOT / "data" / "personal_behaviour_synthetic.csv",
        )
        self.assertEqual(report["personal_sessions_used"], 15)
        self.assertEqual(report["personal_sessions_path"], "data/personal_behaviour_sessions.csv")
        self.assertEqual(report["personal_csv_synthetic_rows_excluded"], 0)
        self.assertEqual(report["self_reported_rating_count"], 15)
        self.assertEqual(report["synthetic_sessions_available_but_excluded"], 90)
        self.assertEqual(report["synthetic_source_sessions_excluded"], 9)
        self.assertEqual(model["synthetic_rows_used"], 0)
        self.assertFalse(model["classifier_trained"])
        self.assertIsNone(report["classifier_metrics"])
        self.assertEqual(report["quality_ready_sessions"], 12)
        self.assertEqual(report["pattern_sessions_used"], 12)
        self.assertEqual(report["missing_feature_counts"]["click_rate_per_min"], 2)
        self.assertIn("mean_key_interval_ms", report["missing_feature_counts"])
        self.assertEqual(
            report["feature_rating_associations"]["click_rate_per_min"]["status"],
            "not_interpreted_active_click_span_normalization",
        )

    def test_synthetic_rows_never_contribute_even_if_rated(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions = Path(tmp) / "sessions.csv"
            synthetic = Path(tmp) / "synthetic.csv"
            real = pd.read_csv(ROOT / "data" / "personal_behaviour_sessions.csv").iloc[:3].copy()
            synthetic_rows = real.copy()
            synthetic_rows["session_id"] = [f"syn-{i}" for i in range(len(synthetic_rows))]
            synthetic_rows["is_synthetic"] = True
            synthetic_rows["sleepiness_rating"] = 9
            real.to_csv(sessions, index=False)
            synthetic_rows.to_csv(synthetic, index=False)
            model, report = build_pattern(sessions, synthetic)
        self.assertEqual(model["sessions"], 3)
        self.assertEqual(sum(model["rating_counts_all_rated_sessions"].values()), 3)
        self.assertEqual(report["synthetic_sessions_available_but_excluded"], 3)

    def test_prepare_writes_separate_descriptive_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "pattern.json"
            report_path = Path(tmp) / "report.json"
            markdown_path = Path(tmp) / "report.md"
            model, report = prepare(
                ROOT / "data" / "personal_behaviour_sessions.csv",
                ROOT / "data" / "personal_behaviour_synthetic.csv",
                model_path,
                report_path,
                markdown_path,
            )
            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertEqual(json.loads(model_path.read_text())["artifact_type"], "descriptive_personal_behavior_pattern")
            self.assertIsNone(json.loads(report_path.read_text())["classifier_metrics"])
            self.assertIn("not a trained fatigue classifier", markdown_path.read_text().lower())
            self.assertEqual(set(model["features"]), set(PATTERN_FEATURES))


if __name__ == "__main__":
    unittest.main()