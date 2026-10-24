import json
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

from logger import FEATURE_NAMES
from train_behaviour_prototype import (
    DISCLAIMER,
    PROTECTED_ARTEFACTS,
    assert_experimental_path,
    feature_matrix,
    grouped_holdout_indices,
    label_vector,
    load_labelled_windows,
    prepare,
    train_prototype,
)


def _synthetic_labelled(n_users: int = 12, per_class: int = 8) -> pd.DataFrame:
    rows = []
    templates = {
        "Alert": dict(typing_speed_kpm=80, std_key_interval_ms=120, mouse_speed_px_per_sec=600, click_rate_per_min=40, mean_click_latency_ms=240, backspace_rate=0.0),
        "Mild Fatigue": dict(typing_speed_kpm=50, std_key_interval_ms=220, mouse_speed_px_per_sec=350, click_rate_per_min=20, mean_click_latency_ms=310, backspace_rate=0.04),
        "High Fatigue": dict(typing_speed_kpm=35, std_key_interval_ms=300, mouse_speed_px_per_sec=180, click_rate_per_min=12, mean_click_latency_ms=360, backspace_rate=0.09),
    }
    rng = np.random.default_rng(0)
    for u in range(n_users):
        for label, base in templates.items():
            for i in range(per_class):
                row = {name: np.nan for name in FEATURE_NAMES}
                row.update(base)
                row["mean_key_interval_ms"] = 60000.0 / row["typing_speed_kpm"]
                row["mouse_distance_px"] = row["mouse_speed_px_per_sec"] * 60.0
                row["idle_time_sec"] = 4.0
                row["session_minutes"] = 5.0
                # One missing voting feature, never filled as 0 in the CSV.
                if i == 0:
                    row["click_rate_per_min"] = np.nan
                row["typing_speed_kpm"] += float(rng.normal(0, 2))
                row["user_anon"] = f"user{u}"
                row["session_anon"] = f"s{u}-{i}"
                row["label"] = label
                row["label_version"] = "heuristic_v1"
                rows.append(row)
    extra = {name: np.nan for name in FEATURE_NAMES}
    extra.update(
        {
            "user_anon": "user0",
            "session_anon": "unc",
            "label": "Uncertain",
            "label_version": "heuristic_v1",
            "idle_time_sec": 20.0,
            "session_minutes": 1.0,
        }
    )
    rows.append(extra)
    return pd.DataFrame(rows)


class TrainBehaviourPrototypeTests(unittest.TestCase):
    def test_load_excludes_uncertain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lab.csv"
            _synthetic_labelled().to_csv(path, index=False)
            frame = load_labelled_windows(path)
            self.assertNotIn("Uncertain", set(frame["label"]))
            self.assertEqual(len(frame), 12 * 3 * 8)

    def test_feature_matrix_keeps_nan_not_zero(self):
        frame = _synthetic_labelled(n_users=2, per_class=2)
        frame = frame[frame.label != "Uncertain"]
        matrix = feature_matrix(frame)
        self.assertTrue(np.isnan(matrix).any())
        click_col = list(FEATURE_NAMES).index("click_rate_per_min")
        nan_rows = np.isnan(matrix[:, click_col])
        self.assertTrue(nan_rows.any())
        self.assertFalse(np.any(matrix[nan_rows, click_col] == 0))

    def test_grouped_split_has_no_user_overlap(self):
        frame = _synthetic_labelled()
        frame = frame[frame.label != "Uncertain"].reset_index(drop=True)
        X = feature_matrix(frame)
        y = label_vector(frame)
        groups = frame["user_anon"].to_numpy()
        train_idx, test_idx, meta = grouped_holdout_indices(X, y, groups)
        self.assertTrue(meta["found"])
        self.assertEqual(set(groups[train_idx]) & set(groups[test_idx]), set())
        self.assertEqual(set(y[train_idx]), {0, 1, 2})
        self.assertEqual(set(y[test_idx]), {0, 1, 2})

    def test_refuses_production_model_paths(self):
        with self.assertRaises(RuntimeError):
            assert_experimental_path(PROTECTED_ARTEFACTS[0])
        with self.assertRaises(RuntimeError):
            assert_experimental_path(ROOT / "models" / "behavior_xgb.pkl")

    def test_train_reports_heuristic_not_fatigue_accuracy(self):
        frame = _synthetic_labelled()
        frame = frame[frame.label != "Uncertain"].reset_index(drop=True)
        pipe, report = train_prototype(frame)
        self.assertTrue(report["trained"])
        self.assertIsNotNone(pipe)
        self.assertIn("NOT measure real fatigue", report["disclaimer"])
        self.assertIn("NOT measure real fatigue", DISCLAIMER)
        metrics = report["metrics"]
        self.assertIn("balanced_accuracy", metrics)
        self.assertEqual(set(metrics["per_class"]), {"Alert", "Mild Fatigue", "High Fatigue"})
        for row in metrics["per_class"].values():
            self.assertIn("precision", row)
            self.assertIn("recall", row)
            self.assertIn("f1", row)
            self.assertIn("support", row)
        self.assertEqual(len(metrics["confusion_matrix"]["rows_true_cols_pred"]), 3)
        self.assertEqual(pipe.named_steps["imputer"].strategy, "median")

    def test_prepare_writes_experimental_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "lab.csv"
            model = Path(tmp) / "behaviour_prototype_heuristic_v1.pkl"
            js = Path(tmp) / "evtrack_prototype_evaluation.json"
            md = Path(tmp) / "evtrack_prototype_evaluation.md"
            _synthetic_labelled().to_csv(src, index=False)
            report = prepare(src, model, js, md)
            self.assertTrue(model.exists())
            self.assertTrue(js.exists())
            self.assertTrue(md.exists())
            payload = json.loads(js.read_text(encoding="utf-8"))
            self.assertTrue(payload["trained"])
            self.assertIn("heuristic_v1", md.read_text(encoding="utf-8"))
            self.assertNotEqual(model.resolve(), (ROOT / "models" / "behavior_xgb.pkl").resolve())
            self.assertIn("class_counts_test", report)


if __name__ == "__main__":
    unittest.main()
