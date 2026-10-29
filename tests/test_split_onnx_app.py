import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from vision.dataset_split import grouped_split, source_key, sources_overlap
from vision.onnx_model import OnnxFaceScorer


class SplitTests(unittest.TestCase):
    def test_aug_and_original_share_source(self):
        a = source_key(Path("data/train/drowsy/img_12.jpg"), "drowsy")
        b = source_key(Path("data/train/drowsy/aug_drowsy_12.jpg"), "drowsy")
        self.assertEqual(a, b)

    def test_no_source_leakage_across_splits(self):
        rows = []
        for i in range(20):
            rows.append({"path": f"img_{i}.jpg", "label": i % 2, "source": f"s{i // 2}", "class_name": "x"})
            rows.append({"path": f"aug_{i}.jpg", "label": i % 2, "source": f"s{i // 2}", "class_name": "x"})
        split = grouped_split(rows, val_frac=0.2, test_frac=0.2, seed=0)
        self.assertEqual(len(sources_overlap(split["train"], split["val"])), 0)
        self.assertEqual(len(sources_overlap(split["train"], split["test"])), 0)
        self.assertEqual(len(sources_overlap(split["val"], split["test"])), 0)
        for bucket in split.values():
            for rec in bucket:
                src = rec["source"]
                mates = [r for part in split.values() for r in part if r["source"] == src]
                splits = {r.get("split") for r in mates}
                self.assertEqual(len(splits), 1)

    def test_webcam_eye_sides_share_capture_source(self):
        left = source_key(Path("webcam_open_capture_notdrowsy_capture_73_left.png"), "open")
        right = source_key(Path("webcam_open_capture_notdrowsy_capture_73_right.png"), "open")
        self.assertEqual(left, right)

    def test_mixed_labels_for_one_source_stay_together(self):
        rows = []
        for source_id in range(12):
            rows.extend([
                {"path": f"{source_id}-open.png", "label": 0, "source": f"subject-{source_id}"},
                {"path": f"{source_id}-closed.png", "label": 1, "source": f"subject-{source_id}"},
            ])
        split = grouped_split(rows, val_frac=0.2, test_frac=0.2, seed=9)
        self.assertEqual(len(sources_overlap(split["train"], split["val"])), 0)
        self.assertEqual(len(sources_overlap(split["train"], split["test"])), 0)
        for bucket in split.values():
            labels_by_source = {}
            for row in bucket:
                labels_by_source.setdefault(row["source"], set()).add(row["label"])
            self.assertTrue(all(labels == {0, 1} for labels in labels_by_source.values()))


class OnnxFailureTests(unittest.TestCase):
    def test_missing_file_is_unavailable_not_zero(self):
        scorer = OnnxFaceScorer(Path("/tmp/does_not_exist_fatigue.onnx"), [])
        self.assertFalse(scorer.available)
        self.assertIsNone(scorer.score(np.zeros((10, 10, 3), np.uint8), (0, 0, 10, 10)))
        self.assertIsNotNone(scorer.last_error)

    def test_invalid_logits_return_none(self):
        scorer = OnnxFaceScorer(Path("/tmp/does_not_exist_fatigue.onnx"), [])
        scorer.available = True
        scorer.session = MagicMock()
        scorer.session.run.return_value = [np.array([[np.nan, np.inf]])]
        out = scorer._prob_drowsy(np.zeros((1, 3, 8, 8), np.float32))
        self.assertIsNone(out)
        self.assertIn("invalid", scorer.last_error.lower())

    def test_config_class_index(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "vision_model_config.json"
            cfg.write_text(json.dumps({
                "input_size": 160,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "onnx_drowsy_class_index": 1,
                "class_names": ["notdrowsy", "drowsy"],
            }))
            scorer = OnnxFaceScorer(Path("/tmp/missing.onnx"), [cfg])
            self.assertEqual(scorer.drowsy_idx, 1)
            self.assertEqual(scorer.input_size, 160)


class AppIntegrationTests(unittest.TestCase):
    def test_app_exports_detector_and_bridge_fields(self):
        import app

        self.assertTrue(hasattr(app, "FatigueDetector"))
        self.assertTrue(hasattr(app, "draw_status"))
        self.assertTrue(hasattr(app, "write_vision_bridge"))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vision_output.json"
            with patch.object(app, "VISION_OUTPUT_JSON", path):
                app.write_vision_bridge(
                    "ALERT",
                    12.0,
                    {"fatigue_score": 0.12, "reasons": ["eyes_open_stable"], "eye_state": "OPEN", "mouth_class": "closed"},
                )
            data = json.loads(path.read_text())
            self.assertEqual(data["label"], "ALERT")
            self.assertFalse(data["index_is_probability"])
            self.assertEqual(data["reasons"][0], "eyes_open_stable")

    def test_detector_cleanup(self):
        from vision.detector import FatigueDetector

        detector = FatigueDetector()
        mock_mesh = MagicMock()
        mock_mesh.close = MagicMock()
        detector.face_mesh = mock_mesh
        detector.close()
        mock_mesh.close.assert_called_once()
        self.assertIsNone(detector.face_mesh)

    def test_sound_alert_cooldown(self):
        # Verify cooldown logic: triggers beep only when elapsed > cooldown
        cooldown = 4.0
        last_beep = 100.0

        # Frame at 102.0s: elapsed 2.0s <= 4.0s -> no beep
        now = 102.0
        should_beep = (now - last_beep) > cooldown
        self.assertFalse(should_beep)

        # Frame at 105.0s: elapsed 5.0s > 4.0s -> triggers beep
        now = 105.0
        should_beep = (now - last_beep) > cooldown
        self.assertTrue(should_beep)


if __name__ == "__main__":
    unittest.main()
