"""Experimental classifier on heuristic EVTRACK labels (not fatigue ground truth).

Trains a separate prototype on Alert/Mild/High windows only. Does not write
models/behavior_xgb.pkl, models/xgb_scaler.pkl, or change dashboard/fusion.
Metrics compare predictions to heuristic_v1 labels only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from logger import FEATURE_NAMES  # noqa: E402

LABEL_VERSION = "heuristic_v1"
KEEP_LABELS = ("Alert", "Mild Fatigue", "High Fatigue")
LABEL_TO_ID = {"Alert": 0, "Mild Fatigue": 1, "High Fatigue": 2}
ID_TO_LABEL = {0: "Alert", 1: "Mild Fatigue", 2: "High Fatigue"}

PROTECTED_ARTEFACTS = (
    ROOT / "models" / "behavior_xgb.pkl",
    ROOT / "models" / "xgb_scaler.pkl",
    ROOT / "models" / "behavior_model.pkl",
)
PROTOTYPE_MODEL_PATH = ROOT / "models" / "behaviour_prototype_heuristic_v1.pkl"
PROTOTYPE_REPORT_JSON = ROOT / "data" / "evtrack_prototype_evaluation.json"
PROTOTYPE_REPORT_MD = ROOT / "data" / "evtrack_prototype_evaluation.md"

DISCLAIMER = (
    "These metrics compare the prototype to heuristic_v1 labels only. "
    "They do NOT measure real fatigue-detection accuracy, clinical validity, "
    "or improvement of the production NeuroPulse XGBoost model."
)

SPLIT_RANDOM_STATE = 42
SPLIT_TEST_SIZE = 0.25
MIN_TEST_HIGH = 8
MIN_TRAIN_HIGH = 20
MIN_TEST_MILD = 12
MIN_TRAIN_MILD = 40


def load_labelled_windows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [name for name in ("user_anon", "label", *FEATURE_NAMES) if name not in frame.columns]
    if missing:
        raise ValueError(f"labelled CSV missing columns {missing}")
    kept = frame[frame["label"].isin(KEEP_LABELS)].copy()
    kept = kept.reset_index(drop=True)
    if kept["label"].isin(["Uncertain"]).any():
        raise RuntimeError("Uncertain rows were not excluded")
    return kept


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    # Preserve NA; do not fill with zero here.
    return frame[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def label_vector(frame: pd.DataFrame) -> np.ndarray:
    return frame["label"].map(LABEL_TO_ID).to_numpy(dtype=int)


def grouped_holdout_indices(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int = SPLIT_RANDOM_STATE,
    min_test_high: int = MIN_TEST_HIGH,
    min_train_high: int = MIN_TRAIN_HIGH,
    min_test_mild: int = MIN_TEST_MILD,
    min_train_mild: int = MIN_TRAIN_MILD,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    splitter = GroupShuffleSplit(
        n_splits=80,
        test_size=SPLIT_TEST_SIZE,
        random_state=random_state,
    )
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        y_train, y_test = y[train_idx], y[test_idx]
        if set(y_train) != {0, 1, 2} or set(y_test) != {0, 1, 2}:
            continue
        train_high = int((y_train == 2).sum())
        test_high = int((y_test == 2).sum())
        train_mild = int((y_train == 1).sum())
        test_mild = int((y_test == 1).sum())
        if (
            test_high >= min_test_high
            and train_high >= min_train_high
            and test_mild >= min_test_mild
            and train_mild >= min_train_mild
        ):
            meta = {
                "found": True,
                "split_index": split_index,
                "random_state": random_state,
                "test_size": SPLIT_TEST_SIZE,
                "train_users": sorted(set(groups[train_idx].tolist())),
                "test_users": sorted(set(groups[test_idx].tolist())),
                "limitation": (
                    f"High Fatigue test n={test_high} is small; High precision/recall "
                    "are unstable. A 12-user grouped holdout cannot support a strong "
                    "3-class generalisation claim."
                ),
            }
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError("user leakage in grouped split")
            return train_idx, test_idx, meta
    return (
        np.array([], dtype=int),
        np.array([], dtype=int),
        {
            "found": False,
            "limitation": (
                "No user-grouped holdout placed all three heuristic classes in both "
                "train and test with the minimum High/Mild counts. Evaluation metrics "
                "are omitted rather than fabricated."
            ),
        },
    )


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _counts(y: np.ndarray) -> dict[str, int]:
    return {ID_TO_LABEL[i]: int((y == i).sum()) for i in range(3)}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    per_class = {}
    for i, name in ID_TO_LABEL.items():
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[ID_TO_LABEL[i] for i in range(3)],
        zero_division=0,
    )
    return {
        "per_class": per_class,
        "confusion_matrix": {
            "order": [ID_TO_LABEL[i] for i in range(3)],
            "rows_true_cols_pred": matrix,
        },
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": report_text,
    }


def assert_experimental_path(path: Path) -> None:
    resolved = path.resolve()
    protected = {item.resolve() for item in PROTECTED_ARTEFACTS}
    if resolved in protected:
        raise RuntimeError(f"refusing to overwrite production artefact {path}")
    name = path.name
    if name in {"behavior_xgb.pkl", "xgb_scaler.pkl", "behavior_model.pkl"}:
        raise RuntimeError(f"refusing to overwrite production artefact {path}")


def train_prototype(frame: pd.DataFrame, **split_kwargs: Any) -> tuple[Optional[Pipeline], dict[str, Any]]:
    X = feature_matrix(frame)
    y = label_vector(frame)
    groups = frame["user_anon"].astype(str).to_numpy()
    train_idx, test_idx, split_meta = grouped_holdout_indices(X, y, groups, **split_kwargs)

    report: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "label_version": LABEL_VERSION,
        "n_windows_labelled": int(len(frame)),
        "class_counts_all": _counts(y),
        "features": list(FEATURE_NAMES),
        "missing_counts": {
            name: int(frame[name].isna().sum()) for name in FEATURE_NAMES
        },
        "preprocessing": (
            "SimpleImputer(strategy=median) and StandardScaler fitted on training "
            "users only. Missing values are not filled with zero."
        ),
        "split": split_meta,
        "evaluation_target": "heuristic_v1 labels (not human fatigue ratings)",
    }

    if not split_meta.get("found"):
        report["metrics"] = None
        report["trained"] = False
        return None, report

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    pipe = make_pipeline()
    weights = compute_sample_weight("balanced", y_train)
    pipe.fit(X_train, y_train, clf__sample_weight=weights)
    y_pred = pipe.predict(X_test)

    imputer: SimpleImputer = pipe.named_steps["imputer"]
    report["class_counts_train"] = _counts(y_train)
    report["class_counts_test"] = _counts(y_test)
    report["imputer_strategy"] = imputer.strategy
    report["imputer_statistics_are_zero"] = [
        name
        for name, stat in zip(FEATURE_NAMES, imputer.statistics_)
        if np.isfinite(stat) and float(stat) == 0.0
    ]
    report["metrics"] = evaluate(y_test, y_pred)
    report["trained"] = True
    return pipe, report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Experimental behaviour prototype (heuristic labels)",
        "",
        DISCLAIMER,
        "",
        f"- Label version: `{report.get('label_version')}`",
        f"- Windows used (Uncertain excluded): {report.get('n_windows_labelled')}",
        f"- Class counts (all): {report.get('class_counts_all')}",
        f"- Preprocessing: {report.get('preprocessing')}",
        "",
        "## User-grouped split",
        "",
        f"- Found: {report['split'].get('found')}",
        f"- Limitation: {report['split'].get('limitation')}",
    ]
    if report.get("trained"):
        lines.extend(
            [
                f"- Train users: {len(report['split']['train_users'])} {report['split']['train_users']}",
                f"- Test users: {len(report['split']['test_users'])} {report['split']['test_users']}",
                f"- Train counts: {report.get('class_counts_train')}",
                f"- Test counts: {report.get('class_counts_test')}",
                "",
                "## Test metrics vs heuristic_v1 (not fatigue accuracy)",
                "",
                f"- Balanced accuracy: {report['metrics']['balanced_accuracy']:.4f}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in report["metrics"]["per_class"].items():
            lines.append(
                f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['support']} |"
            )
        order = report["metrics"]["confusion_matrix"]["order"]
        matrix = report["metrics"]["confusion_matrix"]["rows_true_cols_pred"]
        lines.extend(["", f"Confusion matrix (rows true, cols predicted; order {order}):", "", "```", str(matrix), "```", ""])
        lines.extend(["", "```", report["metrics"]["classification_report"], "```"])
    else:
        lines.extend(["", "No test metrics were written because a reliable grouped split was not found."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_artefacts(pipe: Optional[Pipeline], report: dict[str, Any], *, model_path: Path, json_path: Path, md_path: Path) -> None:
    for path in (model_path, json_path, md_path):
        assert_experimental_path(path)
    if pipe is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)
        report["model_path"] = str(model_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def prepare(
    windows_csv: Path,
    model_path: Path = PROTOTYPE_MODEL_PATH,
    json_path: Path = PROTOTYPE_REPORT_JSON,
    md_path: Path = PROTOTYPE_REPORT_MD,
) -> dict[str, Any]:
    frame = load_labelled_windows(windows_csv)
    pipe, report = train_prototype(frame)
    save_artefacts(pipe, report, model_path=model_path, json_path=json_path, md_path=md_path)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train experimental heuristic-label prototype.")
    parser.add_argument(
        "--windows",
        type=Path,
        default=ROOT / "data" / "evtrack_behaviour_labelled.csv",
    )
    parser.add_argument("--model", type=Path, default=PROTOTYPE_MODEL_PATH)
    parser.add_argument("--report-json", type=Path, default=PROTOTYPE_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=PROTOTYPE_REPORT_MD)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = prepare(args.windows, args.model, args.report_json, args.report_md)
    print(DISCLAIMER)
    print("Trained:", report.get("trained"))
    print("Split found:", report["split"].get("found"))
    if report.get("metrics"):
        print("Balanced accuracy vs heuristic_v1:", report["metrics"]["balanced_accuracy"])
        print(report["metrics"]["classification_report"])
    else:
        print(report["split"].get("limitation"))
    print("Wrote", args.report_json)
    print("Wrote", args.report_md)
    if report.get("model_path"):
        print("Wrote", report["model_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
