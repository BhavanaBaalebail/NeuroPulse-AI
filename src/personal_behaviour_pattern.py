"""Descriptive personal behavior patterns linked to self-reported ratings.

This module deliberately does not fit a fatigue classifier: the current source
contains too few sessions from one person for a defensible generalization test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SESSIONS = ROOT / "data" / "personal_behaviour_sessions.csv"
DEFAULT_SYNTHETIC = ROOT / "data" / "personal_behaviour_synthetic.csv"
DEFAULT_MODEL = ROOT / "models" / "behavior_pattern_prototype_personal_v1.json"
DEFAULT_REPORT = ROOT / "data" / "personal_behaviour_pattern_evaluation.json"
DEFAULT_MARKDOWN = ROOT / "data" / "personal_behaviour_pattern_evaluation.md"

# Exclude mathematically redundant features and fields that are not behavior
# signals. Missing click rates remain missing and are counted per feature.
PATTERN_FEATURES = (
    "typing_speed_kpm",
    "backspace_rate",
    "std_key_interval_ms",
    "mouse_speed_px_per_sec",
    "click_rate_per_min",
    "mean_click_latency_ms",
    "idle_time_sec",
    "session_minutes",
)
AUDIT_FEATURES = (
    "typing_speed_kpm",
    "backspace_rate",
    "mean_key_interval_ms",
    "std_key_interval_ms",
    "mouse_distance_px",
    "mouse_speed_px_per_sec",
    "click_rate_per_min",
    "mean_click_latency_ms",
    "idle_time_sec",
    "session_minutes",
)
RATING_BANDS = ((1, 3, "ratings_1_to_3"), (4, 6, "ratings_4_to_6"), (7, 9, "ratings_7_to_9"))
DISCLAIMER = (
    "Exploratory personal behavior-pattern summary linked to self-reported sleepiness. "
    "This is not a trained fatigue classifier, does not predict sleepiness, and is not "
    "evidence of population-level or clinical accuracy."
)


def _real_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"session_id", "sleepiness_rating", *PATTERN_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"personal sessions CSV missing columns: {missing}")
    if "is_synthetic" in frame.columns:
        synthetic = frame["is_synthetic"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes"}
        )
        frame = frame.loc[~synthetic].copy()
    frame["sleepiness_rating"] = pd.to_numeric(frame["sleepiness_rating"], errors="coerce")
    frame = frame.loc[frame["sleepiness_rating"].between(1, 9)].copy()
    for name in PATTERN_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.reset_index(drop=True)


def _report_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _group_summaries(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    ratings = frame["sleepiness_rating"]
    for lower, upper, name in RATING_BANDS:
        rows = frame.loc[ratings.between(lower, upper)]
        feature_stats = {}
        for feature in PATTERN_FEATURES:
            values = rows[feature].dropna()
            feature_stats[feature] = {
                "n": int(len(values)),
                "median": float(values.median()) if len(values) else None,
                "q1": float(values.quantile(0.25)) if len(values) else None,
                "q3": float(values.quantile(0.75)) if len(values) else None,
            }
        result[name] = {
            "rating_min": lower,
            "rating_max": upper,
            "sessions": int(len(rows)),
            "features": feature_stats,
        }
    return result


def build_pattern(sessions_path: Path, synthetic_path: Path | None = None) -> tuple[dict, dict]:
    source_rows = pd.read_csv(sessions_path)
    personal_synthetic_count = 0
    if "is_synthetic" in source_rows.columns:
        personal_flags = source_rows["is_synthetic"].astype(str).str.strip().str.lower()
        personal_synthetic_count = int(personal_flags.isin({"true", "1", "yes"}).sum())
    real = _real_rows(sessions_path)
    if "quality_ready" in real.columns:
        ready_mask = real["quality_ready"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes"}
        )
        pattern_rows = real.loc[ready_mask].copy()
    else:
        pattern_rows = real.copy()
    synthetic_count = 0
    synthetic_source_count = 0
    if synthetic_path is not None and synthetic_path.is_file():
        synthetic = pd.read_csv(synthetic_path)
        if "is_synthetic" in synthetic.columns:
            flags = synthetic["is_synthetic"].astype(str).str.strip().str.lower()
            synthetic_mask = flags.isin({"true", "1", "yes"})
            synthetic_count = int(synthetic_mask.sum())
            if "source_session_id" in synthetic.columns:
                synthetic_source_count = int(synthetic.loc[synthetic_mask, "source_session_id"].nunique())
        else:
            synthetic_count = int(len(synthetic))

    correlations = {}
    for feature in PATTERN_FEATURES:
        paired = pattern_rows[["sleepiness_rating", feature]].dropna()
        rho = None
        status = "descriptive_only"
        if feature == "click_rate_per_min":
            status = "not_interpreted_active_click_span_normalization"
        elif len(paired) >= 3 and paired[feature].nunique() > 1:
            rho = float(paired["sleepiness_rating"].corr(paired[feature], method="spearman"))
        correlations[feature] = {
            "n": int(len(paired)),
            "spearman_rho": rho,
            "status": status,
        }

    quality_counts = {}
    if "quality_status" in real.columns:
        quality_counts = {
            str(key): int(value)
            for key, value in real["quality_status"].value_counts(dropna=False).items()
        }
    quality_ready = None
    if "quality_ready" in real.columns:
        quality_ready = int(
            real["quality_ready"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).sum()
        )

    rating_counts = {
        str(int(rating)): int(count)
        for rating, count in real["sleepiness_rating"].value_counts().sort_index().items()
    }
    model = {
        "artifact_type": "descriptive_personal_behavior_pattern",
        "version": "personal_rating_pattern_v1",
        "predictive_model": False,
        "classifier_trained": False,
        "scope": "one person's real session summaries only",
        "rating_source": "user self-reported sleepiness_rating on a 1-9 scale",
        "features": list(PATTERN_FEATURES),
        "redundant_features_excluded": ["mean_key_interval_ms", "mouse_distance_px"],
        "synthetic_rows_used": 0,
        "sessions": int(real["session_id"].nunique()),
        "rating_counts_all_rated_sessions": rating_counts,
        "quality_ready_sessions": int(len(pattern_rows)),
        "rating_summary": {
            "median": float(real["sleepiness_rating"].median()) if len(real) else None,
            "min": float(real["sleepiness_rating"].min()) if len(real) else None,
            "max": float(real["sleepiness_rating"].max()) if len(real) else None,
        },
        "feature_rating_associations": correlations,
        "rating_band_summaries": _group_summaries(pattern_rows),
        "missing_feature_counts": {feature: int(real[feature].isna().sum()) for feature in AUDIT_FEATURES},
        "quality_ready_missing_feature_counts": {
            feature: int(pattern_rows[feature].isna().sum()) for feature in AUDIT_FEATURES
        },
        "quality_status_counts": quality_counts,
        "quality_ready_sessions": quality_ready,
        "personal_csv_synthetic_rows_excluded": personal_synthetic_count,
        "disclaimer": DISCLAIMER,
    }
    report = {
        "evaluation_type": "descriptive_only_no_classifier_holdout",
        "disclaimer": DISCLAIMER,
        "personal_sessions_path": _report_path(sessions_path),
        "synthetic_sessions_path": _report_path(synthetic_path),
        "personal_sessions_used": int(real["session_id"].nunique()),
        "personal_csv_synthetic_rows_excluded": personal_synthetic_count,
        "self_reported_rating_count": int(real["sleepiness_rating"].notna().sum()),
        "synthetic_sessions_available_but_excluded": synthetic_count,
        "synthetic_source_sessions_excluded": synthetic_source_count,
        "all_personal_rows_synthetic_flag_false": personal_synthetic_count == 0,
        "quality_status_counts": quality_counts,
        "quality_ready_sessions": quality_ready,
        "pattern_sessions_used": int(len(pattern_rows)),
        "rating_counts": rating_counts,
        "missing_feature_counts": model["missing_feature_counts"],
        "quality_ready_missing_feature_counts": model["quality_ready_missing_feature_counts"],
        "feature_rating_associations": correlations,
        "rating_band_summaries": model["rating_band_summaries"],
        "classifier_metrics": None,
        "data_quality_notes": [
            "All 15 self-ratings are reported, but feature associations and band medians use only collector quality-ready sessions.",
            "click_rate_per_min is clicks divided by clustered active-click duration, not wall-clock session time; extreme values are retained but its rating association is not interpreted.",
            "Only 15 sessions from one person are available; rating-band summaries are small descriptive groups, not validated states.",
            "mean_key_interval_ms duplicates typing_speed_kpm and mouse_distance_px duplicates mouse_speed_px_per_sec; aliases are omitted from pattern associations.",
        ],
        "reason_no_classifier": (
            "Only 15 self-rated sessions from one person are available. Session-level "
            "holdout would leave too little data to evaluate a fatigue classifier, and "
            "the 90 synthetic rows have no sleepiness ratings and are excluded."
        ),
    }
    return model, report


def write_markdown(model: dict, report: dict, path: Path) -> None:
    lines = [
        "# Personal behavior-pattern prototype (exploratory)",
        "",
        DISCLAIMER,
        "",
        f"- Real rated sessions used: {report['personal_sessions_used']}",
        f"- Quality-ready sessions used for feature patterns: {report['pattern_sessions_used']}",
        f"- Self-reported ratings: {report['self_reported_rating_count']} (scale 1–9)",
        f"- Synthetic sessions excluded: {report['synthetic_sessions_available_but_excluded']} across {report['synthetic_source_sessions_excluded']} source sessions",
        f"- Quality status: {report['quality_status_counts']}",
        f"- Missing features (all real rated sessions): {report['missing_feature_counts']}",
        f"- Missing features (quality-ready pattern sessions): {report['quality_ready_missing_feature_counts']}",
        "",
        "## Exploratory associations",
        "",
        "Spearman correlations are descriptive only; they are not significance or accuracy claims.",
        "",
        "| Feature | Paired sessions | Spearman rho with self-rating |",
        "|---|---:|---:|",
    ]
    for feature, stats in report["feature_rating_associations"].items():
        if stats["status"] == "not_interpreted_active_click_span_normalization":
            rho = "not interpreted (active-span rate)"
        else:
            rho = "not estimable" if stats["spearman_rho"] is None else f"{stats['spearman_rho']:.3f}"
        lines.append(f"| {feature} | {stats['n']} | {rho} |")
    lines.extend(["", "## Data quality notes", ""])
    lines.extend(f"- {note}" for note in report["data_quality_notes"])
    lines.extend(["", "## Why no classifier was trained", "", report["reason_no_classifier"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare(
    sessions_path: Path = DEFAULT_SESSIONS,
    synthetic_path: Path | None = DEFAULT_SYNTHETIC,
    model_path: Path = DEFAULT_MODEL,
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
) -> tuple[dict, dict]:
    model, report = build_pattern(sessions_path, synthetic_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(model, report, markdown_path)
    return model, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build descriptive personal behavior-pattern artifact.")
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    model, report = prepare(
        args.sessions,
        args.synthetic,
        args.model_out,
        args.report_out,
        args.markdown_out,
    )
    print(json.dumps({"artifact": str(args.model_out), "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
