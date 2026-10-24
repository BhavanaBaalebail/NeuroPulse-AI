"""Provisional heuristic labels for EVTRACK behaviour windows.

These labels are interpretable proxies from keyboard/mouse deviations versus
each user's own baseline. They are NOT measured fatigue, sleepiness, or
clinical ground truth, and they do not improve model accuracy.

Idle time and session duration never vote, including as a sole criterion.
Missing features stay missing; they are not filled with zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from logger import FEATURE_NAMES  # noqa: E402

LABEL_VERSION = "heuristic_v1"
LABEL_ALERT = "Alert"
LABEL_MILD = "Mild Fatigue"
LABEL_HIGH = "High Fatigue"
LABEL_UNCERTAIN = "Uncertain"

# Channels that can vote. mean_key_interval_ms duplicates typing_speed_kpm
# (60000/WPM). mouse_distance_px duplicates mouse_speed_px_per_sec (speed*60).
# idle_time_sec and session_minutes never vote.
EVIDENCE_CHANNELS = (
    "typing_speed_kpm",
    "std_key_interval_ms",
    "mouse_speed_px_per_sec",
    "click_rate_per_min",
    "mean_click_latency_ms",
    "backspace_rate",
)
LOWER_IS_FATIGUE = frozenset(
    {"typing_speed_kpm", "mouse_speed_px_per_sec", "click_rate_per_min"}
)
HIGHER_IS_FATIGUE = frozenset(
    {"std_key_interval_ms", "mean_click_latency_ms", "backspace_rate"}
)

MIN_BASELINE_N = 5
MIN_EVIDENCE_CHANNELS = 2
MIN_KEYS_FOR_TYPING = 10
MAX_PLAUSIBLE_WPM = 200.0

# Relative thresholds vs the user's leave-one-out median.
MILD_SLOWER = 0.80
HIGH_SLOWER = 0.60
MILD_FASTER_JITTER = 1.30
HIGH_FASTER_JITTER = 1.70
MILD_SLOWER_MOUSE = 0.75
HIGH_SLOWER_MOUSE = 0.55
MILD_SLOWER_CLICKS = 0.70
HIGH_SLOWER_CLICKS = 0.50
MILD_SLOWER_LATENCY = 1.15
HIGH_SLOWER_LATENCY = 1.35
# Backspace is almost always 0 in EVTRACK; use absolute lifts over self-median.
MILD_BACKSPACE_LIFT = 0.03
HIGH_BACKSPACE_LIFT = 0.08

DISCLAIMER = (
    "Labels are heuristic proxies from relative keyboard/mouse deviations, "
    "not actual fatigue ground truth. They must not be used as accuracy claims "
    "or as a substitute for labelled human state."
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def usable_typing_speed(row: pd.Series) -> Optional[float]:
    speed = row.get("typing_speed_kpm")
    keys = row.get("n_keydown")
    if _is_missing(speed) or _is_missing(keys):
        return None
    if int(keys) < MIN_KEYS_FOR_TYPING:
        return None
    if float(speed) > MAX_PLAUSIBLE_WPM:
        return None
    return float(speed)


def channel_value(row: pd.Series, channel: str) -> Optional[float]:
    if channel == "typing_speed_kpm":
        return usable_typing_speed(row)
    value = row.get(channel)
    if _is_missing(value):
        return None
    return float(value)


def leave_one_out_median(series: pd.Series, index, min_n: int = MIN_BASELINE_N) -> Optional[float]:
    others = series.drop(index)
    others = others[~others.map(_is_missing)]
    if len(others) >= min_n:
        return float(others.median())
    return None


def user_baselines(frame: pd.DataFrame) -> dict[Any, dict[str, pd.Series]]:
    tables: dict[Any, dict[str, pd.Series]] = {}
    for user, group in frame.groupby("user_anon", sort=False):
        tables[user] = {channel: group[channel] for channel in EVIDENCE_CHANNELS}
        tables[user]["typing_speed_kpm"] = group.apply(usable_typing_speed, axis=1)
    return tables


def vote_lower(value: float, baseline: float, mild_ratio: float, high_ratio: float) -> int:
    if baseline <= 0:
        return 0
    if value <= high_ratio * baseline:
        return 2
    if value <= mild_ratio * baseline:
        return 1
    return 0


def vote_higher(value: float, baseline: float, mild_ratio: float, high_ratio: float) -> int:
    if baseline < 0:
        return 0
    if value >= high_ratio * baseline:
        return 2
    if value >= mild_ratio * baseline:
        return 1
    return 0


def vote_channel(channel: str, value: float, baseline: float) -> int:
    if channel == "typing_speed_kpm":
        return vote_lower(value, baseline, MILD_SLOWER, HIGH_SLOWER)
    if channel == "mouse_speed_px_per_sec":
        return vote_lower(value, baseline, MILD_SLOWER_MOUSE, HIGH_SLOWER_MOUSE)
    if channel == "click_rate_per_min":
        return vote_lower(value, baseline, MILD_SLOWER_CLICKS, HIGH_SLOWER_CLICKS)
    if channel == "std_key_interval_ms":
        return vote_higher(value, baseline, MILD_FASTER_JITTER, HIGH_FASTER_JITTER)
    if channel == "mean_click_latency_ms":
        return vote_higher(value, baseline, MILD_SLOWER_LATENCY, HIGH_SLOWER_LATENCY)
    if channel == "backspace_rate":
        mild = baseline + MILD_BACKSPACE_LIFT
        high = baseline + HIGH_BACKSPACE_LIFT
        if value >= high:
            return 2
        if value >= mild:
            return 1
        return 0
    return 0


def decide_label(votes: dict[str, int]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if len(votes) < MIN_EVIDENCE_CHANNELS:
        reasons.append(
            f"uncertain_fewer_than_{MIN_EVIDENCE_CHANNELS}_measured_channels"
        )
        return LABEL_UNCERTAIN, reasons

    high = sum(1 for vote in votes.values() if vote == 2)
    mild = sum(1 for vote in votes.values() if vote == 1)
    typical = sum(1 for vote in votes.values() if vote == 0)
    fatigue = high + mild

    for channel, vote in votes.items():
        if vote == 2:
            reasons.append(f"strong_deviation_{channel}")
        elif vote == 1:
            reasons.append(f"mild_deviation_{channel}")
        else:
            reasons.append(f"typical_{channel}")

    if high >= 2:
        reasons.append("rule_two_or_more_strong_deviations")
        return LABEL_HIGH, reasons
    if fatigue >= 2:
        reasons.append("rule_two_or_more_deviations")
        return LABEL_MILD, reasons
    if typical >= MIN_EVIDENCE_CHANNELS and fatigue == 0:
        reasons.append("rule_two_or_more_typical_channels")
        return LABEL_ALERT, reasons
    reasons.append("uncertain_mixed_or_single_deviation")
    return LABEL_UNCERTAIN, reasons


def label_row(
    row: pd.Series,
    baselines: dict[Any, dict[str, pd.Series]],
) -> dict[str, Any]:
    user = row["user_anon"]
    user_tables = baselines.get(user, {})
    votes: dict[str, int] = {}
    notes: list[str] = []
    for channel in EVIDENCE_CHANNELS:
        value = channel_value(row, channel)
        if value is None:
            notes.append(f"missing_{channel}")
            continue
        series = user_tables.get(channel)
        if series is None:
            notes.append(f"no_user_baseline_{channel}")
            continue
        baseline = leave_one_out_median(series, row.name)
        if baseline is None:
            notes.append(f"insufficient_user_baseline_{channel}")
            continue
        votes[channel] = vote_channel(channel, value, baseline)

    label, reasons = decide_label(votes)
    reasons.extend(notes)
    # Explicitly record that idle/session were not used.
    reasons.append("idle_and_session_minutes_excluded_from_vote")
    return {
        "label": label,
        "label_reason": ";".join(reasons),
        "label_version": LABEL_VERSION,
        "n_evidence_channels": len(votes),
    }


def label_windows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        out = frame.copy()
        out["label"] = pd.Series(dtype=object)
        out["label_reason"] = pd.Series(dtype=object)
        out["label_version"] = pd.Series(dtype=object)
        return out
    required = {"user_anon", *EVIDENCE_CHANNELS}
    missing_cols = [name for name in required if name not in frame.columns]
    if missing_cols:
        raise ValueError(f"windows table missing columns {missing_cols}")
    baselines = user_baselines(frame)
    labelled_rows = [label_row(row, baselines) for _, row in frame.iterrows()]
    extra = pd.DataFrame(labelled_rows, index=frame.index)
    out = frame.copy()
    out["label"] = extra["label"]
    out["label_reason"] = extra["label_reason"]
    out["label_version"] = extra["label_version"]
    if len(out) != len(frame):
        raise RuntimeError("labelling dropped windows")
    return out


def label_report(labelled: pd.DataFrame, source: str) -> dict[str, Any]:
    counts = labelled["label"].value_counts(dropna=False).to_dict() if not labelled.empty else {}
    missing = {
        name: int(labelled[name].isna().sum()) if name in labelled.columns else None
        for name in FEATURE_NAMES
    }
    per_user = {}
    if not labelled.empty:
        for user, group in labelled.groupby("user_anon"):
            per_user[str(user)] = {
                "windows": int(len(group)),
                "labels": {str(k): int(v) for k, v in group["label"].value_counts().items()},
            }
    per_session = {}
    if not labelled.empty and "session_anon" in labelled.columns:
        session_counts = labelled.groupby("session_anon")["label"].value_counts().unstack(fill_value=0)
        per_session = {
            "n_sessions": int(labelled["session_anon"].nunique()),
            "sessions_with_high_fatigue": int(
                (labelled.groupby("session_anon")["label"].apply(lambda s: (s == LABEL_HIGH).any())).sum()
            ),
            "sessions_with_only_uncertain": int(
                labelled.groupby("session_anon")["label"].apply(lambda s: set(s) == {LABEL_UNCERTAIN}).sum()
            ),
            "label_totals_across_sessions": {
                col: int(session_counts[col].sum()) for col in session_counts.columns
            },
        }
    return {
        "disclaimer": DISCLAIMER,
        "label_version": LABEL_VERSION,
        "source_csv": source,
        "n_windows": int(len(labelled)),
        "windows_discarded": 0,
        "label_distribution": {str(k): int(v) for k, v in counts.items()},
        "missing_feature_counts": missing,
        "per_user": per_user,
        "per_session": per_session,
        "rules": {
            "baseline": (
                f"leave-one-out median per user; channel unused if fewer than {MIN_BASELINE_N} "
                "other windows have that feature"
            ),
            "min_evidence_channels": MIN_EVIDENCE_CHANNELS,
            "idle_time_sec": "never votes",
            "session_minutes": "never votes",
            "missing_values": "left as NA; not replaced with zero",
            "mean_key_interval_ms": "not a vote (duplicate of typing_speed_kpm)",
            "mouse_distance_px": "not a vote (duplicate of mouse_speed_px_per_sec)",
            "typing_speed_ignored_if": (
                f"n_keydown < {MIN_KEYS_FOR_TYPING} or WPM > {MAX_PLAUSIBLE_WPM} (key-repeat)"
            ),
            "high_fatigue": "two or more strong deviations",
            "mild_fatigue": "two or more deviations, fewer than two strong",
            "alert": "at least two typical channels and no fatigue votes",
            "uncertain": "otherwise, including a single deviation",
        },
    }


def prepare(
    windows_csv: Path,
    out_csv: Path,
    out_report: Path,
) -> dict[str, Any]:
    frame = pd.read_csv(windows_csv)
    labelled = label_windows(frame)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(out_csv, index=False)
    report = label_report(labelled, source=str(windows_csv))
    report["output_csv"] = str(out_csv)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assign heuristic (not ground-truth) labels to EVTRACK windows."
    )
    parser.add_argument(
        "--windows",
        type=Path,
        default=ROOT / "data" / "evtrack_behaviour_windows.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "evtrack_behaviour_labelled.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "evtrack_behaviour_label_report.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = prepare(args.windows, args.out, args.report)
    print("Wrote", args.out, "windows", report["n_windows"])
    print("Wrote", args.report)
    print("Disclaimer:", DISCLAIMER)
    print("Labels:", report["label_distribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
