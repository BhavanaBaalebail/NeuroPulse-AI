"""Preprocess EVTRACK keyboard/mouse logs into NeuroPulse behaviour windows.

Does not train models, invent fatigue labels, or store typed text.
Key identity is kept as numeric keyCode only (Backspace = DOM keyCode 8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from logger import (  # noqa: E402
    CHARS_PER_WORD,
    CLICK_ACTIVE_S,
    CLICK_GESTURE_MS,
    FEATURE_NAMES,
    LIVE_IDLE_CAP_SEC,
    LIVE_WINDOW_SECONDS,
    MIN_MOVE_PX,
    MOUSE_ACTIVE_S,
    TYPING_PAUSE_MS,
    _active_duration,
    _active_motion,
)

ANON_SALT = "evtrack-anon-v1"
BACKSPACE_KEYCODE = 8  # DOM KeyboardEvent.keyCode Backspace
SESSION_SPLIT_GAP_S = 300.0
# Naive file timestamps are converted via this epoch so local TZ does not shift them.
_NAIVE_EPOCH = pd.Timestamp("1970-01-01")
ACTIVITY_EVENTS = frozenset({"keydown", "mousemove", "mousedown"})
MOUSE_PATH_EVENTS = frozenset({"mousemove"})
CLICK_EVENTS = frozenset({"mousedown"})
INFO_COLUMNS = [
    "_id",
    "user",
    "session_id",
    "date",
    "ua",
    "screen",
    "screenh",
    "windoww",
    "windowh",
    "documentw",
    "documenth",
]
TRACK_COLUMNS = [
    "_id",
    "cursor",
    "timestamp",
    "xpos",
    "ypos",
    "key",
    "event",
    "user",
    "session_id",
]
FEATURE_UNITS = {
    "typing_speed_kpm": "WPM from keydown count / 5 chars-per-word over burst typing seconds (live logger convention). Not keys/min.",
    "backspace_rate": "Backspace keydowns (keyCode 8) / keydowns in the window.",
    "mean_key_interval_ms": "60000 / typing_speed_kpm, matching live/training name. Not the raw mean inter-keydown ms.",
    "std_key_interval_ms": "pstdev of inter-keydown gaps <= TYPING_PAUSE_MS. Training used a Monkeytype consistency proxy.",
    "mouse_distance_px": "clustered mouse speed * 60 (one-minute equivalent), matching live snapshot, not raw path pixels.",
    "mouse_speed_px_per_sec": "significant mousemove path / clustered active seconds (gaps <= 0.5s, step >= 3px).",
    "click_rate_per_min": "mousedown count * 60 / clustered click-active seconds (gaps <= 10s).",
    "mean_click_latency_ms": "mean mousemove→mousedown interval among gestures <= 420ms. NaN if none.",
    "idle_time_sec": "max in-window gap among keydown/mousemove/mousedown, capped at 30s.",
    "session_minutes": "minutes from episode start to window end (wall clock), not a rate.",
}

# Codes sometimes confused with Backspace; counted in the quality report only.
OTHER_DELETE_KEYCODES = (14, 42, 67, 111)


def naive_seconds(ts: pd.Timestamp) -> float:
    # TRACK timestamps are millisecond precision; round so float conversion does not drift.
    return round(float((pd.Timestamp(ts) - _NAIVE_EPOCH) / pd.Timedelta(seconds=1)), 3)


def seconds_to_naive_iso(value: float) -> str:
    stamp = _NAIVE_EPOCH + pd.Timedelta(milliseconds=round(float(value) * 1000.0))
    return stamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def anonymize(value: Any) -> str:
    text = str(value)
    digest = hashlib.sha256(f"{ANON_SALT}|{text}".encode("utf-8")).hexdigest()
    return digest[:12]


def detect_separator(path: Path) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    if "\t" in header:
        return "\t"
    if "," in header:
        return ","
    raise ValueError(f"Could not detect a tab or comma separator in {path}")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def load_evtrack_info(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=detect_separator(path))
    missing = [name for name in INFO_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}; found {list(frame.columns)}")
    frame["date_parsed"] = parse_timestamp_series(frame["date"])
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce")
    return frame


def load_evtrack_track(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=detect_separator(path))
    missing = [name for name in TRACK_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}; found {list(frame.columns)}")
    frame["timestamp_parsed"] = parse_timestamp_series(frame["timestamp"])
    frame["session_id"] = pd.to_numeric(frame["session_id"], errors="coerce")
    frame["key"] = pd.to_numeric(frame["key"], errors="coerce")
    frame["xpos"] = pd.to_numeric(frame["xpos"], errors="coerce")
    frame["ypos"] = pd.to_numeric(frame["ypos"], errors="coerce")
    frame["event"] = frame["event"].astype(str)
    return frame


def pair_keydown_keyup(events: list[dict]) -> dict[str, Any]:
    pending: dict[int, deque] = defaultdict(deque)
    holds_ms: list[float] = []
    unmatched_down = 0
    extra_up = 0
    for event in events:
        code = event.get("key")
        if code is None or (isinstance(code, float) and math.isnan(code)):
            continue
        code = int(code)
        name = event["event"]
        ts = event["t"]
        if name == "keydown":
            pending[code].append(ts)
        elif name == "keyup":
            queue = pending[code]
            if queue:
                start = queue.popleft()
                holds_ms.append((ts - start) * 1000.0)
            else:
                extra_up += 1
    unmatched_down = sum(len(queue) for queue in pending.values())
    return {
        "paired": len(holds_ms),
        "unmatched_keydown": unmatched_down,
        "unmatched_keyup": extra_up,
        "median_hold_ms": float(statistics.median(holds_ms)) if holds_ms else None,
    }


def split_episodes(
    events: list[dict],
    gap_s: float = SESSION_SPLIT_GAP_S,
) -> list[list[dict]]:
    if not events:
        return []
    ordered = sorted(events, key=lambda row: row["t"])
    episodes = [[ordered[0]]]
    for event in ordered[1:]:
        if event["t"] - episodes[-1][-1]["t"] > gap_s:
            episodes.append([event])
        else:
            episodes[-1].append(event)
    return episodes


def _window_events(events: list[dict], start: float, end: float) -> list[dict]:
    return [event for event in events if start <= event["t"] < end]


def features_from_events(
    events: list[dict],
    *,
    episode_start: float,
    window_end: float,
    window_seconds: float = LIVE_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Derive the 10 live-logger features. Unavailable measurements are None, not 0."""
    keydowns = [event for event in events if event["event"] == "keydown"]
    moves = [
        event
        for event in events
        if event["event"] in MOUSE_PATH_EVENTS
        and event.get("xpos") is not None
        and event.get("ypos") is not None
        and not math.isnan(float(event["xpos"]))
        and not math.isnan(float(event["ypos"]))
    ]
    clicks = [event for event in events if event["event"] in CLICK_EVENTS]
    activity = [event for event in events if event["event"] in ACTIVITY_EVENTS]

    key_count = len(keydowns)
    backspaces = sum(1 for event in keydowns if int(event.get("key") or -1) == BACKSPACE_KEYCODE)
    key_times = [event["t"] for event in keydowns]
    intervals_ms = [
        (key_times[i] - key_times[i - 1]) * 1000.0 for i in range(1, len(key_times))
    ]
    burst_intervals = [gap for gap in intervals_ms if gap <= TYPING_PAUSE_MS]
    typing_span = sum(burst_intervals) / 1000.0
    typing_speed = None
    mean_interval = None
    std_interval = None
    backspace_rate = None
    if key_count > 0 and typing_span >= 1e-3:
        typing_speed = (key_count / CHARS_PER_WORD) * 60.0 / typing_span
        mean_interval = 60000.0 / max(typing_speed, 1.0)
        backspace_rate = backspaces / key_count
    elif key_count > 0:
        backspace_rate = backspaces / key_count
    if len(burst_intervals) >= 2:
        std_interval = statistics.pstdev(burst_intervals)

    move_deltas: list[tuple[float, float]] = []
    prev = None
    for event in moves:
        point = (float(event["xpos"]), float(event["ypos"]))
        if prev is not None:
            dist = math.hypot(point[0] - prev[0], point[1] - prev[1])
            move_deltas.append((event["t"], dist))
        prev = point
    significant = [(t, dist) for t, dist in move_deltas if dist >= MIN_MOVE_PX]
    significant_distance, mouse_active = _active_motion(significant, MOUSE_ACTIVE_S)
    mouse_speed = None
    mouse_distance_minute = None
    if len(significant) >= 2 and mouse_active > 1e-3 and significant_distance > 0.0:
        mouse_speed = significant_distance / mouse_active
        mouse_distance_minute = mouse_speed * 60.0

    click_times = [event["t"] for event in clicks]
    click_active = _active_duration(click_times, CLICK_ACTIVE_S)
    click_rate = None
    if len(clicks) >= 2 and click_active > 1e-3:
        click_rate = len(clicks) * 60.0 / click_active
    elif len(clicks) >= 1 and window_seconds > 1e-3:
        # Single click has no clustered span; rate is not estimated.
        click_rate = None

    move_times = [event["t"] for event in moves]
    latencies: list[float] = []
    for click in clicks:
        prior = [t for t in move_times if t <= click["t"]]
        if not prior:
            continue
        latency_ms = (click["t"] - prior[-1]) * 1000.0
        if 0.0 <= latency_ms <= CLICK_GESTURE_MS:
            latencies.append(latency_ms)
    click_latency = statistics.mean(latencies) if latencies else None

    activity_times = sorted(event["t"] for event in activity)
    internal_gaps = [
        activity_times[i] - activity_times[i - 1] for i in range(1, len(activity_times))
    ]
    idle_raw = max(internal_gaps) if internal_gaps else None
    idle = min(idle_raw, LIVE_IDLE_CAP_SEC) if idle_raw is not None else None

    session_minutes = max(0.0, window_end - episode_start) / 60.0

    return {
        "typing_speed_kpm": typing_speed,
        "backspace_rate": backspace_rate,
        "mean_key_interval_ms": mean_interval,
        "std_key_interval_ms": std_interval,
        "mouse_distance_px": mouse_distance_minute,
        "mouse_speed_px_per_sec": mouse_speed,
        "click_rate_per_min": click_rate,
        "mean_click_latency_ms": click_latency,
        "idle_time_sec": idle,
        "session_minutes": session_minutes,
        "n_keydown": key_count,
        "n_backspace": backspaces,
        "n_mousemove": len(moves),
        "n_mousedown": len(clicks),
        "n_events": len(events),
        "n_burst_intervals": len(burst_intervals),
        "n_click_latencies": len(latencies),
    }


def windows_for_episode(
    events: list[dict],
    *,
    window_seconds: float = LIVE_WINDOW_SECONDS,
) -> list[dict]:
    if not events:
        return []
    ordered = sorted(events, key=lambda row: row["t"])
    start = ordered[0]["t"]
    end = ordered[-1]["t"]
    rows = []
    window_start = start
    while window_start <= end:
        window_end = window_start + window_seconds
        slice_events = _window_events(ordered, window_start, window_end)
        if slice_events:
            features = features_from_events(
                slice_events,
                episode_start=start,
                window_end=min(window_end, end + 1e-9),
                window_seconds=window_seconds,
            )
            rows.append(
                {
                    "window_start_s": window_start,
                    "window_end_s": window_end,
                    **features,
                }
            )
        window_start = window_end
    return rows


def track_rows_to_events(frame: pd.DataFrame) -> list[dict]:
    events = []
    for row in frame.itertuples(index=False):
        ts = row.timestamp_parsed
        if pd.isna(ts):
            continue
        events.append(
            {
                "t": naive_seconds(ts),
                "timestamp": seconds_to_naive_iso(naive_seconds(ts)),
                "event": str(row.event),
                "key": None if pd.isna(row.key) else int(row.key),
                "xpos": None if pd.isna(row.xpos) else float(row.xpos),
                "ypos": None if pd.isna(row.ypos) else float(row.ypos),
                "user": row.user,
                "session_id": None if pd.isna(row.session_id) else int(row.session_id),
            }
        )
    return events


def inspect_frames(info: pd.DataFrame, track: pd.DataFrame) -> dict[str, Any]:
    track_ok = track.dropna(subset=["timestamp_parsed"])
    info_ok = info.dropna(subset=["date_parsed"])
    event_counts = track["event"].value_counts().to_dict()
    users = sorted(set(info["user"].astype(str)) | set(track["user"].astype(str)))
    info_sessions = set(info["session_id"].dropna().astype(int))
    track_sessions = set(track["session_id"].dropna().astype(int))
    gaps = []
    for _, group in track_ok.groupby("session_id"):
        times = group["timestamp_parsed"].sort_values()
        delta = times.diff().dt.total_seconds().dropna()
        if not delta.empty:
            gaps.append(delta)
    gap_series = pd.concat(gaps, ignore_index=True) if gaps else pd.Series(dtype=float)
    merged = info_ok.merge(
        track_ok.groupby("session_id")["timestamp_parsed"].min().rename("track_min"),
        on="session_id",
        how="inner",
    )
    offset = (merged["date_parsed"] - merged["track_min"]).dt.total_seconds()
    keydowns = track_ok[track_ok["event"] == "keydown"]
    other_delete = {
        str(code): int((keydowns["key"] == code).sum()) for code in OTHER_DELETE_KEYCODES
    }
    return {
        "info_schema": list(info.columns),
        "track_schema": list(track.columns),
        "info_separator_note": "Tab-separated despite .csv extension",
        "track_separator_note": "Tab-separated despite .csv extension",
        "info_rows": int(len(info)),
        "track_rows": int(len(track)),
        "info_nulls": {k: int(v) for k, v in info[INFO_COLUMNS].isna().sum().items()},
        "track_nulls": {k: int(v) for k, v in track[TRACK_COLUMNS].isna().sum().items()},
        "info_unparsed_dates": int(info["date_parsed"].isna().sum()),
        "track_unparsed_timestamps": int(track["timestamp_parsed"].isna().sum()),
        "info_timestamp_format": "YYYY-MM-DD HH:MM:SS.ffffff (microseconds)",
        "track_timestamp_format": "YYYY-MM-DD HH:MM:SS.mmm (milliseconds)",
        "info_date_range": [
            None if info_ok.empty else str(info_ok["date_parsed"].min()),
            None if info_ok.empty else str(info_ok["date_parsed"].max()),
        ],
        "track_timestamp_range": [
            None if track_ok.empty else str(track_ok["timestamp_parsed"].min()),
            None if track_ok.empty else str(track_ok["timestamp_parsed"].max()),
        ],
        "event_types": {str(k): int(v) for k, v in event_counts.items()},
        "unique_users": users,
        "n_users": len(users),
        "n_info_sessions": len(info_sessions),
        "n_track_sessions": len(track_sessions),
        "sessions_in_info_without_track": len(info_sessions - track_sessions),
        "sessions_in_track_without_info": len(track_sessions - info_sessions),
        "intra_session_gap_seconds": {
            "n": int(len(gap_series)),
            "median": None if gap_series.empty else float(gap_series.median()),
            "p95": None if gap_series.empty else float(gap_series.quantile(0.95)),
            "max": None if gap_series.empty else float(gap_series.max()),
            "frac_gt_2s": None if gap_series.empty else float((gap_series > 2).mean()),
            "frac_gt_10s": None if gap_series.empty else float((gap_series > 10).mean()),
            "frac_gt_60s": None if gap_series.empty else float((gap_series > 60).mean()),
            "frac_gt_300s": None if gap_series.empty else float((gap_series > 300).mean()),
        },
        "info_minus_track_start_seconds": {
            "median": None if offset.empty else float(offset.median()),
            "note": "INFO.date is ~1 hour after the first TRACK timestamp for matched sessions; event times use TRACK.",
        },
        "keydown_n": int(len(keydowns)),
        "backspace_keycode_8_n": int((keydowns["key"] == BACKSPACE_KEYCODE).sum()),
        "keydown_key_0_n": int((keydowns["key"] == 0).sum()),
        "other_delete_like_keycodes_not_counted_as_backspace": other_delete,
        "mousedown_n": int((track_ok["event"] == "mousedown").sum()),
        "mouseup_n": int((track_ok["event"] == "mouseup").sum()),
        "mousemove_n": int((track_ok["event"] == "mousemove").sum()),
        "cursor_unique": int(track["cursor"].nunique()) if "cursor" in track.columns else None,
    }


def build_windows(track: pd.DataFrame, info: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    events = track_rows_to_events(track)
    by_session: dict[tuple[Any, Any], list[dict]] = defaultdict(list)
    for event in events:
        by_session[(event["user"], event["session_id"])].append(event)

    rows = []
    for (user, session_id), session_events in by_session.items():
        for episode_index, episode in enumerate(split_episodes(session_events)):
            pairing = pair_keydown_keyup(episode)
            for window in windows_for_episode(episode):
                row = {
                    "user_anon": anonymize(user),
                    "session_anon": anonymize(session_id),
                    "episode_index": episode_index,
                    "window_start": seconds_to_naive_iso(window["window_start_s"]),
                    "window_end": seconds_to_naive_iso(window["window_end_s"]),
                    "episode_start": seconds_to_naive_iso(episode[0]["t"]),
                    "window_seconds": LIVE_WINDOW_SECONDS,
                }
                for name in FEATURE_NAMES:
                    row[name] = window[name]
                row["n_keydown"] = window["n_keydown"]
                row["n_backspace"] = window["n_backspace"]
                row["n_mousemove"] = window["n_mousemove"]
                row["n_mousedown"] = window["n_mousedown"]
                row["n_events"] = window["n_events"]
                row["n_click_latencies"] = window["n_click_latencies"]
                row["episode_unmatched_keydown"] = pairing["unmatched_keydown"]
                row["episode_unmatched_keyup"] = pairing["unmatched_keyup"]
                rows.append(row)
    if not rows:
        columns = [
            "user_anon",
            "session_anon",
            "episode_index",
            "window_start",
            "window_end",
            "episode_start",
            "window_seconds",
            *FEATURE_NAMES,
            "n_keydown",
            "n_backspace",
            "n_mousemove",
            "n_mousedown",
            "n_events",
            "n_click_latencies",
            "episode_unmatched_keydown",
            "episode_unmatched_keyup",
        ]
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def quality_report(info: pd.DataFrame, track: pd.DataFrame, windows: pd.DataFrame) -> dict[str, Any]:
    inspection = inspect_frames(info, track)
    derivable = {}
    for name in FEATURE_NAMES:
        if windows.empty:
            n_present = 0
        else:
            n_present = int(windows[name].notna().sum())
        derivable[name] = {
            "unit": FEATURE_UNITS[name],
            "windows_with_value": n_present,
            "windows_total": int(len(windows)),
            "frac_available": None if windows.empty else float(n_present / len(windows)),
        }
    mismatches = [
        "typing_speed_kpm stores Monkeytype-style WPM, not keys per minute; EVTRACK uses keydown count.",
        "mean_key_interval_ms is 60000/WPM (live/training name), not the arithmetic mean of inter-key ms.",
        "std_key_interval_ms is burst inter-keydown pstdev; the hybrid training set used (100-consistency)×noise.",
        "mouse/click/idle/session in behavior_final_dataset.csv were sampled from a synthetic fatigue label, not telemetry.",
        "mean_click_latency_ms here is mousemove→mousedown ≤420ms. Training used a label-conditioned uniform draw.",
        "Backspace is DOM keyCode 8 only. Codes 14/42/67/111 appear in TRACK but are not counted as Backspace.",
        "keydown key=0 is retained as an anonymous keystroke (no character) and is not Backspace.",
        "Clicks use mousedown. mouseup is fewer (incomplete pairs) and is not counted as a second click.",
        "mouseover/mouseout/mousewheel are not used for distance, click rate, or idle.",
        "INFO.ua / screen sizes are not behaviour features and are not copied to the window CSV.",
        "No fatigue_label column is written.",
    ]
    unavailable = {
        "typed_text": "Never present: TRACK.key is numeric keyCode, not a character. Not stored.",
        "fatigue_label": "Not in EVTRACKINFO or EVTRACKTRACK. Not inferred.",
        "true_reaction_time": "No stimulus timestamps. Click latency is last mousemove→mousedown only.",
        "monkeytype_wpm_acc_consistency": "Not in these files.",
    }
    return {
        "inspection": inspection,
        "window_count": int(len(windows)),
        "window_seconds": LIVE_WINDOW_SECONDS,
        "session_split_gap_s": SESSION_SPLIT_GAP_S,
        "anonymisation": "SHA-256 prefix of salt|value; original user names and session ids omitted from the CSV.",
        "feature_availability": derivable,
        "definition_mismatches_vs_hybrid_training": mismatches,
        "unavailable": unavailable,
        "feature_order": list(FEATURE_NAMES),
    }


def prepare(
    track_path: Path,
    info_path: Path,
    out_csv: Path,
    out_report: Path,
) -> dict[str, Any]:
    info = load_evtrack_info(info_path)
    track = load_evtrack_track(track_path)
    windows = build_windows(track, info)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(out_csv, index=False)
    report = quality_report(info, track, windows)
    report["output_csv"] = str(out_csv)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive NeuroPulse behaviour windows from EVTRACK logs.")
    parser.add_argument("--track", type=Path, default=ROOT / "EVTRACKTRACK.csv")
    parser.add_argument("--info", type=Path, default=ROOT / "EVTRACKINFO.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "evtrack_behaviour_windows.csv")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "evtrack_behaviour_quality.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = prepare(args.track, args.info, args.out, args.report)
    print("Wrote", args.out, "windows", report["window_count"])
    print("Wrote", args.report)
    avail = report["feature_availability"]
    for name in FEATURE_NAMES:
        item = avail[name]
        print(f"  {name}: {item['windows_with_value']}/{item['windows_total']} windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
