"""Temporal mouth logic: open vs speech-like movement vs candidate yawn.

MAR alone cannot reliably separate speech from yawning. A candidate yawn is
only emitted when the mouth stays wide open for a sustained interval.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from vision.config import VisionConfig


@dataclass
class MouthObservation:
    mar: Optional[float]
    mouth_class: str  # closed | open | speaking | candidate_yawn | unknown
    open_duration_s: float
    candidate_yawn: bool
    recent_yawn: bool
    note: str


class MouthStateTracker:
    def __init__(self, config: VisionConfig | None = None, now_fn=None):
        self.config = config or VisionConfig()
        self._now = now_fn
        self._open_started_at: Optional[float] = None
        self._wide_started_at: Optional[float] = None
        self._history: deque[tuple[float, str]] = deque()
        self._last_yawn_at: Optional[float] = None
        self._yawn_latched = False

    def _clock(self, timestamp: Optional[float]) -> float:
        if timestamp is not None:
            return float(timestamp)
        if self._now is None:
            raise ValueError("timestamp is required when now_fn is not set")
        return float(self._now())

    def update(self, mar: Optional[float], timestamp: Optional[float] = None) -> MouthObservation:
        t = self._clock(timestamp)
        cfg = self.config
        if mar is None:
            self._open_started_at = None
            self._wide_started_at = None
            return MouthObservation(
                mar=None,
                mouth_class="unknown",
                open_duration_s=0.0,
                candidate_yawn=False,
                recent_yawn=self._recent_yawn(t),
                note="missing_mouth_landmarks",
            )

        if mar >= cfg.mouth_open_mar:
            if self._open_started_at is None:
                self._open_started_at = t
        else:
            self._open_started_at = None

        if mar >= cfg.yawn_mar:
            if self._wide_started_at is None:
                self._wide_started_at = t
        else:
            self._wide_started_at = None
            self._yawn_latched = False

        open_duration = (t - self._open_started_at) if self._open_started_at is not None else 0.0
        wide_duration = (t - self._wide_started_at) if self._wide_started_at is not None else 0.0

        candidate = wide_duration >= cfg.yawn_min_s
        if candidate and not self._yawn_latched:
            self._last_yawn_at = t
            self._yawn_latched = True

        if mar < cfg.speech_mar_low:
            mouth_class = "closed"
            note = "mouth_closed"
        elif candidate:
            mouth_class = "candidate_yawn"
            note = f"sustained_wide_open_{wide_duration:.2f}s"
        elif cfg.speech_mar_low <= mar <= cfg.speech_mar_high and open_duration <= cfg.speech_max_open_s:
            mouth_class = "speaking"
            note = "short_mouth_opening_possible_speech"
        elif mar >= cfg.mouth_open_mar:
            mouth_class = "open"
            note = "mouth_open_not_yet_yawn"
        else:
            mouth_class = "closed"
            note = "mouth_mostly_closed"

        self._history.append((t, mouth_class))
        cutoff = t - cfg.mouth_history_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        return MouthObservation(
            mar=mar,
            mouth_class=mouth_class,
            open_duration_s=open_duration,
            candidate_yawn=candidate,
            recent_yawn=self._recent_yawn(t),
            note=note,
        )

    def _recent_yawn(self, t: float) -> bool:
        if self._last_yawn_at is None:
            return False
        return (t - self._last_yawn_at) <= self.config.mouth_history_s


class YawnClassifierInterface:
    """Placeholder for a future video yawn model (e.g. trained on YawDD)."""

    def predict(self, frames) -> dict:
        return {
            "available": False,
            "label": None,
            "score": None,
            "missing": "No dedicated yawn video model is installed. MAR duration is used as a candidate only.",
        }
