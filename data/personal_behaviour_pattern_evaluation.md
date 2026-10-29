# Personal behavior-pattern prototype (exploratory)

Exploratory personal behavior-pattern summary linked to self-reported sleepiness. This is not a trained fatigue classifier, does not predict sleepiness, and is not evidence of population-level or clinical accuracy.

- Real rated sessions used: 15
- Quality-ready sessions used for feature patterns: 12
- Self-reported ratings: 15 (scale 1–9)
- Synthetic sessions excluded: 90 across 9 source sessions
- Quality status: {'ready': 12, 'insufficient': 3}
- Missing features (all real rated sessions): {'typing_speed_kpm': 0, 'backspace_rate': 0, 'mean_key_interval_ms': 0, 'std_key_interval_ms': 0, 'mouse_distance_px': 0, 'mouse_speed_px_per_sec': 0, 'click_rate_per_min': 2, 'mean_click_latency_ms': 0, 'idle_time_sec': 0, 'session_minutes': 0}
- Missing features (quality-ready pattern sessions): {'typing_speed_kpm': 0, 'backspace_rate': 0, 'mean_key_interval_ms': 0, 'std_key_interval_ms': 0, 'mouse_distance_px': 0, 'mouse_speed_px_per_sec': 0, 'click_rate_per_min': 0, 'mean_click_latency_ms': 0, 'idle_time_sec': 0, 'session_minutes': 0}

## Exploratory associations

Spearman correlations are descriptive only; they are not significance or accuracy claims.

| Feature | Paired sessions | Spearman rho with self-rating |
|---|---:|---:|
| typing_speed_kpm | 12 | -0.587 |
| backspace_rate | 12 | 0.625 |
| std_key_interval_ms | 12 | 0.272 |
| mouse_speed_px_per_sec | 12 | 0.845 |
| click_rate_per_min | 12 | not interpreted (active-span rate) |
| mean_click_latency_ms | 12 | -0.032 |
| idle_time_sec | 12 | 0.544 |
| session_minutes | 12 | 0.159 |

## Data quality notes

- All 15 self-ratings are reported, but feature associations and band medians use only collector quality-ready sessions.
- click_rate_per_min is clicks divided by clustered active-click duration, not wall-clock session time; extreme values are retained but its rating association is not interpreted.
- Only 15 sessions from one person are available; rating-band summaries are small descriptive groups, not validated states.
- mean_key_interval_ms duplicates typing_speed_kpm and mouse_distance_px duplicates mouse_speed_px_per_sec; aliases are omitted from pattern associations.

## Why no classifier was trained

Only 15 self-rated sessions from one person are available. Session-level holdout would leave too little data to evaluate a fatigue classifier, and the 90 synthetic rows have no sleepiness ratings and are excluded.
