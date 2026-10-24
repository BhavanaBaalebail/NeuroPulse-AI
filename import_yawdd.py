#!/usr/bin/env python3
"""Record provenance for a locally downloaded YawDD tree. Does not scrape IEEE DataPort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder of YawDD AVI files you downloaded manually")
    args = ap.parse_args()
    src = Path(args.root)
    if not src.is_dir():
        raise SystemExit(f"Not a directory: {src}")
    videos = [p for p in src.rglob("*") if p.suffix.lower() in {".avi", ".mp4", ".mov"}]
    dest = ROOT / "data" / "raw_external" / "yawdd_provenance.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "YawDD: Yawning Detection Dataset",
        "doi": "10.21227/e1qm-hb90",
        "url": "https://doi.org/10.21227/e1qm-hb90",
        "access": "IEEE DataPort login required; this script does not download.",
        "license_notes": (
            "Research use of videos. Some subjects forbid public screenshots. "
            "See the participants information file in the official download."
        ),
        "labels": "Official videos are unlabeled; talking vs yawning must be annotated locally.",
        "source_root": str(src.resolve()),
        "video_count": len(videos),
        "videos_sample": [str(p.relative_to(src)) for p in videos[:20]],
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    if not videos:
        print("No video files found yet — provenance recorded anyway.")


if __name__ == "__main__":
    main()
