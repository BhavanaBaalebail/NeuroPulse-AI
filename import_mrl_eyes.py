#!/usr/bin/env python3
"""Import a locally downloaded MRL Eye Dataset zip/folder into data/derived/mrl_eyes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "derived" / "mrl_eyes"


def parse_eye_state(name: str) -> str | None:
    """MRL filenames encode fields separated by underscores; eye state is field index 4 (0/1)."""
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 5:
        return None
    state = parts[4]
    if state == "0":
        return "closed"
    if state == "1":
        return "open"
    return None


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.is_file():
            yield path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw_external/mrlEyes_2018_01.zip", help="Path to mrlEyes zip")
    ap.add_argument("--root", default="", help="Already extracted dataset directory")
    ap.add_argument(
        "--max-per-subject",
        type=int,
        default=100,
        help="Max images per state per subject (0 for all ~85k images; 100 yields ~7.4k images across 37 subjects)",
    )
    args = ap.parse_args()

    open_dir = OUT / "open"
    closed_dir = OUT / "closed"
    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    counts = {"open": 0, "closed": 0, "skipped": 0}
    subject_counts: dict[str, dict[str, int]] = {}
    zpath = Path(args.zip)

    if zpath.is_file():
        print(f"Reading directly from zip: {zpath}")
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.infolist():
                if member.is_dir() or not member.filename.lower().endswith((".png", ".jpg")):
                    continue
                fname = Path(member.filename).name
                state = parse_eye_state(fname)
                if state is None:
                    counts["skipped"] += 1
                    continue
                parts = Path(fname).stem.split("_")
                subj = parts[0]
                if subj not in subject_counts:
                    subject_counts[subj] = {"open": 0, "closed": 0}
                if args.max_per_subject > 0 and subject_counts[subj][state] >= args.max_per_subject:
                    continue

                dest = (open_dir if state == "open" else closed_dir) / fname
                if not dest.exists():
                    with zf.open(member) as source, open(dest, "wb") as target:
                        shutil.copyfileobj(source, target)
                subject_counts[subj][state] += 1
                counts[state] += 1
    elif args.root and Path(args.root).is_dir():
        src_root = Path(args.root)
        print(f"Reading from directory: {src_root}")
        for img in iter_images(src_root):
            state = parse_eye_state(img.name)
            if state is None:
                counts["skipped"] += 1
                continue
            parts = img.stem.split("_")
            subj = parts[0]
            if subj not in subject_counts:
                subject_counts[subj] = {"open": 0, "closed": 0}
            if args.max_per_subject > 0 and subject_counts[subj][state] >= args.max_per_subject:
                continue

            dest = (open_dir if state == "open" else closed_dir) / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
            subject_counts[subj][state] += 1
            counts[state] += 1
    else:
        raise SystemExit(f"Zip or directory not found. Checked: {zpath}")

    provenance = {
        "dataset": "MRL Eye Dataset",
        "url": "https://mrl.cs.vsb.cz/eyedataset.html",
        "download_url": "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip",
        "local_source": str(zpath.resolve()) if zpath.is_file() else str(Path(args.root).resolve()),
        "archive_sha256": hashlib.sha256(zpath.read_bytes()).hexdigest() if zpath.is_file() else None,
        "modality": "infrared_eye_crops",
        "labels": {"0": "closed", "1": "open"},
        "label_source": "MRL filename field 5 (zero-based index 4); no inferred labels",
        "preprocessing": "Original image bytes copied unchanged into open/closed folders; no resizing, color conversion, or augmentation.",
        "subjects_total": len(subject_counts),
        "subjects": sorted(subject_counts.keys()),
        "counts": counts,
        "max_per_subject_filter": args.max_per_subject,
        "domain_gap": "IR crops vs RGB webcam; do not report IR metrics as webcam metrics.",
        "license_status": "The official dataset page offers public download but does not state an explicit reuse license. Confirm permitted use with the dataset maintainers before redistribution or commercial use.",
        "official_page_checked": "https://mrl.cs.vsb.cz/eyedataset.html",
        "citation": "Fusek, Pupil localization using geodesic distance (ISVC 2018)",
    }
    with open(OUT / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(json.dumps(provenance, indent=2))
    if counts["open"] + counts["closed"] == 0:
        raise SystemExit("No labeled MRL images found.")
    print(f"Successfully imported {counts['open']} open and {counts['closed']} closed images into {OUT}")


if __name__ == "__main__":
    main()
