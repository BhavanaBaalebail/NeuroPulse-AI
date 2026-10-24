#!/usr/bin/env python3
"""Build leakage-safe train/val/test manifests from webcam folders."""

from pathlib import Path

from vision.dataset_split import prepare_webcam_manifest


def main() -> None:
    root = Path(__file__).resolve().parent
    dest = root / "data" / "splits" / "webcam_manifest.json"
    split = prepare_webcam_manifest(root / "data" / "train", dest)
    print(f"Wrote {dest}")
    for name, rows in split.items():
        n_src = len({r["source"] for r in rows})
        n_d = sum(1 for r in rows if r["label"] == 1)
        n_a = sum(1 for r in rows if r["label"] == 0)
        print(f"  {name:5s}  images={len(rows):4d}  sources={n_src:3d}  drowsy={n_d:3d}  notdrowsy={n_a:3d}")


if __name__ == "__main__":
    main()
