"""Source-grouped dataset splits that keep augmented copies with their origin image."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_IMG = re.compile(r"^img_(\d+)$", re.I)
_AUG_DROWSY = re.compile(r"^aug_drowsy_(\d+)$", re.I)
_AUG_AWAKE = re.compile(r"^aug_awake_(\d+)$", re.I)
_AUG_GENERIC = re.compile(r"^aug_([^_]+)_(\d+)$", re.I)
_MRL = re.compile(r"^(s\d{4})_", re.I)
_WEBCAM_CROP = re.compile(r"^webcam_(open|closed)_capture_(.+)_(left|right)$", re.I)


def source_key(path: Path, class_name: str) -> str:
    stem = path.stem
    m = _MRL.match(stem)
    if m:
        return f"mrl:subject:{m.group(1).lower()}"
    m = _WEBCAM_CROP.match(stem)
    if m:
        return f"webcam:{m.group(2).lower()}"
    m = _IMG.match(stem)
    if m:
        return f"{class_name}:capture:{m.group(1)}"
    m = _AUG_DROWSY.match(stem)
    if m:
        return f"{class_name}:capture:{m.group(1)}"
    m = _AUG_AWAKE.match(stem)
    if m:
        # notdrowsy augs do not share ids with img_* in this repo; still group same-id augs.
        return f"{class_name}:aug_awake:{m.group(1)}"
    m = _AUG_GENERIC.match(stem)
    if m:
        return f"{class_name}:aug_{m.group(1)}:{m.group(2)}"
    return f"{class_name}:file:{stem}"


def list_class_images(class_dir: Path, class_name: str, label: int) -> list[dict]:
    rows = []
    if not class_dir.is_dir():
        return rows
    for path in sorted(class_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        rows.append(
            {
                "path": str(path.resolve()),
                "label": int(label),
                "class_name": class_name,
                "source": source_key(path, class_name),
                "filename": path.name,
            }
        )
    return rows


def collect_webcam_rows(root: Path) -> list[dict]:
    rows = []
    rows.extend(list_class_images(root / "drowsy", "drowsy", 1))
    rows.extend(list_class_images(root / "notdrowsy", "notdrowsy", 0))
    if not rows:
        raise ValueError(f"No images under {root}/drowsy or {root}/notdrowsy")
    return rows


def grouped_split(
    rows: Iterable[dict],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)

    by_label: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for source, items in groups.items():
        label_signature = tuple(sorted({int(item["label"]) for item in items}))
        by_label[label_signature].append(source)

    rng = random.Random(seed)
    split_of: dict[str, str] = {}
    for label_signature, sources in by_label.items():
        srcs = list(sources)
        rng.shuffle(srcs)
        n = len(srcs)
        n_test = int(round(n * test_frac)) if test_frac > 0 else 0
        n_val = int(round(n * val_frac)) if val_frac > 0 else 0
        if n >= 3 and test_frac > 0:
            n_test = max(1, n_test)
        if n >= 3 and val_frac > 0:
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            n_test = max(0, min(n_test, n - 2))
            n_val = max(0, min(n_val, n - n_test - 1))
        test_src = srcs[:n_test]
        val_src = srcs[n_test : n_test + n_val]
        train_src = srcs[n_test + n_val :]
        for s in train_src:
            split_of[s] = "train"
        for s in val_src:
            split_of[s] = "val"
        for s in test_src:
            split_of[s] = "test"

    out = {"train": [], "val": [], "test": []}
    for source, items in groups.items():
        bucket = split_of.get(source, "train")
        for item in items:
            rec = dict(item)
            rec["split"] = bucket
            out[bucket].append(rec)
    return out


def sources_overlap(a: list[dict], b: list[dict]) -> set[str]:
    return {r["source"] for r in a} & {r["source"] for r in b}


def write_manifest(
    split: dict[str, list[dict]], dest: Path, class_names: list[str] | None = None
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    names = class_names or ["notdrowsy", "drowsy"]
    payload = {
        "description": "Source-grouped splits. All crops and augmentations of one source stay in one split.",
        "class_names": names,
        "label_map": {name: index for index, name in enumerate(names)},
        "counts": {k: len(v) for k, v in split.items()},
        "source_counts": {k: len({r['source'] for r in v}) for k, v in split.items()},
        "splits": split,
    }
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def prepare_webcam_manifest(
    data_root: Path,
    dest: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    rows = collect_webcam_rows(data_root)
    split = grouped_split(rows, val_frac=val_frac, test_frac=test_frac, seed=seed)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sources_overlap(split[a], split[b])
        if overlap:
            raise RuntimeError(f"source leakage between {a} and {b}: {sorted(overlap)[:5]}")
    write_manifest(split, dest)
    return split
