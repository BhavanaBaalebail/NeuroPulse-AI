#!/usr/bin/env python3
"""Evaluate the existing ResNet checkpoint on a source-grouped holdout (no aug leakage)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from vision.dataset_split import prepare_webcam_manifest

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 160
HEAD_DROPOUT = 0.35
ROOT = Path(__file__).resolve().parent


class ManifestDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        img = cv2.imread(rec["path"])
        if img is None:
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img)
        return x, torch.tensor(rec["label"], dtype=torch.long)


def build_model(device):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(m.fc.in_features, 2))
    return m.to(device)


def metrics_from_preds(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(
        y_true,
        y_pred,
        target_names=["notdrowsy (0)", "drowsy (1)"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    alert_n = max(tn + fp, 1)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": cm.tolist(),
        "false_positive_rate_alert": float(fp / alert_n),
        "per_class": report,
        "n": int(len(y_true)),
    }


@torch.no_grad()
def run_split(model, rows, device, tta: bool) -> dict:
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    loader = DataLoader(ManifestDataset(rows, tf), batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    ys, yh = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if tta:
            logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
        ys.extend(y.numpy().tolist())
        yh.extend(logits.argmax(1).cpu().numpy().tolist())
    y_true = np.array(ys)
    y_pred = np.array(yh)
    return metrics_from_preds(y_true, y_pred)


def main():
    device = torch.device("cpu")
    manifest_path = ROOT / "data" / "splits" / "webcam_manifest.json"
    prepare_webcam_manifest(ROOT / "data" / "train", manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    weights = ROOT / "best_model_resnet.pth"
    if not weights.is_file():
        raise SystemExit(f"Missing {weights}")

    model = build_model(device)
    try:
        sd = torch.load(weights, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(weights, map_location=device)
    model.load_state_dict(sd, strict=True)

    out = {
        "weights": str(weights),
        "note": (
            "Holdout is source-grouped. This checkpoint was trained with a leaky "
            "file-level split, so these numbers can be pessimistic or still optimistic "
            "depending on leftover correlation. They are not a webcam field trial."
        ),
        "splits": {},
    }
    print(json.dumps({"counts": manifest["counts"], "source_counts": manifest["source_counts"]}, indent=2))
    for name in ("val", "test"):
        rows = manifest["splits"][name]
        stats = run_split(model, rows, device, tta=True)
        out["splits"][name] = stats
        print(
            f"\n=== {name} (grouped, TTA) acc={stats['accuracy']:.4f} "
            f"FPR_alert={stats['false_positive_rate_alert']:.4f} ==="
        )
        print("confusion", stats["confusion_matrix"])
        print(
            json.dumps(
                {
                    k: v
                    for k, v in stats["per_class"].items()
                    if k in ("notdrowsy (0)", "drowsy (1)", "accuracy")
                },
                indent=2,
            )
        )

    dest = ROOT / "eval_grouped_resnet.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
