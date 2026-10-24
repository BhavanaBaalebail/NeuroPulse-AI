#!/usr/bin/env python3
"""
Evaluate best_model_resnet.pth on the same stratified val split as train_resnet.py (seed=42).

Use this to see if problems are **model** (bad val confusion matrix) vs **deployment** (good val, bad webcam).

  python3 eval_resnet.py
  python3 eval_resnet.py --misclassified   # writes misclassified_val_samples.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

from vision.dataset_split import grouped_split, source_key

# Must match train_resnet.py / export_resnet_onnx.py
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 160
HEAD_DROPOUT = 0.35


def build_model_legacy(device):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m.to(device)


class FatigueDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        for img_file in sorted((self.root_dir / "drowsy").glob("*.*")):
            self.samples.append((str(img_file), 1))
        for img_file in sorted((self.root_dir / "notdrowsy").glob("*.*")):
            self.samples.append((str(img_file), 0))
        if not self.samples:
            raise ValueError(f"No images in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2

        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.tensor(label, dtype=torch.long)


def build_model(device):
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(in_f, 2))
    return m.to(device)


@torch.no_grad()
def predict_loader(model, loader, device, tta: bool):
    model.eval()
    ys, yh = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if tta:
            logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
        else:
            logits = model(x)
        pred = logits.argmax(1)
        ys.extend(y.cpu().numpy().tolist())
        yh.extend(pred.cpu().numpy().tolist())
    return np.array(ys), np.array(yh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train", help="Folder with drowsy/ and notdrowsy/")
    ap.add_argument("--weights", default="best_model_resnet.pth")
    ap.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip TTA")
    ap.add_argument("--misclassified", action="store_true", help="Write misclassified_val_samples.txt")
    args = ap.parse_args()
    tta = not args.no_tta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    full = FatigueDataset(args.data, transform=None)
    rows = []
    for path, label in full.samples:
        cls = "drowsy" if int(label) == 1 else "notdrowsy"
        rows.append({"path": path, "label": int(label), "source": source_key(Path(path), cls)})
    grouped = grouped_split(rows, val_frac=0.2, test_frac=0.0, seed=42)
    path_to_i = {full.samples[i][0]: i for i in range(len(full.samples))}
    va_i = [path_to_i[r["path"]] for r in grouped["val"]]

    va_ds = FatigueDataset(args.data, val_tf)
    va_sub = Subset(va_ds, va_i)

    batch_size = 16 if device.type == "cpu" else 32
    va_ld = DataLoader(va_sub, batch_size=batch_size, shuffle=False, num_workers=0)

    wpath = Path(args.weights)
    if not wpath.is_file():
        raise SystemExit(f"Missing weights: {wpath.resolve()}")
    try:
        sd = torch.load(wpath, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(wpath, map_location=device)

    model = build_model(device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        model = build_model_legacy(device)
        model.load_state_dict(sd, strict=True)
        print("(Loaded legacy head: Linear only, no Dropout.)")

    y_true, y_pred = predict_loader(model, va_ld, device, tta=tta)

    names = ["notdrowsy (0)", "drowsy (1)"]
    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(y_true, y_pred))
    print("\n=== Per-class metrics (validation split, same seed as training) ===")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    val_acc = float((y_true == y_pred).mean())
    print(f"\nVal accuracy: {val_acc:.4f}  (TTA={'on' if tta else 'off'})")

    out = {
        "val_accuracy": val_acc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "tta": tta,
    }
    with open(Path(__file__).resolve().parent / "eval_resnet_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote eval_resnet_report.json")

    if args.misclassified:
        lines = []
        for i in range(len(y_true)):
            idx = va_i[i]
            path, lab = full.samples[idx]
            if y_true[i] != y_pred[i]:
                lines.append(f"true={lab} pred={y_pred[i]}  {path}")
        p = Path(__file__).resolve().parent / "misclassified_val_samples.txt"
        p.write_text("\n".join(lines) if lines else "(none)\n", encoding="utf-8")
        print(f"Wrote {p} ({len(lines)} files)")


if __name__ == "__main__":
    main()
