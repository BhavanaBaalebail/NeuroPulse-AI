#!/usr/bin/env python3
"""
Train lightweight face drowsiness classifiers on a source-grouped split.

Default data: local RGB webcam stills (drowsy / notdrowsy). This is NOT an eye
open/closed dataset. An optional --eyes-root can point at MRL-derived RGB/IR
crops with class folders `open` and `closed`.

Does not download restricted datasets. See datasets/README.md.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from vision.dataset_split import grouped_split, list_class_images, write_manifest

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ROOT = Path(__file__).resolve().parent


class ImageRows(Dataset):
    def __init__(self, rows, transform, input_size):
        self.rows = rows
        self.transform = transform
        self.input_size = input_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        img = cv2.imread(rec["path"])
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), torch.tensor(rec["label"], dtype=torch.long)


def build_backbone(name: str, num_classes: int, dropout: float):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m
    if name in ("mobilenet_v3_small", "mobilenetv3s"):
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unknown backbone {name}")


def collect_rows(args) -> tuple[list[dict], list[str]]:
    if args.eyes_root:
        roots = [Path(p.strip()) for p in args.eyes_root.split(",") if p.strip()]
        rows = []
        for r in roots:
            closed = list_class_images(r / "closed", "closed", 1)
            open_eyes = list_class_images(r / "open", "open", 0)
            for row in closed + open_eyes:
                row["domain"] = r.name
            rows.extend(closed + open_eyes)
        if not rows:
            raise SystemExit(f"No open/closed images in {args.eyes_root}")
        return rows, ["open", "closed"]
    root = Path(args.data)
    rows = []
    rows.extend(list_class_images(root / "drowsy", "drowsy", 1))
    rows.extend(list_class_images(root / "notdrowsy", "notdrowsy", 0))
    if not rows:
        raise SystemExit(f"No drowsy/notdrowsy images in {root}")
    return rows, ["notdrowsy", "drowsy"]


def evaluate_by_domain(model, rows_split, transform, size, device, class_names, batch_size):
    results = {}
    domains = sorted({row.get("domain", "unspecified") for row in rows_split})
    for domain in domains:
        domain_rows = [row for row in rows_split if row.get("domain", "unspecified") == domain]
        loader = DataLoader(
            ImageRows(domain_rows, transform, size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        results[domain] = {"count": len(domain_rows), **evaluate(model, loader, device, class_names)}
    return results


def evaluate(model, loader, device, class_names: list[str]):
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy().tolist()
            ys.extend(y.numpy().tolist())
            yh.extend(pred)
    y_true = np.array(ys)
    y_pred = np.array(yh)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    target_names = [f"{class_names[0]} (0)", f"{class_names[1]} (1)"]
    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "confusion_matrix": cm.tolist(),
        "false_positive_rate_alert": float(fp / max(tn + fp, 1)),
        "per_class": report,
    }


def train_one(args, backbone: str, rows_split: dict, class_names: list[str], device):
    size = args.input_size
    train_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((int(size * 1.1), int(size * 1.1))),
            transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.15, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tr = ImageRows(rows_split["train"], train_tf, size)
    va = ImageRows(rows_split["val"], val_tf, size)
    te = ImageRows(rows_split["test"], val_tf, size)
    labels = [r["label"] for r in rows_split["train"]]
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / np.clip(counts, 1, None)
    sampler = WeightedRandomSampler(
        [weights[y] for y in labels], num_samples=len(labels), replacement=True
    )
    tr_ld = DataLoader(tr, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    va_ld = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_backbone(backbone, num_classes=2, dropout=0.3).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_metric = -1.0
    run_name = args.run_name or ("eyes" if args.eyes_root else "face")
    ckpt = ROOT / "models" / "vision" / f"best_{backbone}_{run_name}.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    hist = []

    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        val_stats = evaluate(model, va_ld, device, class_names)
        val_by_domain = evaluate_by_domain(
            model, rows_split["val"], val_tf, size, device, class_names, args.batch_size
        ) if args.eyes_root else {}
        rec = {"epoch": epoch + 1, "train_acc": correct / max(total, 1), "val": val_stats}
        if val_by_domain:
            rec["val_by_domain"] = val_by_domain
        hist.append(rec)
        print(f"{backbone} epoch {epoch+1}/{args.epochs} train_acc={rec['train_acc']:.3f} val_acc={val_stats['accuracy']:.3f}")
        selection = val_by_domain.get("webcam_eyes", val_stats)
        metric = selection["per_class"].get("macro avg", {}).get("f1-score", selection["accuracy"])
        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), ckpt)

    try:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    test_stats = evaluate(model, te_ld, device, class_names)
    test_by_domain = evaluate_by_domain(
        model, rows_split["test"], val_tf, size, device, class_names, args.batch_size
    ) if args.eyes_root else {}
    domain_counts = {
        split_name: {
            domain: sum(1 for row in rows_split[split_name] if row.get("domain", "unspecified") == domain)
            for domain in sorted({row.get("domain", "unspecified") for row in rows_split[split_name]})
        }
        for split_name in rows_split
    }
    meta = {
        "backbone": backbone,
        "input_size": size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "class_names": class_names,
        "onnx_drowsy_class_index": 1,
        "checkpoint": str(ckpt),
        "val_best_selection_metric": "webcam_eyes macro F1" if args.eyes_root else "overall accuracy",
        "val_best_metric": best_metric,
        "test": test_stats,
        "test_by_domain": test_by_domain,
        "domain_counts_by_split": domain_counts,
        "split_manifest": str(args.manifest_path) if args.manifest_path else None,
        "pretrained_weights": "ImageNet1K_V1",
        "training_seed": args.seed,
        "deployment_approved": False,
        "deployment_gate_note": "Disabled by default. The current RGB held-out set is small and its labels are EAR-threshold-derived; use only after a larger manually verified webcam evaluation.",
        "history": hist,
        "task": "eye_open_closed" if args.eyes_root else "face_drowsy_notdrowsy",
        "domain_note": "MRL IR and local RGB webcam crops are reported separately. RGB labels originate from EAR thresholds and are not manually annotated." if args.eyes_root else "Labels are folder-level drowsy/notdrowsy, not verified eye-closed labels.",
    }
    meta_path = ckpt.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {ckpt} and {meta_path}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train"))
    ap.add_argument("--eyes-root", default="", help="Optional open/closed crop root (e.g. data/derived/mrl_eyes,data/derived/webcam_eyes)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--input-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbones", default="resnet18,mobilenet_v3_small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default="", help="Output artifact suffix; use a new name to preserve prior runs")
    args = ap.parse_args()
    args.eyes_root = args.eyes_root.strip() or None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows, class_names = collect_rows(args)
    split = grouped_split(rows, val_frac=0.15, test_frac=0.15, seed=args.seed)
    args.manifest_path = ROOT / "data" / "splits" / (
        f"{args.run_name or 'eyes'}_manifest.json" if args.eyes_root else "webcam_manifest.json"
    )
    write_manifest(split, args.manifest_path, class_names=class_names)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    comparison = {}
    for name in [s.strip() for s in args.backbones.split(",") if s.strip()]:
        comparison[name] = train_one(args, name, split, class_names, device)

    dest = ROOT / "models" / "vision" / f"comparison_{args.run_name or ('eyes' if args.eyes_root else 'face')}.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Wrote {dest}")
    print("Export a chosen checkpoint with: python export_vision_onnx.py --checkpoint models/vision/best_mobilenet_v3_small.pth")


if __name__ == "__main__":
    main()
