"""
Transfer-learning fatigue classifier: ResNet-18 (ImageNet) → binary drowsy / notdrowsy.

Why this instead of LSTM?
- Your data are independent images (folders), not time-aligned video clips.
- LSTM helps when labels depend on *motion over time* (e.g. blink rate). It needs
  sequence data and usually *more* labels. Stacking random images into fake sequences
  rarely beats a strong image model.
- **Pretrained CNNs** (ResNet, EfficientNet) almost always gain several points on
  small datasets vs training a tiny CNN from scratch.

After training + export, copy/rename outputs and use vision_model_config.json with app.py
(see export_resnet_onnx.py).

Requires: torch, torchvision, scikit-learn, opencv-python
"""

import json
import os
import random
import ssl
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms

from vision.dataset_split import grouped_split, source_key

# ImageNet normalization (must match app.py when vision_model_config.json is present)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 160  # good balance: faster than 224, still strong for ResNet
HEAD_DROPOUT = 0.35  # classifier regularization (must match export_resnet_onnx.py)


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


def build_sampler(dataset: FatigueDataset, indices):
    labels = [dataset.samples[i][1] for i in indices]
    counts = np.bincount(labels, minlength=2)
    if counts[0] == counts[1]:
        return None
    w = 1.0 / np.clip(counts, 1, None)
    sw = [w[l] for l in labels]
    return WeightedRandomSampler(
        torch.DoubleTensor(sw), num_samples=len(sw), replacement=True
    )


def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot, cor, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss_sum += crit(logits, y).item()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)
    n = max(len(loader), 1)
    return loss_sum / n, cor / max(tot, 1)


@torch.no_grad()
def val_predictions(model, loader, device, tta: bool = True):
    """Returns y_true, y_pred numpy arrays for confusion matrix."""
    model.eval()
    ys, yh = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if tta:
            logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
        else:
            logits = model(x)
        ys.extend(y.cpu().numpy().tolist())
        yh.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.array(ys), np.array(yh)


def _load_resnet18_imagenet():
    """Load torchvision ResNet-18 with ImageNet weights (may download once)."""
    try:
        w = models.ResNet18_Weights.IMAGENET1K_V1
        return models.resnet18(weights=w)
    except (AttributeError, TypeError):
        return models.resnet18(pretrained=True)


def build_model(device):
    """
    ImageNet init. If download fails with SSL errors (common on macOS python.org builds):
    - Run ``Applications/Python 3.x/Install Certificates.command``, or
    - Download ``resnet18-f37072fd.pth`` from PyTorch and set
      ``RESNET18_LOCAL_WEIGHTS=/path/to/resnet18-f37072fd.pth``, or
    - Retry is attempted once with SSL verification disabled for this download only.
    """
    local = os.environ.get("RESNET18_LOCAL_WEIGHTS", "").strip()
    if local:
        p = Path(local).expanduser()
        if p.is_file():
            m = models.resnet18(weights=None)
            try:
                state = torch.load(p, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(p, map_location="cpu")
            m.load_state_dict(state)
            in_f = m.fc.in_features
            m.fc = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(in_f, 2))
            return m.to(device)
        print(f"⚠️  RESNET18_LOCAL_WEIGHTS not found ({p}); will try download.")

    try:
        m = _load_resnet18_imagenet()
    except (urllib.error.URLError, ssl.SSLError) as e:
        err = str(e).lower()
        if "certificate verify failed" not in err:
            raise
        _prev = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            print(
                "⚠️  SSL certificate verify failed while downloading weights; "
                "retrying once without verification. For a proper fix, run "
                "Install Certificates.command (from python.org) or set "
                "RESNET18_LOCAL_WEIGHTS to a local resnet18-*.pth file."
            )
            m = _load_resnet18_imagenet()
        finally:
            ssl._create_default_https_context = _prev

    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(in_f, 2))
    return m.to(device)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((int(INPUT_SIZE * 1.12), int(INPUT_SIZE * 1.12))),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.22, 0.22, 0.16, 0.05),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    full = FatigueDataset("data/train", transform=None)
    rows = []
    for path, label in full.samples:
        cls = "drowsy" if int(label) == 1 else "notdrowsy"
        rows.append({"path": path, "label": int(label), "source": source_key(Path(path), cls)})
    grouped = grouped_split(rows, val_frac=0.2, test_frac=0.0, seed=42)
    path_to_i = {full.samples[i][0]: i for i in range(len(full.samples))}
    tr_i = [path_to_i[r["path"]] for r in grouped["train"]]
    va_i = [path_to_i[r["path"]] for r in grouped["val"]]
    print(
        f"Grouped split (no aug leakage): train={len(tr_i)} val={len(va_i)} "
        f"sources train={len({r['source'] for r in grouped['train']})} "
        f"val={len({r['source'] for r in grouped['val']})}"
    )

    tr_ds = FatigueDataset("data/train", train_tf)
    va_ds = FatigueDataset("data/train", val_tf)
    tr_sub = Subset(tr_ds, tr_i)
    va_sub = Subset(va_ds, va_i)

    batch_size = 16 if device.type == "cpu" else 32
    sampler = build_sampler(tr_ds, tr_i)
    if sampler is None:
        tr_ld = DataLoader(tr_sub, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        tr_ld = DataLoader(tr_sub, batch_size=batch_size, sampler=sampler, num_workers=0)
    va_ld = DataLoader(va_sub, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(device)
    tr_labels = [full.samples[i][1] for i in tr_i]
    counts = np.bincount(tr_labels, minlength=2).astype(np.float64)
    n_tr = len(tr_i)
    class_w = n_tr / (2.0 * np.maximum(counts, 1.0))
    cw = torch.tensor(class_w, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    print(
        f"Class weights (train): notdrowsy={class_w[0]:.3f} drowsy={class_w[1]:.3f}  "
        f"counts={counts.astype(int).tolist()}"
    )

    # Two LR groups: lower for backbone, higher for head
    params_fc = list(model.fc.parameters())
    params_rest = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    opt = optim.AdamW(
        [
            {"params": params_rest, "lr": 3e-5},
            {"params": params_fc, "lr": 2e-3},
        ],
        weight_decay=1e-4,
    )
    epochs = 35
    steps = max(len(tr_ld), 1)
    sched = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[1e-4, 5e-3],
        epochs=epochs,
        steps_per_epoch=steps,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=1e3,
    )

    best_val = 0.0
    best_loss = float("inf")
    stale = 0
    patience = 10
    hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for ep in range(epochs):
        model.train()
        loss_sum, cor, tot = 0.0, 0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item()
            cor += (logits.argmax(1) == y).sum().item()
            tot += y.size(0)

        tr_loss = loss_sum / max(len(tr_ld), 1)
        tr_acc = cor / max(tot, 1)
        va_loss, va_acc = evaluate(model, va_ld, device)
        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        print(
            f"Epoch {ep+1:02d}/{epochs}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}  val_loss={va_loss:.4f}"
        )

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), "best_model_resnet.pth")
            print(f"  ★ saved best_model_resnet.pth (val_acc={best_val:.4f})")

        if va_loss < best_loss - 1e-4:
            best_loss = va_loss
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop at epoch {ep+1}")
                break

    root = Path(__file__).resolve().parent
    best_path = root / "best_model_resnet.pth"
    if best_path.is_file():
        try:
            sd_best = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            sd_best = torch.load(best_path, map_location=device)
        model.load_state_dict(sd_best)
        y_true, y_pred = val_predictions(model, va_ld, device, tta=True)
        names = ["notdrowsy (0)", "drowsy (1)"]
        print("\n=== Validation confusion matrix (best checkpoint, TTA) ===")
        print(confusion_matrix(y_true, y_pred))
        print("\n=== Per-class (validation) ===")
        print(classification_report(y_true, y_pred, target_names=names, digits=4))
        eval_summary = {
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "val_accuracy": float((y_true == y_pred).mean()),
        }
        with open(root / "eval_resnet_train_end.json", "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)
        print("Wrote eval_resnet_train_end.json")
    else:
        print("\n(No best_model_resnet.pth — skipped confusion matrix.)")

    cfg = {
        "input_size": INPUT_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "backbone": "resnet18",
        "class_names": ["notdrowsy", "drowsy"],
        "onnx_drowsy_class_index": 1,
        "head_dropout": HEAD_DROPOUT,
    }
    with open(root / "vision_model_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote vision_model_config.json (input {INPUT_SIZE}, ImageNet norm)")

    with open(root / "training_history_resnet.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    print(f"\nBest val_acc={best_val:.4f}")
    print("Diagnose model quality: python eval_resnet.py")
    print("Next: python export_resnet_onnx.py")
    print("Then: copy fatigue_model.onnx from export output OR set app to load new onnx path.")


if __name__ == "__main__":
    main()
