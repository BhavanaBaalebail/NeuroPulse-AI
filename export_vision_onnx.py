#!/usr/bin/env python3
"""Export a vision checkpoint to ONNX and require numerical agreement with PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parent


def build_model(backbone: str, num_classes: int = 2, dropout: float = 0.3):
    backbone = backbone.lower()
    if backbone == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m
    if backbone in ("mobilenet_v3_small", "mobilenetv3s"):
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(backbone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metadata", default="", help="JSON written by train_vision_models.py")
    ap.add_argument("--out", default="models/vision/fatigue_model_candidate.onnx")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--parity-image", default="", help="RGB image crop to verify through deployment preprocessing")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    meta_path = Path(args.metadata) if args.metadata else ckpt.with_suffix(".json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    backbone = meta["backbone"]
    size = int(meta["input_size"])
    model = build_model(backbone)
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    parity_batch = dummy.numpy()
    if args.parity_image:
        image = Image.open(args.parity_image).convert("RGB")
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - np.asarray(meta["mean"], dtype=np.float32)) / np.asarray(
            meta["std"], dtype=np.float32
        )
        parity_batch = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes={"image": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    parity_tensor = torch.from_numpy(parity_batch)
    onnx_out = sess.run(None, {name: parity_batch})[0]
    ref = model(parity_tensor).detach().numpy()
    ok = np.allclose(onnx_out, ref, atol=args.atol)
    max_abs = float(np.max(np.abs(onnx_out - ref)))
    report = {
        "onnx_path": str(out),
        "pytorch_onnx_max_abs_diff": max_abs,
        "atol": args.atol,
        "agreement": ok,
        "class_names": meta.get("class_names"),
        "onnx_drowsy_class_index": meta.get("onnx_drowsy_class_index", 1),
        "input_size": size,
        "mean": meta.get("mean"),
        "std": meta.get("std"),
        "backbone": backbone,
        "parity_input": args.parity_image or "random_tensor",
    }
    report_path = out.with_suffix(".onnx_check.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(f"ONNX disagreement max_abs={max_abs} > atol={args.atol}")
    print("Numerical agreement OK. Existing fatigue_model.onnx is unchanged until you copy this file over.")


if __name__ == "__main__":
    main()
