#!/usr/bin/env python3
"""Export best_model_resnet.pth → fatigue_model.onnx (ResNet18, opset 18)."""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models

# Must match train_resnet.py (Dropout disabled at eval → ONNX is fine)
HEAD_DROPOUT = 0.35


def build_model():
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(in_f, 2))
    return m


def main():
    device = torch.device("cpu")
    model = build_model()
    sd = torch.load("best_model_resnet.pth", map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    cfg_path = Path(__file__).resolve().parent / "vision_model_config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            cfg = json.load(f)
        size = int(cfg.get("input_size", 160))
    else:
        size = 160

    dummy = torch.randn(1, 3, size, size)
    onnx_path = "fatigue_model.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["output"],
        opset_version=18,
        dynamic_axes={"image": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported {onnx_path}  input (1,3,{size},{size})")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: dummy.numpy()})[0]
    ref = model(dummy).detach().numpy()
    max_abs = float(np.max(np.abs(out - ref)))
    if not np.allclose(out, ref, atol=1e-4):
        raise SystemExit(f"ONNX vs PyTorch disagreement max_abs={max_abs}")
    print(f"ONNX vs PyTorch: OK (max_abs={max_abs:.2e})")
    print("Keep vision_model_config.json next to fatigue_model.onnx for app.py preprocessing.")


if __name__ == "__main__":
    main()
