"""Modular vision pipeline: landmarks, eye/mouth state, fusion, ONNX."""

from vision.config import VisionConfig, load_vision_config

__all__ = ["FatigueDetector", "VisionConfig", "load_vision_config"]


def __getattr__(name):
    if name == "FatigueDetector":
        from vision.detector import FatigueDetector

        return FatigueDetector
    raise AttributeError(name)
