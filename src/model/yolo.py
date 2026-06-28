from __future__ import annotations

from ultralytics import YOLO


def build_yolo_model(runtime_cfg: dict) -> YOLO:
    weight_ref = runtime_cfg["paths"]["weights"]
    if not weight_ref:
        raise ValueError("model weight is missing for YOLO")
    return YOLO(weight_ref)
