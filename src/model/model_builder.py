from __future__ import annotations

from src.model.rf_detr import build_rf_detr_model
from src.model.rt_detr import build_rt_detr_model
from src.model.yolo import build_yolo_model
from src.model.yolo_world import build_yolo_world_model


def build_model(runtime_cfg: dict):
    family = runtime_cfg.get("family")

    if family in {"yolo11", "yolo12"}:
        return build_yolo_model(runtime_cfg)
    if family == "yolo_world":
        return build_yolo_world_model(runtime_cfg)
    if family == "rt_detr":
        return build_rt_detr_model(runtime_cfg)
    if family == "rf_detr":
        return build_rf_detr_model(runtime_cfg)

    raise ValueError(f"Unsupported model family: {family}")