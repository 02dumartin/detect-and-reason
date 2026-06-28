from __future__ import annotations

from ultralytics import RTDETR


def build_rt_detr_model(runtime_cfg: dict) -> RTDETR:
    weight_ref = runtime_cfg["paths"]["weights"]
    if not weight_ref:
        raise ValueError("model weight is missing for RT-DETR")
    return RTDETR(weight_ref)

