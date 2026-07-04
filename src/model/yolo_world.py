from __future__ import annotations

from ultralytics import YOLOWorld


def build_yolo_world_model(runtime_cfg: dict) -> YOLOWorld:
    weight_ref = runtime_cfg["paths"]["weights"]
    if not weight_ref:
        raise ValueError("model weight is missing for YOLO-World")

    model = YOLOWorld(weight_ref)
    prompts = runtime_cfg.get("prompts") or list(runtime_cfg.get("class_names", {}).values())
    if prompts:
        model.set_classes(prompts)
    return model