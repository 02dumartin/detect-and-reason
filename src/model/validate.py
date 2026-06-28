from __future__ import annotations

from src.model.model_builder import build_model


def run_validate(runtime_cfg: dict):
    model = build_model(runtime_cfg)

    val_cfg = runtime_cfg.get("val", {})
    dataset_cfg = runtime_cfg["dataset"]
    eval_dir = runtime_cfg["paths"]["eval_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "data": str(dataset_cfg["data_yaml"]),
        "split": val_cfg.get("resolved_split"),
        "plots": val_cfg.get("plots"),
        "conf": val_cfg.get("conf"),
        "iou": val_cfg.get("resolved_iou"),
        "agnostic_nms": val_cfg.get("agnostic_nms"),
        "max_det": val_cfg.get("max_det"),
        "project": str(eval_dir.parent),
        "name": eval_dir.name,
        "exist_ok": True,
    }

    return model.val(**_drop_none(kwargs))


def _drop_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}

