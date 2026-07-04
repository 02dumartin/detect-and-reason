from __future__ import annotations

from src.model.model_builder import build_model
from src.model.rf_detr_runner import run_rf_detr_train


def run_train(runtime_cfg: dict):
    if runtime_cfg.get("family") == "rf_detr":
        return run_rf_detr_train(runtime_cfg)
    if runtime_cfg.get("family") == "dino":
        from src.model.dino_runner import run_dino_train

        return run_dino_train(runtime_cfg)

    model = build_model(runtime_cfg)

    train_cfg = runtime_cfg.get("train", {})
    val_cfg = runtime_cfg.get("val", {})
    dataset_cfg = runtime_cfg["dataset"]
    runs_dir = runtime_cfg["paths"]["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "data": str(dataset_cfg["data_yaml"]),
        "epochs": train_cfg.get("epochs"),
        "imgsz": train_cfg.get("imgsz"),
        "batch": train_cfg.get("batch"),
        "workers": train_cfg.get("workers"),
        "patience": train_cfg.get("patience"),
        "amp": train_cfg.get("amp"),
        "cache": dataset_cfg.get("cache"),
        "optimizer": train_cfg.get("optimizer"),
        "lr0": train_cfg.get("lr0"),
        "lrf": train_cfg.get("lrf"),
        "cos_lr": train_cfg.get("cos_lr"),
        # Ultralytics train-time validation defaults to val split + library defaults
        # unless we forward the validation block explicitly.
        "split": val_cfg.get("resolved_split"),
        "plots": val_cfg.get("plots"),
        "conf": val_cfg.get("conf"),
        "iou": val_cfg.get("resolved_iou"),
        "agnostic_nms": val_cfg.get("agnostic_nms"),
        "max_det": val_cfg.get("max_det"),
        "project": str(runs_dir.parent),
        "name": runs_dir.name,
        "exist_ok": True,
    }
    kwargs.update(train_cfg.get("augmentation", {}))

    return model.train(**_drop_none(kwargs))


def _drop_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}
