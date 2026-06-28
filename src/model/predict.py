from __future__ import annotations

import shutil

from src.model.model_builder import build_model


def run_predict(runtime_cfg: dict):
    model = build_model(runtime_cfg)

    predict_cfg = runtime_cfg.get("predict", {})
    prediction_dir = runtime_cfg["paths"]["prediction_dir"]
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "source": str(predict_cfg["resolved_source"]),
        "imgsz": predict_cfg.get("imgsz") or runtime_cfg.get("train", {}).get("imgsz"),
        "save": predict_cfg.get("save", True),
        "save_txt": predict_cfg.get("save_txt", True),
        "save_conf": predict_cfg.get("save_conf", True),
        "conf": predict_cfg.get("conf"),
        "iou": predict_cfg.get("resolved_iou"),
        "agnostic_nms": predict_cfg.get("agnostic_nms"),
        "max_det": predict_cfg.get("max_det"),
        "project": str(prediction_dir.parent),
        "name": prediction_dir.name,
        "exist_ok": True,
    }

    return model.predict(**_drop_none(kwargs))


def _drop_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}
