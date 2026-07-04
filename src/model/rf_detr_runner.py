from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.evaluation.coco_metrics import coco_map_from_json
from src.model.rf_detr import (
    DatasetPreprocessor,
    build_rf_detr_model,
    convert_predictions_to_coco,
    normalize_rfdetr_split,
    resolve_rfdetr_resolution,
    resolve_rfdetr_split_dir,
    run_split_inference,
    set_eval_mode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def run_rf_detr_train(runtime_cfg: dict) -> Any:
    train_cfg = runtime_cfg.get("train", {})
    dataset_cfg = runtime_cfg["dataset"]
    runs_dir = runtime_cfg["paths"]["runs_dir"]
    rfdetr_dir = Path(dataset_cfg["rfdetr_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = DatasetPreprocessor(rfdetr_dir)
    preprocessor.prepare()

    resolution = resolve_rfdetr_resolution(train_cfg)
    model = build_rf_detr_model(runtime_cfg)
    kwargs = {
        "dataset_dir": str(rfdetr_dir),
        "output_dir": str(runs_dir),
        "resolution": resolution,
        "epochs": train_cfg.get("epochs"),
        "batch_size": train_cfg.get("batch_size"),
        "grad_accum_steps": train_cfg.get("grad_accum_steps"),
        "lr": train_cfg.get("lr"),
        "lr_encoder": train_cfg.get("lr_encoder"),
        "weight_decay": train_cfg.get("weight_decay"),
        "use_ema": train_cfg.get("use_ema"),
        "ema_decay": train_cfg.get("ema_decay"),
        "early_stopping": train_cfg.get("early_stopping"),
        "early_stopping_patience": train_cfg.get("early_stopping_patience"),
        "early_stopping_min_delta": train_cfg.get("early_stopping_min_delta"),
        "early_stopping_use_ema": train_cfg.get("use_ema"),
        "num_workers": train_cfg.get("num_workers"),
        "tensorboard": train_cfg.get("tensorboard"),
        "dataset_file": train_cfg.get("dataset_file", "roboflow"),
    }
    if train_cfg.get("resume"):
        kwargs["resume"] = train_cfg["resume"]

    logger.info(
        "[RF-DETR] train dataset_dir=%s output_dir=%s resolution=%s epochs=%s",
        rfdetr_dir,
        runs_dir,
        resolution,
        train_cfg.get("epochs"),
    )
    return model.train(**_drop_none(kwargs))


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def run_rf_detr_validate(runtime_cfg: dict) -> dict:
    val_cfg = runtime_cfg.get("val", {})
    train_cfg = runtime_cfg.get("train", {})
    dataset_cfg = runtime_cfg["dataset"]
    eval_dir = runtime_cfg["paths"]["eval_dir"]
    rfdetr_dir = Path(dataset_cfg["rfdetr_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    split = normalize_rfdetr_split(val_cfg.get("resolved_split", val_cfg.get("split", "test")))
    split_dir = resolve_rfdetr_split_dir(rfdetr_dir, split)
    batch_size = int(train_cfg.get("batch_size") or 4)
    conf_threshold = float(val_cfg.get("conf_threshold") or 0.01)

    model = build_rf_detr_model(runtime_cfg)
    set_eval_mode(model)
    predictions, image_ids, ann_path = run_split_inference(
        model,
        split_dir,
        batch_size=batch_size,
        conf_threshold=0.0,
    )

    coco_preds = convert_predictions_to_coco(predictions, image_ids, conf_threshold)
    pred_path = eval_dir / f"{split}_predictions_coco.json"
    pred_path.write_text(json.dumps(coco_preds, indent=2), encoding="utf-8")

    metrics = {"split": split, "num_predictions": len(coco_preds)}
    if coco_preds:
        map_all, map50 = coco_map_from_json(ann_path, pred_path)
        metrics["mAP50_95"] = map_all
        metrics["mAP50"] = map50
        logger.info("[RF-DETR] validate split=%s mAP50-95=%.4f mAP50=%.4f", split, map_all, map50)
    else:
        metrics["mAP50_95"] = None
        metrics["mAP50"] = None
        logger.warning("[RF-DETR] validate split=%s produced zero detections", split)

    metrics_path = eval_dir / f"{split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def _resolve_predict_splits(runtime_cfg: dict) -> list[str]:
    predict_cfg = runtime_cfg.get("predict", {})
    splits = predict_cfg.get("splits")
    if splits:
        return [normalize_rfdetr_split(str(split_name)) for split_name in splits]

    source = runtime_cfg.get("predict", {}).get("resolved_source_name")
    if source in {"train", "val", "test"}:
        return [normalize_rfdetr_split(source)]
    return ["test"]


def run_rf_detr_predict(runtime_cfg: dict) -> dict[str, Path]:
    predict_cfg = runtime_cfg.get("predict", {})
    train_cfg = runtime_cfg.get("train", {})
    dataset_cfg = runtime_cfg["dataset"]
    prediction_dir = runtime_cfg["paths"]["prediction_dir"]
    rfdetr_dir = Path(dataset_cfg["rfdetr_dir"])

    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(train_cfg.get("batch_size") or 4)
    conf_threshold = float(predict_cfg.get("conf") or 0.25)
    splits = _resolve_predict_splits(runtime_cfg)

    model = build_rf_detr_model(runtime_cfg)
    set_eval_mode(model)

    saved: dict[str, Path] = {}
    for split in splits:
        split_dir = resolve_rfdetr_split_dir(rfdetr_dir, split)
        predictions, image_ids, _ = run_split_inference(
            model,
            split_dir,
            batch_size=batch_size,
            conf_threshold=0.0,
        )
        coco_preds = convert_predictions_to_coco(predictions, image_ids, conf_threshold)
        split_out = prediction_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        out_json = split_out / "predictions_coco.json"
        out_json.write_text(json.dumps(coco_preds, indent=2), encoding="utf-8")
        saved[split] = out_json
        logger.info(
            "[RF-DETR] predict split=%s detections=%d -> %s",
            split,
            len(coco_preds),
            out_json,
        )
    return saved


def _drop_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}
