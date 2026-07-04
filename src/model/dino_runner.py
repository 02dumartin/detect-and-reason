"""DINO stage 러너 (train / validate / predict).

train/validate/predict.py 디스패처가 family=="dino" 일 때 호출한다.
rf_detr_runner.py 와 동일한 패턴:
    - predict 는 split 별 predictions_coco.json 을 뱉어 기존 evaluate_coco_predictions 와
      동일 metric 으로 비교 가능하게 한다.
    - train 은 working_title engine.Trainer 를 구동하고 best/last.pt 를 runs_dir/weights/ 에 저장.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from src.evaluation.coco_metrics import coco_map_from_json
from src.model.dino import (
    build_dino_model,
    build_vendor_cfg,
    ensure_vendor_on_path,
    normalize_dino_split,
    resolve_dino_split_ann,
    run_split_inference_to_coco,
    set_eval_mode,
)

logger = logging.getLogger(__name__)


def _drop_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def run_dino_train(runtime_cfg: dict) -> Any:
    ensure_vendor_on_path()
    from data.build import build_dataloaders  # noqa: E402
    from engine import Trainer, build_optimizer, build_scheduler  # noqa: E402

    from src.model.dino import _device  # noqa: E402

    cfg = build_vendor_cfg(runtime_cfg, with_outputs=True)
    for out_path in cfg["outputs"].values():
        Path(out_path).mkdir(parents=True, exist_ok=True)

    # 모델 (ImageNet pretrained 백본 포함). trained 체크포인트는 로드 안 함(fresh).
    model = build_dino_model(runtime_cfg, load_weights=False)
    device = _device(runtime_cfg.get("train", {}))

    # COCO 사전학습 DINO 가중치(init_weight)를 부분 로드 → 검출 transformer/neck/bbox 를
    # COCO 지식으로 초기화(랜덤 대비 큰 개선). resume 시엔 건너뛴다.
    init_weight = runtime_cfg.get("model", {}).get("init_weight")
    if init_weight and not runtime_cfg.get("train", {}).get("resume"):
        from src.model.dino import load_dino_pretrain

        ppath = Path(init_weight)
        if not ppath.is_absolute():
            ppath = Path(runtime_cfg["project_root"]) / init_weight
        if ppath.is_file():
            load_dino_pretrain(model, ppath)
        else:
            logger.warning("[DINO] init_weight 없음: %s (COCO pretrain 건너뜀, 랜덤 헤드로 학습)", ppath)

    train_loader, val_loader = build_dataloaders(cfg)
    accum_steps = int(cfg.get("train", {}).get("gradient_accumulation_steps", 1))
    steps_per_epoch = max(1, (len(train_loader) + accum_steps - 1) // accum_steps)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=steps_per_epoch)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device,
    )
    resume = runtime_cfg.get("train", {}).get("resume")
    logger.info(
        "[DINO] train runs_dir=%s epochs=%s steps/epoch=%s",
        runtime_cfg["paths"]["runs_dir"],
        cfg["train"].get("max_epochs"),
        steps_per_epoch,
    )
    trainer.fit(resume=resume)
    return {"runs_dir": str(runtime_cfg["paths"]["runs_dir"])}


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def run_dino_validate(runtime_cfg: dict) -> dict:
    val_cfg = runtime_cfg.get("val", {})
    eval_dir = runtime_cfg["paths"]["eval_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)

    split = normalize_dino_split(val_cfg.get("resolved_split", val_cfg.get("split", "test")))

    model = build_dino_model(runtime_cfg)
    set_eval_mode(model)

    pred_path = eval_dir / f"{split}_predictions_coco.json"
    preds = run_split_inference_to_coco(model, runtime_cfg, split, pred_path)
    ann_path = resolve_dino_split_ann(runtime_cfg, split)

    metrics = {"split": split, "num_predictions": len(preds)}
    if preds:
        map_all, map50 = coco_map_from_json(ann_path, pred_path)
        metrics["mAP50_95"] = map_all
        metrics["mAP50"] = map50
        logger.info("[DINO] validate split=%s mAP50-95=%.4f mAP50=%.4f", split, map_all, map50)
    else:
        metrics["mAP50_95"] = None
        metrics["mAP50"] = None
        logger.warning("[DINO] validate split=%s produced zero detections", split)

    (eval_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def _resolve_predict_splits(runtime_cfg: dict) -> list[str]:
    predict_cfg = runtime_cfg.get("predict", {})
    splits = predict_cfg.get("splits")
    if splits:
        return [normalize_dino_split(str(s)) for s in splits]

    source = predict_cfg.get("resolved_source_name")
    if source in {"train", "val", "test"}:
        return [normalize_dino_split(source)]
    return ["test"]


def run_dino_predict(runtime_cfg: dict) -> dict[str, Path]:
    prediction_dir = runtime_cfg["paths"]["prediction_dir"]
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    splits = _resolve_predict_splits(runtime_cfg)

    # rf_detr 와 동일: predict conf(기본 0.25)로 export 단계에서 필터해 예측
    # flooding(=300 queries/img)을 막고 다른 모델과 공정 비교되게 한다.
    conf_threshold = float(runtime_cfg.get("predict", {}).get("conf") or 0.25)

    model = build_dino_model(runtime_cfg)
    set_eval_mode(model)

    saved: dict[str, Path] = {}
    for split in splits:
        split_out = prediction_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        out_json = split_out / "predictions_coco.json"
        preds = run_split_inference_to_coco(model, runtime_cfg, split, out_json, score_threshold=conf_threshold)
        saved[split] = out_json
        logger.info("[DINO] predict split=%s detections=%d -> %s", split, len(preds), out_json)
    return saved
