"""DINO (DETR 계열) 모델 빌더 + COCO 추론 헬퍼.

working_title 기반 self-contained DINO 구현(`src/dino_vendor/`)을 감싼다.
detectron2 비의존, MSDeformAttn 은 CUDA 확장 미빌드 시 순수 PyTorch fallback.

역할 분담 (rf_detr.py 와 동일한 패턴):
    dino.py        ← 모델 빌더 + 추론/COCO 변환 헬퍼 (model_builder 가 소비)
    dino_runner.py ← train/validate/predict stage 러너 (train/validate/predict.py 가 소비)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# vendoring 된 working_title 검출 스택을 import 가능하게 한다.
#   내부 import 가 `from models.X` / `from data.X` / `from engine.X` (top-level)
#   형태라, vendor 루트를 sys.path 앞에 넣어 그대로 해석되게 한다.
# ---------------------------------------------------------------------------
_VENDOR_ROOT = Path(__file__).resolve().parents[1] / "dino_vendor"


def ensure_vendor_on_path() -> None:
    path = str(_VENDOR_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


ensure_vendor_on_path()

import torch  # noqa: E402

# RF-DETR 와 동일한 split 명명 규칙 (Roboflow COCO 레이아웃: train/valid/test)
_SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


class ModelBuildError(RuntimeError):
    """DINO 모델 생성 실패."""


def normalize_dino_split(split: str) -> str:
    """'val' → 'valid' 등 디스크 레이아웃에 맞는 split 이름으로 정규화한다."""
    return _SPLIT_ALIASES.get(str(split).strip().lower(), str(split).strip().lower())


def _device(train_cfg: dict) -> "torch.device":
    requested = str(train_cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# working_title cfg dict 조립
# ---------------------------------------------------------------------------
def build_dino_data_cfg(runtime_cfg: dict) -> dict:
    """rfdetr_dir(COCO) 에서 split 별 annotation/image 경로를 유도하고
    모델 config 의 augmentation 블록(`data:`)과 합쳐 working_title data cfg 를 만든다."""
    rfdetr_dir = Path(runtime_cfg["dataset"]["rfdetr_dir"])
    raw_model = runtime_cfg.get("raw", {}).get("model", {})
    data_cfg = dict(raw_model.get("data", {}))  # train/val augmentation 등

    # 우리 디스크 레이아웃: <rfdetr_dir>/{train,valid,test}/_annotations.coco.json
    #   file_name 이 "images/xxx.jpg" 라 image_root = split 디렉터리.
    split_dirs = {"train": "train", "val": "valid", "test": "test"}
    for key, sub in split_dirs.items():
        sdir = rfdetr_dir / sub
        data_cfg[f"{key}_annotation_file"] = str(sdir / "_annotations.coco.json")
        data_cfg[f"{key}_image_root"] = str(sdir)

    data_cfg.setdefault("image_format", "RGB")
    data_cfg.setdefault("num_workers", runtime_cfg.get("train", {}).get("num_workers", 4))
    return data_cfg


def build_dino_outputs_cfg(runtime_cfg: dict) -> dict:
    """Trainer 가 요구하는 outputs 경로를 runs_dir 기준으로 만든다.
    체크포인트는 runs_dir/weights/{best,last}.pt 에 저장 → YOLO 와 동일 규칙
    (config_loader 의 기본 weight 탐색 및 benchmark.sh 와 호환)."""
    runs_dir = Path(runtime_cfg["paths"]["runs_dir"])
    return {
        "root": str(runs_dir),
        "checkpoints": str(runs_dir / "weights"),
        "predictions": str(runs_dir / "predictions"),
        "metrics": str(runs_dir / "metrics"),
        "reports": str(runs_dir / "reports"),
        "figures": str(runs_dir / "figures"),
        "configs": str(runs_dir / "configs"),
    }


def build_vendor_cfg(runtime_cfg: dict, *, with_outputs: bool = True) -> dict:
    """runtime_cfg → working_title build_model/Trainer 가 먹는 cfg dict."""
    model_cfg = dict(runtime_cfg.get("model", {}))
    # num_classes 는 config 의 model.num_classes 를 그대로 신뢰 (class_mode 와 일치하게 작성됨)
    cfg: dict[str, Any] = {
        "model": model_cfg,
        "train": dict(runtime_cfg.get("train", {})),
        "data": build_dino_data_cfg(runtime_cfg),
        "visualization": {"period": 0},
    }
    if with_outputs:
        cfg["outputs"] = build_dino_outputs_cfg(runtime_cfg)
    return cfg


# ---------------------------------------------------------------------------
# 모델 빌드 / 가중치 로드
# ---------------------------------------------------------------------------
def _vendor_build_model(cfg: dict):
    ensure_vendor_on_path()
    try:
        from models.build import build_model as vendor_build_model
    except Exception as exc:  # pragma: no cover
        raise ModelBuildError(f"DINO vendor import 실패: {exc}") from exc
    return vendor_build_model(cfg)


def load_dino_weights(model, weights_path: str | Path) -> None:
    """체크포인트(state dict) 를 로드한다. EMA 가중치가 있으면 우선 사용."""
    ckpt = torch.load(str(weights_path), map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        if ckpt.get("ema"):
            ema = ckpt["ema"]
            state = ema.get("module", ema) if isinstance(ema, dict) else None
        if state is None:
            state = ckpt.get("model", ckpt)
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


# num_classes 에 의존하는 파라미터(데이터셋마다 크기 다름) → COCO pretrain 로드 시 제외.
_NUM_CLASS_DEPENDENT = ("class_embed", "label_enc")


def load_dino_pretrain(model, path: str | Path) -> dict:
    """COCO 로 사전학습된 detrex DINO-R50 가중치를 부분 로드한다.

    detrex 내부 DINO 는 우리 vendored DINO 와 모듈 구조가 동일해 backbone/neck/
    transformer/bbox_embed 등은 그대로 로드된다. 다만 class_embed·label_enc 는
    num_classes(=COCO 80/91) 크기라 우리(3/1/2)와 안 맞으므로 건너뛴다(=랜덤 초기화 유지).
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    model_state = model.state_dict()
    to_load: dict = {}
    skipped_cls = 0
    skipped_shape = 0
    for key, value in state.items():
        if any(token in key for token in _NUM_CLASS_DEPENDENT):
            skipped_cls += 1
            continue
        target = model_state.get(key)
        if target is None:
            continue  # 우리 모델에 없는 키(unexpected)
        if tuple(target.shape) != tuple(value.shape):
            # num_queries 등 하이퍼파라미터 차이로 크기 다른 텐서 → 스킵(랜덤 유지)
            skipped_shape += 1
            continue
        to_load[key] = value

    model.load_state_dict(to_load, strict=False)
    matched = len(to_load)
    total_model = len(model_state)
    # logging 미설정 환경(train_model.py)에서도 보이도록 print 로 확실히 남긴다.
    print(
        f"[DINO] COCO pretrain 로드: {matched}/{total_model} 파라미터 매칭 "
        f"(class/label 제외 {skipped_cls}, shape 불일치 스킵 {skipped_shape}) <- {Path(path).name}"
    )
    if matched < total_model * 0.5:
        logger.warning(
            "[DINO] pretrain 매칭률이 낮습니다(%d/%d). 체크포인트 아키텍처가 다를 수 있음.",
            matched,
            total_model,
        )
    return {"matched": matched, "total": total_model, "skipped_shape": skipped_shape}


def build_dino_model(runtime_cfg: dict, *, load_weights: bool = True):
    """DINO 모델을 만들어 device 로 올린다. stage 가 validate/predict 이거나
    weights 가 resolved 되어 있으면 가중치를 로드한다."""
    cfg = build_vendor_cfg(runtime_cfg, with_outputs=False)
    model = _vendor_build_model(cfg)

    device = _device(runtime_cfg.get("train", {}))
    model.to(device)

    weights = runtime_cfg.get("paths", {}).get("weights")
    if load_weights and weights and Path(str(weights)).is_file():
        load_dino_weights(model, weights)
    return model


def set_eval_mode(model) -> None:
    model.eval()


# ---------------------------------------------------------------------------
# 추론 → COCO 예측 JSON (YOLO/RT-DETR/RF-DETR 과 동일 metric 으로 비교 가능)
# ---------------------------------------------------------------------------
def _build_eval_loader(vendor_cfg: dict, split_key: str):
    """split_key ∈ {train,val,test} 에 대한 평가용 DataLoader 를 만든다.
    build_split_dataloader 는 val/test 만 허용하므로 직접 구성해 train 도 지원."""
    ensure_vendor_on_path()
    from torch.utils.data import DataLoader
    from data.detection_dataset import DetectionDataset, collate_detection
    from data.transforms import build_val_transforms

    data_cfg = vendor_cfg["data"]
    train_cfg = vendor_cfg.get("train", {})
    img_format = data_cfg.get("image_format", "RGB")

    dataset = DetectionDataset(
        annotation_file=data_cfg[f"{split_key}_annotation_file"],
        image_root=data_cfg[f"{split_key}_image_root"],
        transforms=build_val_transforms(data_cfg),
        img_format=img_format,
        is_train=False,
    )
    batch_size = int(
        data_cfg.get("val_batch_size", train_cfg.get("batch_size", train_cfg.get("total_batch_size", 1)))
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        collate_fn=collate_detection,
    )


def run_split_inference_to_coco(model, runtime_cfg: dict, split: str, out_json: str | Path, score_threshold: float = 0.0) -> list[dict]:
    """split 에 대해 추론하고 COCO detection 포맷 예측 리스트를 out_json 에 저장한다.
    반환 포맷: [{image_id, category_id, bbox:[x,y,w,h], score}] (절대 픽셀).
    score_threshold 이상인 박스만 저장한다(예측 flooding 방지)."""
    ensure_vendor_on_path()
    from engine import export_predictions_to_coco  # noqa: E402

    # cfg 의 data split key 는 train/val/test. 입력 split('valid' 등) → key 로 역매핑.
    norm = normalize_dino_split(split)
    split_key = {"train": "train", "valid": "val", "test": "test"}.get(norm, norm)

    vendor_cfg = build_vendor_cfg(runtime_cfg, with_outputs=False)
    loader = _build_eval_loader(vendor_cfg, split_key)

    device = _device(runtime_cfg.get("train", {}))
    model.eval()
    preds = export_predictions_to_coco(model, loader, out_json, device=device, score_threshold=score_threshold)
    return preds


def resolve_dino_split_ann(runtime_cfg: dict, split: str) -> Path:
    """split 의 GT annotation 경로 (<rfdetr_dir>/<split>/_annotations.coco.json)."""
    rfdetr_dir = Path(runtime_cfg["dataset"]["rfdetr_dir"])
    return rfdetr_dir / normalize_dino_split(split) / "_annotations.coco.json"
