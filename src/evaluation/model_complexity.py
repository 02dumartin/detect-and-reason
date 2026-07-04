from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def estimate_ultralytics_model_complexity(
    weights: str | Path | None,
    *,
    imgsz: int = 640,
) -> dict[str, Any]:
    """Ultralytics `.pt` 가중치 기준으로 모델 복잡도를 추정한다.

    이 평가는 라벨 txt만으로도 가능하지만, `tomato-detection-agentic` 쪽과
    비슷한 summary CSV를 만들려면 파라미터 수와 GFLOPs 같은 보조 정보도
    함께 남기는 편이 좋다.

    다만 이 값들은 "예측 txt의 품질" 자체와는 직접 관계가 없으므로,
    로딩이나 FLOPs 계산이 실패해도 전체 평가를 멈추지 않도록 설계한다.
    """
    if weights is None:
        return {}

    raw_value = str(weights)
    weight_path = Path(raw_value)
    if not weight_path.exists():
        return {
            "weight_reference": raw_value,
            "params_m": None,
            "gflops": None,
            "note": "local weight file not found; complexity skipped",
        }

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        return {
            "weight_reference": str(weight_path.resolve()),
            "params_m": None,
            "gflops": None,
            "note": "ultralytics is not installed; complexity skipped",
        }

    try:
        model = YOLO(str(weight_path.resolve()))
        torch_model = model.model
        total_params = sum(parameter.numel() for parameter in torch_model.parameters())
    except Exception:
        # ultralytics 로 못 읽는 체크포인트(rf_detr / dino 등) → state_dict 로 파라미터만 집계.
        # GFLOPs 는 forward 가 필요해 생략한다.
        return _complexity_from_state_dict(
            weight_path,
            note="non-ultralytics checkpoint; params from state_dict, GFLOPs skipped",
        )

    trainable_params = sum(parameter.numel() for parameter in torch_model.parameters() if parameter.requires_grad)
    model_size_mb = sum(parameter.numel() * 4 for parameter in torch_model.parameters()) / (1024**2)

    gflops: float | None = None
    gflops_note: str | None = None
    try:
        from thop import profile

        dummy_input = torch.randn(1, 3, int(imgsz), int(imgsz))
        torch_model_cpu = torch_model.cpu().eval()
        flops, _ = profile(torch_model_cpu, inputs=(dummy_input,), verbose=False)
        gflops = float(flops / 1e9)
    except ModuleNotFoundError:
        gflops_note = "thop is not installed; GFLOPs skipped"
    except Exception as exc:
        gflops_note = f"GFLOPs calculation failed: {exc}"

    result = {
        "weight_reference": str(weight_path.resolve()),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "params_m": float(total_params / 1e6),
        "model_size_mb": float(model_size_mb),
        "gflops": gflops,
    }
    if gflops_note is not None:
        result["note"] = gflops_note
    return result


def _extract_state_dict(checkpoint: Any) -> dict:
    """torch 체크포인트에서 파라미터 텐서가 담긴 state_dict 를 꺼낸다."""
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        ema = checkpoint.get("ema")
        if isinstance(ema, dict):
            return ema.get("module", ema)
        return checkpoint
    return {}


def _complexity_from_state_dict(weight_path: Path, *, note: str) -> dict[str, Any]:
    """ultralytics 가 아닌 체크포인트(rf_detr/dino)의 파라미터 수를 state_dict 로 센다."""
    try:
        checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=False)
    except TypeError:  # 구버전 torch 는 weights_only 인자 없음
        checkpoint = torch.load(str(weight_path), map_location="cpu")
    except Exception as exc:
        return {
            "weight_reference": str(weight_path.resolve()),
            "params_m": None,
            "gflops": None,
            "note": f"param count failed: {exc}",
        }

    state = _extract_state_dict(checkpoint)
    total_params = sum(int(tensor.numel()) for tensor in state.values() if hasattr(tensor, "numel"))
    return {
        "weight_reference": str(weight_path.resolve()),
        "total_params": int(total_params),
        "params_m": float(total_params / 1e6) if total_params else None,
        "model_size_mb": float(total_params * 4 / (1024**2)) if total_params else None,
        "gflops": None,
        "note": note,
    }
