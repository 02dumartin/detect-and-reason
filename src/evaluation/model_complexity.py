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

    model = YOLO(str(weight_path.resolve()))
    torch_model = model.model

    total_params = sum(parameter.numel() for parameter in torch_model.parameters())
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
