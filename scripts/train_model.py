"""
단일 모델 기준으로 train / validate / predict를 실행하는 진입 스크립트

Examples:
    python scripts/train_model.py --model yolo11_3cls --dataset big --stage train
    python scripts/train_model.py --model yolo11_3cls --dataset big --stage validate
    python scripts/train_model.py --model yolo11_3cls --dataset big --stage predict --source test
"""

from __future__ import annotations

import argparse

import autorootcwd

# 설정 파일을 읽어 stage별 실행 함수를 호출
from src.config_loader import resolve_runtime_config
from src.model.predict import run_predict
from src.model.train import run_train
from src.model.validate import run_validate


def parse_args() -> argparse.Namespace:
    """CLI 인자를 정의하고 파싱한다."""
    parser = argparse.ArgumentParser(description="Train / validate / predict a single detector.")

    # 모델은 이름으로 찾거나, YAML 경로를 직접 넘기거나 둘 중 하나만 허용
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Model config name under config/model.")
    model_group.add_argument("--model-config", type=str, help="Explicit model config path.")

    # 데이터셋도 동일하게 이름 기반 또는 파일 경로 기반 입력만 받음
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str, help="Dataset config name under config/dataset.")
    dataset_group.add_argument("--dataset-config", type=str, help="Explicit dataset config path.")

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["train", "validate", "predict"],
        help="Execution stage.",
    )
    parser.add_argument("--weights", type=str, help="Optional model weight path or model alias.")
    parser.add_argument(
        "--source",
        type=str,
        help="Predict source path or split name (train/val/test). Only used for predict stage.",
    )

    return parser.parse_args()


def main() -> None:
    """실행 인자를 바탕으로 런타임 설정을 만들고 해당 stage 함수를 호출한다."""
    args = parse_args()

    # 사용자가 config 파일 경로를 직접 넘겼다면 그것을, 아니면 이름 기반 참조를 사용
    model_ref = args.model_config or args.model
    dataset_ref = args.dataset_config or args.dataset

    runtime_cfg = resolve_runtime_config(
        model_ref=model_ref,
        dataset_ref=dataset_ref,
        stage=args.stage,
        weights=args.weights,
        source=args.source,
    )

    _print_runtime_summary(runtime_cfg)

    # stage별 실행 함수는 동일한 runtime dict를 입력으로 받도록 맞춤
    if args.stage == "train":
        run_train(runtime_cfg)
    elif args.stage == "validate":
        run_validate(runtime_cfg)
    elif args.stage == "predict":
        run_predict(runtime_cfg)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


def _print_runtime_summary(runtime_cfg: dict) -> None:
    """실행 직전에 핵심 설정값을 로그로 출력해 디버깅을 쉽게 만든다."""
    print(f"[train_model] stage={runtime_cfg['stage']}")
    print(
        f"[train_model] model={runtime_cfg['model_name']} "
        f"family={runtime_cfg['family']} class_mode={runtime_cfg['class_mode']}"
    )
    print(
        f"[train_model] dataset={runtime_cfg['dataset']['key']} "
        f"data_yaml={runtime_cfg['dataset']['data_yaml']}"
    )

    # stage마다 확인하고 싶은 출력 경로가 달라서 로그 항목도 구분
    if runtime_cfg["stage"] == "train":
        print(f"[train_model] runs_dir={runtime_cfg['paths']['runs_dir']}")
    elif runtime_cfg["stage"] == "validate":
        print(
            f"[train_model] split={runtime_cfg['val']['resolved_split']} "
            f"eval_dir={runtime_cfg['paths']['eval_dir']}"
        )
    elif runtime_cfg["stage"] == "predict":
        print(
            f"[train_model] source={runtime_cfg['predict']['resolved_source']} "
            f"prediction_dir={runtime_cfg['paths']['prediction_dir']}"
        )


if __name__ == "__main__":
    main()
