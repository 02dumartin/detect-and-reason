from __future__ import annotations

import argparse
from pathlib import Path

import autorootcwd

from src.config_loader import resolve_runtime_config
from src.model.predict import run_predict


def parse_args() -> argparse.Namespace:
    """YOLO txt 예측 전용 CLI 인자를 정의한다.

    이 스크립트의 역할은 "예측 라벨 생성"에만 집중하는 것이다.
    그래서 evaluation/overlay는 하지 않고, 기본값도 txt 저장 중심으로 맞춘다.
    """
    parser = argparse.ArgumentParser(description="YOLO txt 예측 결과만 별도 디렉터리에 저장한다.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="config/model 아래의 모델 설정 이름")
    model_group.add_argument("--model-config", type=str, help="직접 지정할 모델 YAML 경로")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str, help="config/dataset 아래의 데이터셋 설정 이름")
    dataset_group.add_argument("--dataset-config", type=str, help="직접 지정할 데이터셋 YAML 경로")

    parser.add_argument(
        "--weights",
        type=str,
        help="예측에 사용할 가중치 경로. 생략하면 resolve_runtime_config 규칙을 따른다.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="test",
        help="예측 입력. train/val/test 같은 split 이름 또는 이미지 폴더 경로를 받을 수 있다.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="예측 결과 저장 디렉터리. 생략하면 model config의 prediction_dir를 사용한다.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Ultralytics 기본 예측 이미지까지 함께 저장한다. 기본값은 txt만 저장한다.",
    )
    parser.add_argument("--conf", type=float, help="confidence threshold override")
    parser.add_argument("--iou", type=float, help="NMS IoU threshold override")
    return parser.parse_args()


def main() -> None:
    """런타임 설정을 만든 뒤 txt 예측만 저장하도록 실행한다."""
    args = parse_args()

    model_ref = args.model_config or args.model
    dataset_ref = args.dataset_config or args.dataset
    runtime_cfg = resolve_runtime_config(
        model_ref=model_ref,
        dataset_ref=dataset_ref,
        stage="predict",
        weights=args.weights,
        source=args.source,
    )

    # prediction / evaluation / overlay를 서로 다른 폴더로 분리해서 쓰고 싶다는
    # 요구가 많아서, prediction 스크립트는 output-dir override를 직접 받는다.
    if args.output_dir:
        runtime_cfg["paths"]["prediction_dir"] = _resolve_output_dir(
            output_dir_ref=args.output_dir,
            project_root=runtime_cfg["project_root"],
        )

    # 이 스크립트의 기본 목적은 "라벨(txt) 생성"이므로, 예측 이미지 저장은 꺼 둔다.
    # 필요할 때만 --save-images를 줘서 Ultralytics 기본 결과 이미지를 남긴다.
    runtime_cfg["predict"]["save"] = bool(args.save_images)
    runtime_cfg["predict"]["save_txt"] = True

    # score/conf 값이 평가나 후처리에 쓰일 수 있으므로, save_conf는 config 기본값을
    # 따르되 비어 있으면 안전하게 True로 맞춘다.
    if runtime_cfg["predict"].get("save_conf") is None:
        runtime_cfg["predict"]["save_conf"] = True

    if args.conf is not None:
        runtime_cfg["predict"]["conf"] = float(args.conf)
    if args.iou is not None:
        runtime_cfg["predict"]["resolved_iou"] = float(args.iou)

    _print_runtime_summary(runtime_cfg)
    run_predict(runtime_cfg)

    prediction_dir = Path(runtime_cfg["paths"]["prediction_dir"])
    print(f"[predict_yolo_labels] prediction_dir={prediction_dir}")
    print(f"[predict_yolo_labels] labels_dir={prediction_dir / 'labels'}")
    print(f"[predict_yolo_labels] save_images={runtime_cfg['predict']['save']}")


def _resolve_output_dir(*, output_dir_ref: str, project_root: Path) -> Path:
    """사용자가 지정한 output dir를 실제 경로로 정규화한다.

    절대 경로면 그대로 쓰고, 상대 경로면 프로젝트 루트 기준으로 해석한다.
    아직 존재하지 않는 새 폴더를 만들 수 있어야 하므로 `resolve_path` 대신
    여기서 직접 규칙을 단순하게 적용한다.
    """
    raw = Path(output_dir_ref).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (project_root / raw).resolve()


def _print_runtime_summary(runtime_cfg: dict) -> None:
    """실행 전에 핵심 경로와 옵션을 로그로 남긴다."""
    print(f"[predict_yolo_labels] model={runtime_cfg['model_name']}")
    print(f"[predict_yolo_labels] dataset={runtime_cfg['dataset']['key']}")
    print(f"[predict_yolo_labels] weights={runtime_cfg['paths']['weights']}")
    print(f"[predict_yolo_labels] source={runtime_cfg['predict']['resolved_source']}")
    print(f"[predict_yolo_labels] prediction_dir={runtime_cfg['paths']['prediction_dir']}")
    print(f"[predict_yolo_labels] conf={runtime_cfg['predict'].get('conf')}")
    print(f"[predict_yolo_labels] iou={runtime_cfg['predict'].get('resolved_iou')}")


if __name__ == "__main__":
    main()
