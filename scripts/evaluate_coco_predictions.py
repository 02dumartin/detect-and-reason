from __future__ import annotations

import argparse
from pathlib import Path

import autorootcwd

from src.config_loader import resolve_runtime_config
from src.evaluation import evaluate_coco_predictions, save_detection_evaluation_artifacts

# COCO/Roboflow split 폴더명 정규화 (rf_detr 산출물과 동일 규칙)
_SPLIT_ALIASES = {"val": "valid", "validation": "valid"}


def _normalize_split(split: str) -> str:
    return _SPLIT_ALIASES.get(split.strip().lower(), split.strip().lower())


def parse_args() -> argparse.Namespace:
    """평가에 필요한 인자를 정의한다."""
    parser = argparse.ArgumentParser(
        description="RF-DETR(COCO json) 예측 결과를 GT와 비교해 YOLO 경로와 동일 metric으로 평가한다."
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="config/model 아래의 모델 설정 이름")
    model_group.add_argument("--model-config", type=str, help="직접 지정할 모델 YAML 경로")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str, help="config/dataset 아래의 데이터셋 설정 이름")
    dataset_group.add_argument("--dataset-config", type=str, help="직접 지정할 데이터셋 YAML 경로")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--pred-coco",
        type=str,
        help="예측 COCO json 경로. 생략하면 prediction_dir/<split>/predictions_coco.json을 사용한다.",
    )
    parser.add_argument("--output-dir", type=str, help="평가 산출물 저장 디렉터리")
    parser.add_argument("--iou-threshold", type=float, help="TP 매칭에 사용할 IoU threshold")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="예측 추가 필터 conf (기본 0.0; 예측 json이 이미 필터돼 있으면 0으로 둔다).",
    )
    return parser.parse_args()


def main() -> None:
    """런타임 설정을 해석하고 COCO 평가를 실행한 뒤 결과를 저장한다."""
    args = parse_args()

    model_ref = args.model_config or args.model
    dataset_ref = args.dataset_config or args.dataset
    runtime_cfg = resolve_runtime_config(model_ref=model_ref, dataset_ref=dataset_ref, stage="predict")

    if runtime_cfg.get("family") not in {"rf_detr", "dino"}:
        raise SystemExit("evaluate_coco_predictions는 COCO 계열(rf_detr / dino) 전용이다. YOLO 계열은 evaluate_yolo_predictions를 사용하라.")

    split = _normalize_split(args.split)
    rfdetr_dir = Path(runtime_cfg["dataset"]["rfdetr_dir"])
    gt_ann_path = rfdetr_dir / split / "_annotations.coco.json"
    if not gt_ann_path.is_file():
        raise FileNotFoundError(f"COCO GT 어노테이션을 찾을 수 없다: {gt_ann_path}")

    prediction_dir = runtime_cfg["paths"]["prediction_dir"]
    pred_coco_path = (
        Path(args.pred_coco)
        if args.pred_coco
        else Path(prediction_dir) / split / "predictions_coco.json"
    )
    if not pred_coco_path.is_file():
        raise FileNotFoundError(
            f"예측 COCO json을 찾을 수 없다: {pred_coco_path} "
            "(먼저 --stage predict로 예측을 생성하라)"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path(prediction_dir) / "evaluation"

    iou_threshold = (
        args.iou_threshold
        if args.iou_threshold is not None
        else runtime_cfg["predict"].get("resolved_iou") or 0.5
    )

    # 모델 설정의 canonical 클래스명을 우선 사용하고, 없으면 GT categories에서 추론한다.
    class_names = _resolve_display_class_names(runtime_cfg)

    results = evaluate_coco_predictions(
        gt_ann_path=gt_ann_path,
        pred_coco_path=pred_coco_path,
        class_names=class_names,
        iou_threshold=iou_threshold,
        conf_threshold=args.conf_threshold,
        weight_reference=runtime_cfg["paths"]["weights"],
        model_imgsz=int(runtime_cfg.get("train", {}).get("imgsz", 640) or 640),
    )

    results["evaluation_info"].update(
        {
            "model_name": runtime_cfg["model_name"],
            "dataset_key": runtime_cfg["dataset"]["key"],
            "split": args.split,
            "model_config_path": str(runtime_cfg["model_config_path"]),
            "dataset_config_path": str(runtime_cfg["dataset_config_path"]),
        }
    )

    artifact_paths = save_detection_evaluation_artifacts(results, output_dir)

    stats = results["detailed_statistics"]["total_statistics"]
    metrics = results["detection_metrics"]
    print(f"[evaluate_coco_predictions] split={args.split}")
    print(f"[evaluate_coco_predictions] gt_ann={gt_ann_path}")
    print(f"[evaluate_coco_predictions] pred_coco={pred_coco_path}")
    print(f"[evaluate_coco_predictions] output_dir={output_dir}")
    print(
        "[evaluate_coco_predictions] "
        f"precision={stats['overall_precision']:.4f} "
        f"recall={stats['overall_recall']:.4f} "
        f"f1={stats['overall_f1']:.4f} "
        f"detection_acc={stats['detection_acc']:.4f} "
        f"classification_acc={stats['classification_acc']:.4f} "
        f"overall_acc={stats['overall_acc']:.4f}"
    )
    print(
        "[evaluate_coco_predictions] "
        f"mAP50={_fmt_metric(metrics.get('map_50'))} "
        f"mAP50-95={_fmt_metric(metrics.get('map'))} "
        f"mAP75={_fmt_metric(metrics.get('map_75'))} "
        f"CA-mAP50={_fmt_metric(metrics.get('ca_map_50'))} "
        f"CA-mAP50-95={_fmt_metric(metrics.get('ca_map'))}"
    )
    print(f"[evaluate_coco_predictions] saved_json={artifact_paths['json']}")


def _resolve_display_class_names(runtime_cfg: dict) -> dict[int, str] | list[str] | None:
    """모델 설정 클래스명이 있으면 그것을, 없으면 데이터셋 클래스명, 둘 다 없으면 None."""
    model_class_names = runtime_cfg.get("class_names") or {}
    if model_class_names:
        return model_class_names
    return runtime_cfg["dataset"].get("class_names") or None


def _fmt_metric(value: float | None) -> str:
    """None 값이 섞여도 출력 문자열이 깨지지 않게 처리한다."""
    return "N/A" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
