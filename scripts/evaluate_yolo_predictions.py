from __future__ import annotations

import argparse
from pathlib import Path

import autorootcwd

from src.config_loader import resolve_runtime_config
from src.evaluation import evaluate_yolo_label_predictions, save_detection_evaluation_artifacts
from src.evaluation.yolo_label_io import resolve_prediction_labels_dir, resolve_yolo_split_dirs


def parse_args() -> argparse.Namespace:
    """평가에 필요한 인자를 정의한다."""
    parser = argparse.ArgumentParser(description="YOLO txt 예측 결과를 GT와 비교해 평가한다.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="config/model 아래의 모델 설정 이름")
    model_group.add_argument("--model-config", type=str, help="직접 지정할 모델 YAML 경로")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str, help="config/dataset 아래의 데이터셋 설정 이름")
    dataset_group.add_argument("--dataset-config", type=str, help="직접 지정할 데이터셋 YAML 경로")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--pred-labels-dir",
        type=str,
        help="예측 labels 폴더 또는 prediction 루트 폴더. 생략하면 runtime prediction_dir/labels를 사용한다.",
    )
    parser.add_argument("--output-dir", type=str, help="평가 산출물 저장 디렉터리")
    parser.add_argument("--iou-threshold", type=float, help="TP 매칭에 사용할 IoU threshold")
    return parser.parse_args()


def main() -> None:
    """런타임 설정을 해석하고 평가를 실행한 뒤 결과를 저장한다."""
    args = parse_args()

    model_ref = args.model_config or args.model
    dataset_ref = args.dataset_config or args.dataset
    runtime_cfg = resolve_runtime_config(model_ref=model_ref, dataset_ref=dataset_ref, stage="predict")

    split_path = runtime_cfg["dataset"][f"{args.split}_dir"]
    gt_images_dir, gt_labels_dir = resolve_yolo_split_dirs(split_path)

    default_pred_root = runtime_cfg["paths"]["prediction_dir"]
    pred_labels_dir = resolve_prediction_labels_dir(args.pred_labels_dir or default_pred_root)
    output_dir = Path(args.output_dir) if args.output_dir else pred_labels_dir.parent / "evaluation"

    iou_threshold = (
        args.iou_threshold
        if args.iou_threshold is not None
        else runtime_cfg["predict"].get("resolved_iou") or 0.5
    )

    # 데이터셋 설정에 보조용 이름이 들어 있을 수 있어서,
    # 사용자에게 보여 줄 클래스명은 모델 설정의 canonical label을 우선 사용한다.
    class_names = _resolve_display_class_names(runtime_cfg)

    results = evaluate_yolo_label_predictions(
        gt_images_dir=gt_images_dir,
        gt_labels_dir=gt_labels_dir,
        pred_labels_dir=pred_labels_dir,
        class_names=class_names,
        iou_threshold=iou_threshold,
        weight_reference=runtime_cfg["paths"]["weights"],
        model_imgsz=int(runtime_cfg.get("train", {}).get("imgsz", 640)),
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
    print(f"[evaluate_yolo_predictions] split={args.split}")
    print(f"[evaluate_yolo_predictions] pred_labels_dir={pred_labels_dir}")
    print(f"[evaluate_yolo_predictions] output_dir={output_dir}")
    print(
        "[evaluate_yolo_predictions] "
        f"precision={stats['overall_precision']:.4f} "
        f"recall={stats['overall_recall']:.4f} "
        f"f1={stats['overall_f1']:.4f} "
        f"detection_acc={stats['detection_acc']:.4f} "
        f"classification_acc={stats['classification_acc']:.4f} "
        f"overall_acc={stats['overall_acc']:.4f}"
    )
    print(
        "[evaluate_yolo_predictions] "
        f"mAP50={_fmt_metric(metrics.get('map_50'))} "
        f"mAP50-95={_fmt_metric(metrics.get('map'))} "
        f"mAP75={_fmt_metric(metrics.get('map_75'))} "
        f"CA-mAP50={_fmt_metric(metrics.get('ca_map_50'))} "
        f"CA-mAP50-95={_fmt_metric(metrics.get('ca_map'))}"
    )
    print(f"[evaluate_yolo_predictions] saved_json={artifact_paths['json']}")


def _resolve_display_class_names(runtime_cfg: dict) -> dict[int, str] | list[str]:
    """모델 설정 클래스명이 있으면 그것을, 없으면 데이터셋 클래스명을 사용한다."""
    model_class_names = runtime_cfg.get("class_names") or {}
    if model_class_names:
        return model_class_names
    return runtime_cfg["dataset"]["class_names"]


def _fmt_metric(value: float | None) -> str:
    """None 값이 섞여도 출력 문자열이 깨지지 않게 처리한다."""
    return "N/A" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
