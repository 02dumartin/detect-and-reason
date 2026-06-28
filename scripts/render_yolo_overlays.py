from __future__ import annotations

import argparse
from pathlib import Path

import autorootcwd

from src.config_loader import load_yaml, resolve_path, resolve_runtime_config
from src.evaluation.yolo_label_io import resolve_prediction_labels_dir, resolve_yolo_split_dirs
from src.visualization import save_yolo_label_overlays


def parse_args() -> argparse.Namespace:
    """overlay 렌더링에 필요한 인자를 정의한다."""
    parser = argparse.ArgumentParser(description="YOLO txt 예측 또는 GT를 bbox overlay 이미지로 저장한다.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="config/model 아래의 모델 설정 이름")
    model_group.add_argument("--model-config", type=str, help="직접 지정할 모델 YAML 경로")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str, help="config/dataset 아래의 데이터셋 설정 이름")
    dataset_group.add_argument("--dataset-config", type=str, help="직접 지정할 데이터셋 YAML 경로")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--labels-dir",
        type=str,
        help="예측 labels 폴더 또는 prediction 루트 폴더. 생략하면 runtime prediction_dir/labels를 사용한다.",
    )
    parser.add_argument("--img-root", type=str, help="원본 이미지 폴더. 생략하면 split의 images 폴더를 사용한다.")
    parser.add_argument("--output-dir", type=str, help="overlay 저장 디렉터리")
    parser.add_argument(
        "--overlay-config",
        type=str,
        help="시각화 스타일 YAML 경로. 생략하면 config/visualization/detection_overlay.yaml을 사용한다.",
    )
    parser.add_argument("--font-size", type=int, help="폰트 크기 override")
    parser.add_argument("--box-thickness", type=int, help="bbox 선 두께 override")
    parser.add_argument("--label-only", action="store_true", help="confidence 없이 클래스명만 표시")
    parser.add_argument("--ground-truth", action="store_true", help="예측 대신 GT labels를 overlay한다.")
    parser.add_argument("--max-images", type=int, help="앞에서부터 N장만 렌더링한다.")
    return parser.parse_args()


def main() -> None:
    """런타임 설정과 overlay 스타일을 합쳐 실제 렌더링을 수행한다."""
    args = parse_args()

    model_ref = args.model_config or args.model
    dataset_ref = args.dataset_config or args.dataset
    runtime_cfg = resolve_runtime_config(model_ref=model_ref, dataset_ref=dataset_ref, stage="predict")

    gt_images_dir, gt_labels_dir = resolve_yolo_split_dirs(runtime_cfg["dataset"][f"{args.split}_dir"])
    img_root = Path(args.img_root) if args.img_root else gt_images_dir

    if args.ground_truth:
        labels_dir = gt_labels_dir
        output_dir = Path(args.output_dir) if args.output_dir else gt_images_dir.parent / "gt_overlays"
    else:
        default_pred_root = runtime_cfg["paths"]["prediction_dir"]
        labels_dir = resolve_prediction_labels_dir(args.labels_dir or default_pred_root)
        output_dir = Path(args.output_dir) if args.output_dir else labels_dir.parent / "overlays"

    overlay_cfg = _load_overlay_config(
        project_root=runtime_cfg["project_root"],
        overlay_config_ref=args.overlay_config,
    )
    class_names = _resolve_display_class_names(runtime_cfg)

    rendered = save_yolo_label_overlays(
        labels_dir=labels_dir,
        img_root=img_root,
        output_dir=output_dir,
        class_names=class_names,
        overlay_config=overlay_cfg,
        font_size=args.font_size,
        box_thickness=args.box_thickness,
        label_only=True if args.label_only else None,
        max_images=args.max_images,
    )

    print(f"[render_yolo_overlays] labels_dir={labels_dir}")
    print(f"[render_yolo_overlays] img_root={img_root}")
    print(f"[render_yolo_overlays] output_dir={output_dir}")
    print(f"[render_yolo_overlays] rendered={rendered}")


def _load_overlay_config(*, project_root: Path, overlay_config_ref: str | None) -> dict:
    """overlay YAML 경로를 해석하고 실제 내용을 읽는다."""
    config_ref = overlay_config_ref or "config/visualization/detection_overlay.yaml"
    config_path = resolve_path(config_ref, project_root)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"overlay config not found: {config_ref}")
    return load_yaml(config_path)


def _resolve_display_class_names(runtime_cfg: dict) -> dict[int, str] | list[str]:
    """사용자에게 보여 줄 클래스명은 모델 설정의 canonical label을 우선 사용한다."""
    model_class_names = runtime_cfg.get("class_names") or {}
    if model_class_names:
        return model_class_names
    return runtime_cfg["dataset"]["class_names"]


if __name__ == "__main__":
    main()
