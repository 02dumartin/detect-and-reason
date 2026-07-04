from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:  # 형제 스크립트와 동일하게 프로젝트 루트를 cwd 로 맞춘다.
    import autorootcwd  # noqa: F401
except Exception:  # autorootcwd 미설치 환경(예: system python)에서도 동작하게 fallback
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)

from src.config_loader import load_yaml, resolve_path, resolve_runtime_config
from src.visualization import save_coco_prediction_overlays

# CLI --split (train/val/test) → 디스크 COCO split 디렉터리 (Roboflow: valid)
_SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


def parse_args() -> argparse.Namespace:
    """COCO 예측/GT overlay 렌더링 인자 (render_yolo_overlays.py 와 동일 체계)."""
    parser = argparse.ArgumentParser(
        description="COCO 예측(predictions_coco.json) 또는 GT를 bbox overlay 이미지로 저장한다 "
        "(rf_detr / dino 공용)."
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="config/model 아래의 모델 설정 이름")
    model_group.add_argument("--model-config", type=str, help="직접 지정할 모델 YAML 경로")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str, help="config/dataset 아래의 데이터셋 설정 이름")
    dataset_group.add_argument("--dataset-config", type=str, help="직접 지정할 데이터셋 YAML 경로")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--pred-json",
        type=str,
        help="예측 COCO json 경로. 생략하면 runtime prediction_dir/<split>/predictions_coco.json 사용.",
    )
    parser.add_argument("--img-root", type=str, help="원본 이미지 폴더. 생략하면 COCO split 디렉터리 사용.")
    parser.add_argument("--output-dir", type=str, help="overlay 저장 디렉터리")
    parser.add_argument(
        "--overlay-config",
        type=str,
        help="시각화 스타일 YAML. 생략하면 config/visualization/detection_overlay.yaml 사용.",
    )
    parser.add_argument("--font-size", type=int, help="폰트 크기 override")
    parser.add_argument("--box-thickness", type=int, help="bbox 선 두께 override")
    parser.add_argument("--label-only", action="store_true", help="confidence 없이 클래스명만 표시")
    parser.add_argument("--ground-truth", action="store_true", help="예측 대신 GT annotation 을 overlay한다.")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="이 score 미만 예측은 그리지 않는다.")
    parser.add_argument("--max-images", type=int, help="앞에서부터 N장만 렌더링한다.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_ref = args.model_config or args.model
    dataset_ref = args.dataset_config or args.dataset
    runtime_cfg = resolve_runtime_config(model_ref=model_ref, dataset_ref=dataset_ref, stage="predict")

    split_dir_name = _SPLIT_DIR[args.split]
    rfdetr_dir = Path(runtime_cfg["dataset"]["rfdetr_dir"])
    split_dir = rfdetr_dir / split_dir_name
    gt_json = split_dir / "_annotations.coco.json"
    if not gt_json.is_file():
        raise FileNotFoundError(f"COCO GT annotation not found: {gt_json}")

    img_root = Path(args.img_root) if args.img_root else split_dir

    if args.ground_truth:
        pred_json = None
        output_dir = Path(args.output_dir) if args.output_dir else split_dir / "gt_overlays"
    else:
        default_pred = Path(runtime_cfg["paths"]["prediction_dir"]) / split_dir_name / "predictions_coco.json"
        pred_json = Path(args.pred_json) if args.pred_json else default_pred
        if not pred_json.is_file():
            raise FileNotFoundError(
                f"예측 json 이 없습니다: {pred_json}\n"
                f"먼저 predict 를 돌리세요: python scripts/train_model.py "
                f"--model {model_ref} --dataset {dataset_ref} --stage predict --source {args.split}"
            )
        output_dir = Path(args.output_dir) if args.output_dir else pred_json.parent / "overlays"

    overlay_cfg = _load_overlay_config(
        project_root=runtime_cfg["project_root"],
        overlay_config_ref=args.overlay_config,
    )
    class_names = _resolve_display_class_names(runtime_cfg)

    rendered = save_coco_prediction_overlays(
        gt_json=gt_json,
        pred_json=pred_json,
        img_root=img_root,
        output_dir=output_dir,
        class_names=class_names,
        overlay_config=overlay_cfg,
        font_size=args.font_size,
        box_thickness=args.box_thickness,
        label_only=True if args.label_only else None,
        max_images=args.max_images,
        conf_threshold=args.conf_threshold,
        ground_truth=args.ground_truth,
    )

    print(f"[render_coco_overlays] mode={'GT' if args.ground_truth else 'pred'}")
    if not args.ground_truth:
        print(f"[render_coco_overlays] pred_json={pred_json} conf>={args.conf_threshold}")
    print(f"[render_coco_overlays] img_root={img_root}")
    print(f"[render_coco_overlays] output_dir={output_dir}")
    print(f"[render_coco_overlays] rendered={rendered}")


def _load_overlay_config(*, project_root: Path, overlay_config_ref: str | None) -> dict:
    config_ref = overlay_config_ref or "config/visualization/detection_overlay.yaml"
    config_path = resolve_path(config_ref, project_root)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"overlay config not found: {config_ref}")
    return load_yaml(config_path)


def _resolve_display_class_names(runtime_cfg: dict) -> dict[int, str] | list[str]:
    model_class_names = runtime_cfg.get("class_names") or {}
    if model_class_names:
        return model_class_names
    return runtime_cfg["dataset"]["class_names"]


if __name__ == "__main__":
    main()
