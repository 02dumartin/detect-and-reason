from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.coco_label_io import load_coco_gt
from src.evaluation.yolo_label_io import (
    build_image_index,
    load_yolo_label_rows,
    resolve_prediction_labels_dir,
    resolve_yolo_split_dirs,
    yolo_row_to_detection,
)


def load_or_create_normalized_detections(
    *,
    runtime_cfg: dict[str, Any],
    split: str,
    prediction_dir: Path,
    refresh: bool = False,
) -> tuple[dict[str, Any], Path, bool]:
    cache_dir = prediction_dir / "vlm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_normalized_input_detections.json"

    if cache_path.is_file() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8")), cache_path, True

    family = str(runtime_cfg.get("family") or "")
    if family in {"rf_detr", "dino"}:
        normalized = _normalize_coco_predictions(runtime_cfg=runtime_cfg, split=split, prediction_dir=prediction_dir)
    else:
        normalized = _normalize_yolo_predictions(runtime_cfg=runtime_cfg, prediction_dir=prediction_dir)

    cache_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalized, cache_path, False


def _normalize_yolo_predictions(*, runtime_cfg: dict[str, Any], prediction_dir: Path) -> dict[str, Any]:
    split_name = str(runtime_cfg.get("predict", {}).get("resolved_source_name") or "test")
    split_dir = runtime_cfg["dataset"][f"{split_name}_dir"]
    images_dir, _ = resolve_yolo_split_dirs(split_dir)
    labels_dir = resolve_prediction_labels_dir(prediction_dir)
    image_index = build_image_index(images_dir)
    detector_class_names = runtime_cfg.get("class_names") or runtime_cfg["dataset"].get("class_names") or {}

    records: list[dict[str, Any]] = []
    for stem, image_info in image_index.items():
        rows = load_yolo_label_rows(labels_dir / f"{stem}.txt")
        for detection_index, row in enumerate(rows):
            class_id, bbox_xyxy, score = yolo_row_to_detection(row, image_info.width, image_info.height)
            x1, y1, x2, y2 = bbox_xyxy
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            records.append(
                {
                    "record_id": f"{stem}#{detection_index}",
                    "image_id": None,
                    "file_name": image_info.file_name,
                    "image_path": str(image_info.path),
                    "image_width": image_info.width,
                    "image_height": image_info.height,
                    "source_format": "yolo_txt",
                    "source_prediction_path": str(labels_dir / f"{stem}.txt"),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, box_w, box_h],
                    "bbox_yolo": [float(row[idx]) for idx in range(1, 5)],
                    "det_score": float(score),
                    "detector_class_id": int(class_id),
                    "detector_class_name": str(detector_class_names.get(int(class_id), class_id)),
                    "detection_index": detection_index,
                }
            )

    return {
        "schema_version": "vlm-normalized-detections-v1",
        "detector_family": runtime_cfg.get("family"),
        "model_name": runtime_cfg["model_name"],
        "dataset_key": runtime_cfg["dataset"]["key"],
        "source_prediction_dir": str(prediction_dir),
        "source_image_root": str(images_dir),
        "records": records,
    }


def _normalize_coco_predictions(
    *,
    runtime_cfg: dict[str, Any],
    split: str,
    prediction_dir: Path,
) -> dict[str, Any]:
    coco_split = _normalize_coco_split_name(split)
    pred_path = prediction_dir / coco_split / "predictions_coco.json"
    if not pred_path.is_file():
        raise FileNotFoundError(f"prediction COCO json not found: {pred_path}")

    split_name = str(runtime_cfg.get("predict", {}).get("resolved_source_name") or split or "test")
    split_dir = runtime_cfg["dataset"][f"{split_name}_dir"]
    images_dir, _ = resolve_yolo_split_dirs(split_dir)
    image_index = build_image_index(images_dir)

    gt_ann_path = Path(runtime_cfg["dataset"]["rfdetr_dir"]) / coco_split / "_annotations.coco.json"
    gt_by_image, _ = load_coco_gt(gt_ann_path)
    file_to_image_id = {info["file_name"]: image_id for image_id, info in gt_by_image.items()}

    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    detector_class_names = runtime_cfg.get("class_names") or runtime_cfg["dataset"].get("class_names") or {}
    per_stem_count: dict[str, int] = {}
    records: list[dict[str, Any]] = []

    for det in predictions:
        image_id = int(det["image_id"])
        image_meta = gt_by_image.get(image_id)
        if image_meta is None:
            continue

        file_name = str(image_meta["file_name"])
        stem = Path(file_name).stem
        image_info = image_index.get(stem)
        if image_info is None:
            continue

        x, y, w, h = (float(value) for value in det["bbox"][:4])
        x1, y1, x2, y2 = [x, y, x + w, y + h]
        detection_index = per_stem_count.get(stem, 0)
        per_stem_count[stem] = detection_index + 1
        xc = ((x1 + x2) / 2.0) / image_info.width
        yc = ((y1 + y2) / 2.0) / image_info.height
        wn = max(0.0, x2 - x1) / image_info.width
        hn = max(0.0, y2 - y1) / image_info.height
        class_id = int(det.get("category_id", 0))

        records.append(
            {
                "record_id": f"{stem}#{detection_index}",
                "image_id": image_id,
                "file_name": image_info.file_name,
                "image_path": str(image_info.path),
                "image_width": image_info.width,
                "image_height": image_info.height,
                "source_format": "coco_json",
                "source_prediction_path": str(pred_path),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x, y, w, h],
                "bbox_yolo": [xc, yc, wn, hn],
                "det_score": float(det.get("score", 1.0)),
                "detector_class_id": class_id,
                "detector_class_name": str(detector_class_names.get(class_id, class_id)),
                "detection_index": detection_index,
            }
        )

    return {
        "schema_version": "vlm-normalized-detections-v1",
        "detector_family": runtime_cfg.get("family"),
        "model_name": runtime_cfg["model_name"],
        "dataset_key": runtime_cfg["dataset"]["key"],
        "source_prediction_dir": str(prediction_dir),
        "source_image_root": str(images_dir),
        "source_prediction_coco": str(pred_path),
        "source_gt_coco": str(gt_ann_path),
        "records": records,
        "image_ids": file_to_image_id,
    }


def _normalize_coco_split_name(split: str) -> str:
    return {"val": "valid", "validation": "valid"}.get(split.strip().lower(), split.strip().lower())
