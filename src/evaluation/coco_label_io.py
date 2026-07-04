from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _coco_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """COCO bbox [x, y, w, h](좌상단 기준)를 [x1, y1, x2, y2] 픽셀 좌표로 바꾼다."""
    x, y, w, h = (float(value) for value in bbox[:4])
    return [x, y, x + w, y + h]


def class_name_map_from_categories(categories: list[dict[str, Any]]) -> dict[int, str]:
    """COCO categories 배열을 `class_id -> class_name` dict로 정리한다."""
    return {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in sorted(categories, key=lambda item: int(item["id"]))
    }


def load_coco_gt(ann_path: str | Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """COCO GT 어노테이션을 읽어 `(image_id -> gt 정보, categories)`로 돌려준다.

    각 이미지의 gt 정보에는 `file_name`, `gt_labels`, `gt_boxes`(xyxy 픽셀)가 담긴다.
    """
    ann_path = Path(ann_path)
    coco = json.loads(ann_path.read_text(encoding="utf-8"))

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in coco.get("images", []):
        image_id = int(image["id"])
        images_by_id[image_id] = {
            "file_name": str(image.get("file_name", f"{image_id}")),
            "gt_labels": [],
            "gt_boxes": [],
        }

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in images_by_id:
            continue
        images_by_id[image_id]["gt_labels"].append(int(ann["category_id"]))
        images_by_id[image_id]["gt_boxes"].append(_coco_xywh_to_xyxy(ann["bbox"]))

    return images_by_id, coco.get("categories", [])


def load_coco_predictions(
    pred_path: str | Path,
    conf_threshold: float = 0.0,
) -> dict[int, dict[str, list]]:
    """COCO 예측 json(평면 detection 리스트)을 `image_id -> 예측 배열`로 묶는다.

    `convert_predictions_to_coco`가 저장하는 형식
    `[{"image_id", "category_id", "bbox": [x,y,w,h], "score"}, ...]`을 그대로 받는다.
    `conf_threshold` 미만 점수는 버린다(기본 0.0 = 추가 필터 없음).
    """
    pred_path = Path(pred_path)
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))

    grouped: dict[int, dict[str, list]] = defaultdict(
        lambda: {"pred_labels": [], "pred_boxes": [], "pred_scores": []}
    )
    for det in predictions:
        score = float(det.get("score", 1.0))
        if score < conf_threshold:
            continue
        image_id = int(det["image_id"])
        grouped[image_id]["pred_labels"].append(int(det["category_id"]))
        grouped[image_id]["pred_boxes"].append(_coco_xywh_to_xyxy(det["bbox"]))
        grouped[image_id]["pred_scores"].append(score)
    return grouped
