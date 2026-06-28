from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .model_complexity import estimate_ultralytics_model_complexity
from .yolo_label_io import (
    build_image_index,
    load_yolo_label_rows,
    normalize_class_name_map,
    yolo_row_to_detection,
)


def evaluate_yolo_label_predictions(
    *,
    gt_images_dir: str | Path,
    gt_labels_dir: str | Path,
    pred_labels_dir: str | Path,
    class_names: dict[int, str] | list[str],
    iou_threshold: float = 0.5,
    weight_reference: str | Path | None = None,
    model_imgsz: int = 640,
) -> dict[str, Any]:
    """YOLO txt 예측 결과를 GT와 비교해 검출 성능을 평가한다.

    이 함수는 `detect-and-reason` 쪽의 현재 워크플로우에 맞춰, 이미지와
    라벨 txt만 있으면 바로 평가할 수 있도록 구성했다. 동시에 결과 포맷은
    `tomato-detection-agentic`에서 저장하던 항목들과 최대한 비슷하게 맞춘다.
    """
    image_index = build_image_index(gt_images_dir)
    gt_labels_dir = Path(gt_labels_dir).resolve()
    pred_labels_dir = Path(pred_labels_dir).resolve()
    class_name_map = normalize_class_name_map(class_names)

    map_metric = _build_map_metric(class_metrics=True)
    class_agnostic_metric = _build_map_metric(class_metrics=False)
    per_image_rows: list[dict[str, Any]] = []

    total_gt = 0
    total_pred = 0
    matched_gt = 0
    correct_class_matches = 0

    per_class_gt = defaultdict(int)
    per_class_pred = defaultdict(int)
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    for stem, image_info in image_index.items():
        gt_rows = load_yolo_label_rows(gt_labels_dir / f"{stem}.txt")
        pred_rows = load_yolo_label_rows(pred_labels_dir / f"{stem}.txt")

        gt_labels, gt_boxes, _ = _rows_to_arrays(gt_rows, image_info.width, image_info.height)
        pred_labels, pred_boxes, pred_scores = _rows_to_arrays(pred_rows, image_info.width, image_info.height)

        predictions = _as_torch_predictions(pred_boxes, pred_scores, pred_labels)
        targets = _as_torch_targets(gt_boxes, gt_labels)
        map_metric.update([predictions], [targets])
        class_agnostic_metric.update([_as_class_agnostic_predictions(predictions)], [_as_class_agnostic_targets(targets)])

        gt_count = len(gt_labels)
        pred_count = len(pred_labels)
        total_gt += gt_count
        total_pred += pred_count

        for label in gt_labels:
            per_class_gt[int(label)] += 1
        for label in pred_labels:
            per_class_pred[int(label)] += 1

        matches, unmatched_gt, unmatched_pred = _match_boxes(gt_boxes, pred_boxes, iou_threshold)
        matched_gt += len(matches)

        for gt_idx, pred_idx, _ in matches:
            gt_label = int(gt_labels[gt_idx])
            pred_label = int(pred_labels[pred_idx])
            confusion[gt_label][pred_label] += 1

            if gt_label == pred_label:
                correct_class_matches += 1
                per_class_tp[gt_label] += 1
            else:
                # IoU는 충분하지만 클래스가 틀린 경우이므로,
                # 예측 클래스에는 FP, 정답 클래스에는 FN으로 기록한다.
                per_class_fp[pred_label] += 1
                per_class_fn[gt_label] += 1

        for gt_idx in unmatched_gt:
            per_class_fn[int(gt_labels[gt_idx])] += 1
        for pred_idx in unmatched_pred:
            per_class_fp[int(pred_labels[pred_idx])] += 1

        per_image_rows.append(
            {
                "stem": stem,
                "file_name": image_info.file_name,
                "num_gt": gt_count,
                "num_predictions": pred_count,
                "matched_detections": len(matches),
                "correct_class_matches": sum(
                    1 for gt_idx, pred_idx, _ in matches if int(gt_labels[gt_idx]) == int(pred_labels[pred_idx])
                ),
            }
        )

    map_results = _convert_metric_output(map_metric.compute(), class_name_map)
    ca_map_results = _convert_ca_metric_output(class_agnostic_metric.compute())
    model_complexity = estimate_ultralytics_model_complexity(weight_reference, imgsz=model_imgsz)

    overall_precision = correct_class_matches / total_pred if total_pred else 0.0
    overall_recall = correct_class_matches / total_gt if total_gt else 0.0
    overall_f1 = _safe_f1(overall_precision, overall_recall)
    detection_acc = matched_gt / total_gt if total_gt else 0.0
    classification_acc = correct_class_matches / matched_gt if matched_gt else 0.0
    overall_acc = correct_class_matches / total_gt if total_gt else 0.0

    per_class_rows = []
    class_statistics = {
        "class_ids": [],
        "class_names": [],
        "class_ground_truths": [],
        "class_predictions": [],
        "class_tp": [],
        "class_fp": [],
        "class_fn": [],
        "class_precision": [],
        "class_recall": [],
        "class_f1": [],
    }
    for class_id, class_name in class_name_map.items():
        tp = per_class_tp[class_id]
        fp = per_class_fp[class_id]
        fn = per_class_fn[class_id]
        class_precision = tp / (tp + fp) if (tp + fp) else 0.0
        class_recall = tp / (tp + fn) if (tp + fn) else 0.0
        class_f1 = _safe_f1(class_precision, class_recall)
        class_ap = map_results["per_class_map_by_id"].get(class_id)

        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "num_gt": per_class_gt[class_id],
                "num_predictions": per_class_pred[class_id],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": class_precision,
                "recall": class_recall,
                "f1": class_f1,
                "ap_50_95": class_ap,
            }
        )

        class_statistics["class_ids"].append(class_id)
        class_statistics["class_names"].append(class_name)
        class_statistics["class_ground_truths"].append(per_class_gt[class_id])
        class_statistics["class_predictions"].append(per_class_pred[class_id])
        class_statistics["class_tp"].append(tp)
        class_statistics["class_fp"].append(fp)
        class_statistics["class_fn"].append(fn)
        class_statistics["class_precision"].append(class_precision)
        class_statistics["class_recall"].append(class_recall)
        class_statistics["class_f1"].append(class_f1)

    detection_metrics = {
        **map_results["summary"],
        "classes": list(class_name_map.keys()),
        "map_per_class": map_results["map_per_class"],
        "ca_map": ca_map_results.get("ca_map"),
        "ca_map_50": ca_map_results.get("ca_map_50"),
        "ca_map_75": ca_map_results.get("ca_map_75"),
        "mar_100": map_results["summary"].get("mar_100"),
    }

    total_statistics = {
        "num_images": len(image_index),
        "total_ground_truths": total_gt,
        "total_predictions": total_pred,
        "matched_detections": matched_gt,
        "correct_class_matches": correct_class_matches,
        "iou_threshold": float(iou_threshold),
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "detection_acc": detection_acc,
        "classification_acc": classification_acc,
        "overall_acc": overall_acc,
    }

    return {
        "evaluation_info": {
            "gt_images_dir": str(Path(gt_images_dir).resolve()),
            "gt_labels_dir": str(gt_labels_dir),
            "pred_labels_dir": str(pred_labels_dir),
            "weight_reference": str(weight_reference) if weight_reference is not None else None,
            "model_imgsz": int(model_imgsz),
        },
        "detection_metrics": detection_metrics,
        "detailed_statistics": {
            "total_statistics": total_statistics,
            "class_statistics": class_statistics,
        },
        "model_complexity": model_complexity,
        "per_class": per_class_rows,
        "per_image": per_image_rows,
        "confusion": _materialize_confusion(confusion, class_name_map),
        "paths": {
            "gt_images_dir": str(Path(gt_images_dir).resolve()),
            "gt_labels_dir": str(gt_labels_dir),
            "pred_labels_dir": str(pred_labels_dir),
        },
        # 기존 스크립트 호환을 위해 이전 키도 함께 남긴다.
        "summary": {
            "num_images": len(image_index),
            "total_ground_truths": total_gt,
            "total_predictions": total_pred,
            "matched_detections": matched_gt,
            "correct_class_matches": correct_class_matches,
            "iou_threshold": float(iou_threshold),
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "detection_acc": detection_acc,
            "classification_acc": classification_acc,
            "overall_acc": overall_acc,
        },
        "map_metrics": {
            "map": detection_metrics["map"],
            "map_50": detection_metrics["map_50"],
            "map_75": detection_metrics["map_75"],
            "mar_100": detection_metrics["mar_100"],
            "ca_map": detection_metrics["ca_map"],
            "ca_map_50": detection_metrics["ca_map_50"],
            "ca_map_75": detection_metrics["ca_map_75"],
        },
    }


def evaluate_yolo_txt_predictions(**kwargs: Any) -> dict[str, Any]:
    """기존 함수명을 쓰던 코드도 계속 동작하도록 남겨 둔 호환 alias다."""
    return evaluate_yolo_label_predictions(**kwargs)


def _build_map_metric(*, class_metrics: bool):
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchmetrics is required for evaluation. Install project dependencies first."
        ) from exc

    return MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=class_metrics)


def _rows_to_arrays(
    rows: list[list[float]],
    width: int,
    height: int,
) -> tuple[list[int], list[list[float]], list[float]]:
    """YOLO row들을 label/box/score 배열로 나눈다."""
    labels: list[int] = []
    boxes: list[list[float]] = []
    scores: list[float] = []

    for row in rows:
        label, box, score = yolo_row_to_detection(row, width, height)
        labels.append(label)
        boxes.append(box)
        scores.append(score)
    return labels, boxes, scores


def _as_torch_predictions(boxes: list[list[float]], scores: list[float], labels: list[int]) -> dict[str, torch.Tensor]:
    """torchmetrics가 기대하는 prediction 포맷으로 변환한다."""
    return {
        "boxes": _tensor_boxes(boxes),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _as_torch_targets(boxes: list[list[float]], labels: list[int]) -> dict[str, torch.Tensor]:
    """torchmetrics가 기대하는 target 포맷으로 변환한다."""
    return {
        "boxes": _tensor_boxes(boxes),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _as_class_agnostic_predictions(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """클래스를 모두 0으로 통일해 class-agnostic mAP를 계산한다."""
    return {
        "boxes": predictions["boxes"],
        "scores": predictions["scores"],
        "labels": torch.zeros_like(predictions["labels"]),
    }


def _as_class_agnostic_targets(targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """정답도 동일하게 class-agnostic 형태로 바꾼다."""
    return {
        "boxes": targets["boxes"],
        "labels": torch.zeros_like(targets["labels"]),
    }


def _tensor_boxes(boxes: list[list[float]]) -> torch.Tensor:
    """빈 박스도 안전하게 처리할 수 있는 텐서 형태로 만든다."""
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def _match_boxes(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """IoU 기준으로 GT와 예측 박스를 1:1 greedy 매칭한다."""
    if not gt_boxes or not pred_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    candidate_pairs: list[tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            iou = _box_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, gt_idx, pred_idx))

    # IoU가 큰 매칭부터 확정해야, 한 GT에 여러 예측이 몰릴 때 가장 타당한
    # 쌍을 먼저 잡아낼 수 있다.
    candidate_pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, gt_idx, pred_idx in candidate_pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in used_gt]
    unmatched_pred = [idx for idx in range(len(pred_boxes)) if idx not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """두 개의 xyxy 박스 IoU를 직접 계산한다."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _safe_f1(precision: float, recall: float) -> float:
    """precision/recall이 둘 다 0인 경우를 안전하게 처리한다."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _convert_metric_output(metric_output: dict[str, Any], class_name_map: dict[int, str]) -> dict[str, Any]:
    """torchmetrics 출력을 저장하기 쉬운 파이썬 기본 타입으로 정리한다."""
    classes = _to_python(metric_output.get("classes", []))
    map_per_class = _to_python(metric_output.get("map_per_class", []))

    per_class_map_by_id: dict[int, float | None] = {class_id: None for class_id in class_name_map}
    if isinstance(classes, list) and isinstance(map_per_class, list):
        for class_id, class_map in zip(classes, map_per_class):
            per_class_map_by_id[int(class_id)] = float(class_map) if class_map is not None else None

    return {
        "summary": {
            "map": _float_or_none(metric_output.get("map")),
            "map_50": _float_or_none(metric_output.get("map_50")),
            "map_75": _float_or_none(metric_output.get("map_75")),
            "mar_100": _float_or_none(metric_output.get("mar_100")),
        },
        "map_per_class": [per_class_map_by_id[class_id] for class_id in class_name_map],
        "per_class_map_by_id": per_class_map_by_id,
    }


def _convert_ca_metric_output(metric_output: dict[str, Any]) -> dict[str, float | None]:
    """class-agnostic mAP 결과만 별도로 추려낸다."""
    return {
        "ca_map": _float_or_none(metric_output.get("map")),
        "ca_map_50": _float_or_none(metric_output.get("map_50")),
        "ca_map_75": _float_or_none(metric_output.get("map_75")),
    }


def _materialize_confusion(
    confusion: dict[int, dict[int, int]],
    class_name_map: dict[int, str],
) -> dict[str, dict[str, int]]:
    """숫자 class id 기반 confusion 정보를 사람이 읽기 좋은 이름으로 바꾼다."""
    out: dict[str, dict[str, int]] = {}
    for gt_id in class_name_map:
        gt_name = class_name_map.get(gt_id, str(gt_id))
        out[gt_name] = {}
        for pred_id in class_name_map:
            pred_name = class_name_map.get(pred_id, str(pred_id))
            out[gt_name][pred_name] = int(confusion.get(gt_id, {}).get(pred_id, 0))
    return out


def _float_or_none(value: Any) -> float | None:
    """텐서/숫자/None 입력을 float 또는 None으로 정리한다."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_python(value: Any) -> Any:
    """torch Tensor 출력을 기본 파이썬 타입으로 풀어낸다."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
