from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.yolo_label_io import (
    build_image_index,
    load_yolo_label_rows,
    normalize_class_name_map,
    yolo_row_to_detection,
)
from src.vlm.schema import CANONICAL_CLASS_ID_TO_NAME


def evaluate_vlm_classification_predictions(
    *,
    gt_images_dir: str | Path,
    gt_labels_dir: str | Path,
    predictions: list[dict[str, Any]],
    class_names: dict[int, str] | list[str] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    class_name_map = normalize_class_name_map(class_names or CANONICAL_CLASS_ID_TO_NAME)
    image_index = build_image_index(gt_images_dir)
    pred_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        pred_by_stem[Path(str(prediction["file_name"])).stem].append(prediction)

    total_gt = 0
    total_pred = 0
    matched_gt = 0
    correct_class_matches = 0

    per_class_gt = defaultdict(int)
    per_class_pred = defaultdict(int)
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_image_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for stem, image_info in image_index.items():
        gt_rows = load_yolo_label_rows(Path(gt_labels_dir) / f"{stem}.txt")
        gt_labels: list[int] = []
        gt_boxes: list[list[float]] = []
        for row in gt_rows:
            label, box, _ = yolo_row_to_detection(row, image_info.width, image_info.height)
            gt_labels.append(int(label))
            gt_boxes.append(box)

        pred_rows = pred_by_stem.get(stem, [])
        pred_labels = [int(row.get("predicted_class_id", -1)) for row in pred_rows]
        pred_boxes = [[float(value) for value in row["bbox_xyxy"]] for row in pred_rows]

        total_gt += len(gt_labels)
        total_pred += len(pred_labels)
        for label in gt_labels:
            per_class_gt[int(label)] += 1
        for label in pred_labels:
            if label >= 0:
                per_class_pred[int(label)] += 1

        matches, unmatched_gt, unmatched_pred = _match_boxes(gt_boxes, pred_boxes, iou_threshold)
        matched_gt += len(matches)
        image_correct = 0

        for gt_idx, pred_idx, iou in matches:
            gt_label = int(gt_labels[gt_idx])
            pred_label = int(pred_labels[pred_idx])
            pred_row = pred_rows[pred_idx]
            pred_name = class_name_map.get(pred_label, "none")
            gt_name = class_name_map.get(gt_label, str(gt_label))
            confusion[gt_name][pred_name] += 1

            if gt_label == pred_label:
                correct_class_matches += 1
                image_correct += 1
                per_class_tp[gt_label] += 1
            else:
                per_class_fn[gt_label] += 1
                if pred_label >= 0:
                    per_class_fp[pred_label] += 1

            eval_rows.append(
                {
                    "row_type": "match",
                    "stem": stem,
                    "file_name": image_info.file_name,
                    "gt_class_id": gt_label,
                    "gt_class_name": gt_name,
                    "pred_class_id": pred_label,
                    "pred_class_name": pred_name,
                    "iou": iou,
                    "detection_correct": True,
                    "classification_correct": gt_label == pred_label,
                    "det_score": float(pred_row.get("det_score", 1.0)),
                    "reasoning": str(pred_row.get("reasoning") or ""),
                    "raw_response": str(pred_row.get("raw_response") or ""),
                    "record_id": pred_row.get("record_id"),
                    "parse_error": pred_row.get("parse_error"),
                }
            )

        for gt_idx in unmatched_gt:
            gt_label = int(gt_labels[gt_idx])
            gt_name = class_name_map.get(gt_label, str(gt_label))
            per_class_fn[gt_label] += 1
            confusion[gt_name]["none"] += 1
            eval_rows.append(
                {
                    "row_type": "miss",
                    "stem": stem,
                    "file_name": image_info.file_name,
                    "gt_class_id": gt_label,
                    "gt_class_name": gt_name,
                    "pred_class_id": -1,
                    "pred_class_name": "none",
                    "iou": 0.0,
                    "detection_correct": False,
                    "classification_correct": False,
                    "det_score": None,
                    "reasoning": "",
                    "raw_response": "",
                    "record_id": None,
                    "parse_error": None,
                }
            )

        for pred_idx in unmatched_pred:
            pred_row = pred_rows[pred_idx]
            pred_label = int(pred_labels[pred_idx])
            pred_name = class_name_map.get(pred_label, "none")
            if pred_label >= 0:
                per_class_fp[pred_label] += 1
            confusion["none"][pred_name] += 1
            eval_rows.append(
                {
                    "row_type": "false_positive",
                    "stem": stem,
                    "file_name": image_info.file_name,
                    "gt_class_id": -1,
                    "gt_class_name": "none",
                    "pred_class_id": pred_label,
                    "pred_class_name": pred_name,
                    "iou": 0.0,
                    "detection_correct": False,
                    "classification_correct": False,
                    "det_score": float(pred_row.get("det_score", 1.0)),
                    "reasoning": str(pred_row.get("reasoning") or ""),
                    "raw_response": str(pred_row.get("raw_response") or ""),
                    "record_id": pred_row.get("record_id"),
                    "parse_error": pred_row.get("parse_error"),
                }
            )

        gt_count = len(gt_labels)
        pred_count = len(pred_labels)
        matched_count = len(matches)
        per_image_rows.append(
            {
                "stem": stem,
                "file_name": image_info.file_name,
                "num_gt": gt_count,
                "num_predictions": pred_count,
                "matched_detections": matched_count,
                "correct_class_matches": image_correct,
                "precision_acc_pct": (image_correct / pred_count * 100.0) if pred_count else 0.0,
                "classification_acc_pct": (image_correct / matched_count * 100.0) if matched_count else 0.0,
                "overall_acc_pct": (image_correct / gt_count * 100.0) if gt_count else 0.0,
            }
        )

    overall_precision = correct_class_matches / total_pred if total_pred else 0.0
    overall_recall = correct_class_matches / total_gt if total_gt else 0.0
    overall_f1 = _safe_f1(overall_precision, overall_recall)
    detection_acc = matched_gt / total_gt if total_gt else 0.0
    classification_acc = correct_class_matches / matched_gt if matched_gt else 0.0
    overall_acc = correct_class_matches / total_gt if total_gt else 0.0

    per_class_rows = []
    for class_id, class_name in class_name_map.items():
        tp = per_class_tp[class_id]
        fp = per_class_fp[class_id]
        fn = per_class_fn[class_id]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "num_gt": per_class_gt[class_id],
                "num_predictions": per_class_pred[class_id],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": _safe_f1(precision, recall),
            }
        )

    metrics = {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "detection_acc": detection_acc,
        "classification_acc": classification_acc,
        "overall_acc": overall_acc,
        "matched_detections": matched_gt,
        "correct_class_matches": correct_class_matches,
        "total_ground_truths": total_gt,
        "total_predictions": total_pred,
        "num_images": len(image_index),
        "iou_threshold": float(iou_threshold),
    }

    return {
        "metrics": metrics,
        "per_class": per_class_rows,
        "per_image": per_image_rows,
        "confusion": {row_name: dict(columns) for row_name, columns in confusion.items()},
        "eval_rows": eval_rows,
    }


def _match_boxes(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not gt_boxes or not pred_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    candidate_pairs: list[tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            iou = _box_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, gt_idx, pred_idx))

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
    return inter_area / union if union > 0 else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
