"""llm-classify(외부 SoM VLM 저장소 `som_common.py`의 match_det/det_score)와 동일한
방식으로 detection을 채점하는 core.

기존 DaR 지표(detection_acc/classification_acc/overall_acc, yolo_prediction_evaluator.py)와
다르게:
- 매칭 자체가 클래스까지 같아야 성립하고(class-conditioned greedy matching, IoU만으로 먼저
  매칭한 뒤 클래스를 보는 DaR과 반대),
- 놓친 GT는 "miss", 남는 예측은 "none"으로 pairs에 강제 편입되어
- false positive가 정확도(acc)에 직접 페널티를 준다 (DaR의 overall_acc는 분모가
  total_gt 고정이라 FP를 못 봄).

`evaluate_detection_samples`(yolo_prediction_evaluator.py)가 이 모듈을 가져다 써서,
YOLO/COCO 어느 입력이든 DaR 지표와 llm-classify 지표를 한 번에 같이 낸다.
"""

from __future__ import annotations


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """xyxy 픽셀 좌표 두 박스의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def match_det(
    gts: list[tuple[tuple, str]],
    preds: list[tuple[tuple, str, float]],
    thr: float = 0.5,
) -> list[tuple[str, str]]:
    """클래스-조건부 greedy 매칭. 매칭 자체가 클래스까지 같아야 성립한다.

    매칭 실패한 GT는 ("gt_cls","miss")로, 매칭 실패한 예측은 ("none","pred_cls")로
    pairs에 강제 편입된다.
    """
    preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
    used: set[int] = set()
    matched: set[int] = set()
    pairs: list[tuple[str, str]] = []
    fp: list[str] = []

    for box, cls, _score in preds_sorted:
        best, biou = -1, thr
        for i, (gt_box, gt_cls) in enumerate(gts):
            if i in used or gt_cls != cls:
                continue
            v = iou_xyxy(box, gt_box)
            if v > biou:
                best, biou = i, v
        if best >= 0:
            used.add(best)
            matched.add(best)
            pairs.append((gts[best][1], cls))
        else:
            fp.append(cls)

    for i, (_, gt_cls) in enumerate(gts):
        if i not in matched:
            pairs.append((gt_cls, "miss"))
    for cls in fp:
        pairs.append(("none", cls))

    return pairs


def det_score(pairs: list[tuple[str, str]], class_list: list[str]) -> dict:
    """클래스별 P/R/F1 + macro-F1 + 전체 acc(FP/miss 포함)."""
    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    for c in class_list:
        tp = sum(g == p == c for g, p in pairs)
        fp = sum(p == c and g != c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "n_gt": sum(g == c for g, _ in pairs)}
        f1s.append(f1)

    acc = sum(g == p for g, p in pairs) / max(1, len(pairs))
    macro_f1 = sum(f1s) / len(class_list) if class_list else 0.0

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "n_pairs": len(pairs),
    }


def boxes_to_llm_classify_pairs(
    gt_boxes: list[list[float]],
    gt_labels: list[int],
    pred_boxes: list[list[float]],
    pred_labels: list[int],
    pred_scores: list[float],
    class_name_map: dict[int, str],
    iou_threshold: float,
) -> list[tuple[str, str]]:
    """DetectionSample 한 장 분의 xyxy 박스+정수 라벨을 match_det 입력으로 바꿔 pairs를 낸다.

    GT에 없는 label(모델이 가끔 뱉는 잉여 클래스)은 어떤 GT와도 매칭될 수 없는 별도
    라벨로 취급한다 — 매칭 실패 → 그대로 FP("none"->cls)로 집계된다.
    """
    gts = [(tuple(box), class_name_map[label]) for box, label in zip(gt_boxes, gt_labels)]
    preds = [
        (tuple(box), class_name_map.get(label, f"unknown_label_{label}"), score)
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores)
    ]
    return match_det(gts, preds, thr=iou_threshold)
