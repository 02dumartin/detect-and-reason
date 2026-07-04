"""저conf(예: 0.001) 예측을 받아 mAP(전체 박스)와 best-F1 동작점 지표를 함께 낸다.

- mAP / mAP50 / mAP75 : confidence 전 구간(전체 예측)으로 계산 (torchmetrics, 표준).
- precision/recall/F1/Acc : conf threshold 를 sweep 해 F1 최대 지점을 골라 그 값을 보고.
  (NMS 없는 RT-DETR 처럼 과다예측하는 모델을 고정 conf 로 비교할 때의 불공정을 제거)

YOLO(yolo11/12/yolo_world/rt_detr) 와 COCO(rf_detr/dino) 예측 모두 처리한다.

Usage:
    python scripts/evaluate_bestf1.py --model yolo11_3cls --dataset big --split test \
        --output-dir benchmark/bestf1_3cls/yolo11_3cls__big
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import autorootcwd  # noqa: F401  (src 임포트 경로 확보)
import yaml

from src.config_loader import resolve_runtime_config
from src.evaluation.yolo_label_io import (
    build_image_index,
    resolve_prediction_labels_dir,
    resolve_yolo_split_dirs,
)
from src.evaluation.yolo_prediction_evaluator import (
    _coco_samples,
    _match_boxes,
    _yolo_samples,
    evaluate_detection_samples,
)

_SPLIT_ALIASES = {"val": "valid", "validation": "valid"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mAP(all) + best-F1 operating point evaluation")
    p.add_argument("--model", type=str)
    p.add_argument("--model-config", type=str)
    p.add_argument("--dataset", type=str)
    p.add_argument("--dataset-config", type=str)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                   help="최종 지표를 보고할 split (기본 test)")
    p.add_argument("--select-split", type=str, default=None, choices=["train", "val", "test"],
                   help="best-F1 conf 를 고를 split (권장 val). 생략 시 --split 에서 골라 test-tuning 이 되므로 주의.")
    p.add_argument("--report-pred", type=str, help="보고 split 예측 경로 override (yolo labels dir / coco json)")
    p.add_argument("--select-pred", type=str, help="선택 split 예측 경로 override (yolo labels dir / coco json)")
    p.add_argument("--output-dir", type=str, help="결과 json 저장 폴더")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--sweep-step", type=float, default=0.01, help="conf sweep 간격")
    p.add_argument("--sweep-min", type=float, default=0.01)
    p.add_argument("--sweep-max", type=float, default=0.95)
    return p.parse_args()


def _class_names(runtime_cfg: dict) -> dict[int, str]:
    mc = runtime_cfg.get("model_config_path")
    try:
        cn = (yaml.safe_load(open(mc)) or {}).get("class_names")
        if isinstance(cn, dict):
            return {int(k): v for k, v in cn.items()}
    except Exception:
        pass
    return {}


def _build_samples(runtime_cfg: dict, split: str, pred_override: str | None = None):
    family = runtime_cfg.get("family", "")
    if family in ("rf_detr", "dino"):
        from src.evaluation.coco_label_io import load_coco_gt, load_coco_predictions

        rfdetr_dir = Path(runtime_cfg["dataset"]["rfdetr_dir"])
        cs = _SPLIT_ALIASES.get(split, split)
        gt_ann = rfdetr_dir / cs / "_annotations.coco.json"
        pred_coco = Path(pred_override) if pred_override else (
            Path(runtime_cfg["paths"]["prediction_dir"]) / cs / "predictions_coco.json"
        )
        gt_by_image, _ = load_coco_gt(gt_ann)
        pred_by_image = load_coco_predictions(pred_coco, conf_threshold=0.0)
        return list(_coco_samples(gt_by_image, pred_by_image)), str(pred_coco)
    # yolo family
    split_path = runtime_cfg["dataset"][f"{split}_dir"]
    gt_images_dir, gt_labels_dir = resolve_yolo_split_dirs(split_path)
    pred_dir = resolve_prediction_labels_dir(pred_override or runtime_cfg["paths"]["prediction_dir"])
    samples = list(_yolo_samples(build_image_index(gt_images_dir), Path(gt_labels_dir), Path(pred_dir)))
    return samples, str(pred_dir)


def _oppoint(samples, t: float, iou: float) -> dict:
    """conf>=t 로 거른 예측에 대한 동작점 지표(P/R/F1/Loc/Cls/Overall)."""
    tg = tp = matched = correct = 0
    for s in samples:
        pb, pl = [], []
        for b, l, sc in zip(s.pred_boxes, s.pred_labels, s.pred_scores):
            if sc >= t:
                pb.append(b)
                pl.append(l)
        tg += len(s.gt_boxes)
        tp += len(pb)
        m, _, _ = _match_boxes(s.gt_boxes, pb, iou)
        matched += len(m)
        for gi, pi, _iou in m:
            if s.gt_labels[gi] == pl[pi]:
                correct += 1
    precision = correct / tp if tp else 0.0
    recall = correct / tg if tg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "conf_threshold": round(t, 4),
        "total_ground_truths": tg,
        "total_predictions": tp,
        "detection_acc": matched / tg if tg else 0.0,      # Loc
        "classification_acc": correct / matched if matched else 0.0,
        "overall_acc": correct / tg if tg else 0.0,
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
    }


def main() -> None:
    args = parse_args()
    runtime_cfg = resolve_runtime_config(
        model_ref=args.model_config or args.model,
        dataset_ref=args.dataset_config or args.dataset,
        stage="predict",
    )
    class_names = _class_names(runtime_cfg)

    # 보고(report) split = test : mAP 와 최종 지표를 여기서 낸다.
    report_samples, pred_ref = _build_samples(runtime_cfg, args.split, args.report_pred)

    # 1) mAP: report split 전체 박스 (torchmetrics, conf 무관)
    full = evaluate_detection_samples(
        samples=report_samples, class_names=class_names, iou_threshold=args.iou_threshold
    )
    dm = full["detection_metrics"]

    # 2) conf* 선택: --select-split(권장 val)에서 F1 최대 conf 를 고른다.
    #    (test 에서 고르면 test-tuning 이 되므로 val 사용을 권장)
    n = int(round((args.sweep_max - args.sweep_min) / args.sweep_step)) + 1
    grid = [round(args.sweep_min + i * args.sweep_step, 4) for i in range(n)]

    if args.select_split and args.select_split != args.split:
        sel_samples, sel_ref = _build_samples(runtime_cfg, args.select_split, args.select_pred)
        sel_sweep = [_oppoint(sel_samples, t, args.iou_threshold) for t in grid]
        sel_best = max(sel_sweep, key=lambda r: r["overall_f1"])
        conf_star = sel_best["conf_threshold"]
        selection = {"split": args.select_split, "pred_reference": sel_ref,
                     "conf_threshold": conf_star, "f1_on_select": round(sel_best["overall_f1"], 4)}
    else:
        # fallback: report split 에서 선택 (test-tuning; 경고)
        sweep = [_oppoint(report_samples, t, args.iou_threshold) for t in grid]
        conf_star = max(sweep, key=lambda r: r["overall_f1"])["conf_threshold"]
        selection = {"split": args.split, "pred_reference": pred_ref,
                     "conf_threshold": conf_star, "warning": "conf selected on report split (test-tuning)"}

    # 3) conf* 를 report split(test)에 적용한 동작점 지표
    op = _oppoint(report_samples, conf_star, args.iou_threshold)

    result = {
        "evaluation_info": {
            "model_name": runtime_cfg["model_name"],
            "dataset_key": runtime_cfg["dataset"]["key"],
            "split": args.split,
            "family": runtime_cfg.get("family"),
            "iou_threshold": args.iou_threshold,
            "pred_reference": pred_ref,
            "weight_reference": str(runtime_cfg["paths"]["weights"]),
        },
        "map_metrics": {
            "map": dm.get("map"),
            "map_50": dm.get("map_50"),
            "map_75": dm.get("map_75"),
        },
        "conf_selection": selection,
        "operating_point": op,          # conf*(=val선택) 를 test 에 적용한 값
    }

    print(f"[bestf1] {runtime_cfg['model_name']} x {runtime_cfg['dataset']['key']}")
    print(f"[bestf1] mAP50={dm.get('map_50'):.4f} mAP={dm.get('map'):.4f}  "
          f"conf* from {selection['split']}={conf_star}")
    print(
        f"[bestf1] test@conf* F1={op['overall_f1']:.4f} "
        f"P={op['overall_precision']:.4f} R={op['overall_recall']:.4f} "
        f"Loc={op['detection_acc']:.4f} Cls={op['classification_acc']:.4f} Overall={op['overall_acc']:.4f} "
        f"(pred={op['total_predictions']}/gt={op['total_ground_truths']})"
    )

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bestf1_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[bestf1] saved={out / 'bestf1_results.json'}")


if __name__ == "__main__":
    main()
