"""llm-classify 스타일(FP까지 acc에 반영하는) 평가 결과를
baseline VLM ablation(`vlm_classification_artifact_writer.save_vlm_classification_artifacts`)과
같은 디렉토리 구조(evaluation_results.json + summary/per_class/per_image csv +
confusion_matrix.csv)로 남긴다. SoM 실험(`src/llm-classify/*_som.py`)이 요약 json
한 장만 쓰던 것을 다른 실험들과 비교 가능한 포맷으로 맞추기 위한 것.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_llm_classify_artifacts(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    summary_csv_path = output_dir / "summary_metrics.csv"
    per_class_csv_path = output_dir / "per_class_metrics.csv"
    per_image_csv_path = output_dir / "per_image_metrics.csv"
    confusion_csv_path = output_dir / "confusion_matrix.csv"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame(_build_summary_rows(results)).to_csv(summary_csv_path, index=False)
    pd.DataFrame(_build_per_class_rows(results)).to_csv(per_class_csv_path, index=False)
    pd.DataFrame(results.get("per_image", [])).to_csv(per_image_csv_path, index=False)
    pd.DataFrame.from_dict(results.get("confusion", {}), orient="index").fillna(0).astype(int).to_csv(
        confusion_csv_path
    )

    return {
        "json": json_path,
        "summary_csv": summary_csv_path,
        "per_class_csv": per_class_csv_path,
        "per_image_csv": per_image_csv_path,
        "confusion_csv": confusion_csv_path,
    }


def _build_summary_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"Metric": "acc", "Value": results.get("acc")},
        {"Metric": "macro_f1", "Value": results.get("macro_f1")},
        {"Metric": "n_pairs", "Value": results.get("n_pairs")},
        {"Metric": "num_images", "Value": results.get("num_images")},
        {"Metric": "iou_threshold", "Value": results.get("iou_threshold")},
        {"Metric": "dataset", "Value": results.get("dataset")},
        {"Metric": "proposal_source", "Value": results.get("proposal_source")},
    ]
    # DaR 논문 표(Table I/VIII)의 Loc.Acc/Cls.Acc/Overall Acc과 바로 비교하기 위한 값.
    # recall형(분모가 GT 개수 고정)이라 FP를 못 보는 지표라서, 위 acc/macro_f1(FP까지
    # 반영하는 llm-classify 스타일)을 대체하는 게 아니라 나란히 남기는 것이다.
    for key in ("loc_acc", "cls_acc", "overall_acc"):
        if key in results:
            rows.append({"Metric": key, "Value": results[key]})
    return rows


def _build_per_class_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"class_name": cls, **stats} for cls, stats in results.get("per_class", {}).items()]


def save_llm_classify_artifacts_from_detection_results(
    results: dict[str, Any], output_dir: str | Path, *, dataset: str, model: str
) -> dict[str, Path]:
    """`evaluate_detection_samples`(YOLO/COCO 공용 detector 평가)가 낸 `results["llm_classify_style"]`을
    SoM 실험(`save_llm_classify_artifacts`)과 똑같은 evaluation_results.json/summary/per_class/
    per_image/confusion 5종 세트로 남긴다 — 두 파이프라인의 결과표를 서로 비교 가능하게 맞추는 것.
    """
    style = results["llm_classify_style"]
    total_statistics = results["detailed_statistics"]["total_statistics"]
    per_image = [
        {
            "stem": row["stem"],
            "n_gt": row["num_gt"],
            "n_pred": row["num_predictions"],
            "n_pairs": row["llm_classify_n_pairs"],
            "n_correct": row["llm_classify_n_correct"],
            "n_miss": row["llm_classify_n_miss"],
            "n_fp": row["llm_classify_n_fp"],
            "acc": row["llm_classify_acc_pct"] / 100.0,
        }
        for row in results.get("per_image", [])
    ]
    shaped = {
        "acc": style["acc"],
        "macro_f1": style["macro_f1"],
        "n_pairs": style["n_pairs"],
        "num_images": style["num_images"],
        "iou_threshold": style["iou_threshold"],
        "dataset": dataset,
        "proposal_source": model,
        "per_class": style["per_class"],
        "per_image": per_image,
        "confusion": style["confusion"],
        "loc_acc": total_statistics["detection_acc"],
        "cls_acc": total_statistics["classification_acc"],
        "overall_acc": total_statistics["overall_acc"],
    }
    return save_llm_classify_artifacts(shaped, output_dir)
