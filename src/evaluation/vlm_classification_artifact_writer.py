from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_vlm_classification_artifacts(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    summary_csv_path = output_dir / "summary_metrics.csv"
    per_class_csv_path = output_dir / "per_class_metrics.csv"
    per_image_csv_path = output_dir / "per_image_metrics.csv"
    confusion_csv_path = output_dir / "confusion_matrix.csv"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(_build_summary_rows(results)).to_csv(summary_csv_path, index=False)
    pd.DataFrame(results.get("per_class", [])).to_csv(per_class_csv_path, index=False)
    pd.DataFrame(results.get("per_image", [])).to_csv(per_image_csv_path, index=False)
    pd.DataFrame.from_dict(results.get("confusion", {}), orient="index").fillna(0).to_csv(confusion_csv_path)

    return {
        "json": json_path,
        "summary_csv": summary_csv_path,
        "per_class_csv": per_class_csv_path,
        "per_image_csv": per_image_csv_path,
        "confusion_csv": confusion_csv_path,
    }


def _build_summary_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = results.get("metrics", {})
    rows = []
    for key in (
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "detection_acc",
        "classification_acc",
        "overall_acc",
        "matched_detections",
        "correct_class_matches",
        "total_ground_truths",
        "total_predictions",
        "num_images",
        "iou_threshold",
    ):
        rows.append({"Metric": key, "Value": metrics.get(key)})
    return rows

