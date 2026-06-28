from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_detection_evaluation_artifacts(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """평가 결과를 JSON/CSV 형태로 저장한다.

    저장 포맷은 `tomato-detection-agentic`의 summary CSV 구성에 최대한 맞추되,
    현재 프로젝트에서 실무적으로 자주 보는 per-class / per-image / confusion
    테이블도 같이 남긴다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_results.json"
    summary_csv_path = output_dir / "summary_metrics.csv"
    per_class_csv_path = output_dir / "per_class_metrics.csv"
    per_image_csv_path = output_dir / "per_image_metrics.csv"
    confusion_csv_path = output_dir / "confusion_matrix.csv"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    summary_frame = pd.DataFrame(_build_summary_rows(results))
    summary_frame.to_csv(summary_csv_path, index=False)

    pd.DataFrame(results.get("per_class", [])).to_csv(per_class_csv_path, index=False)
    pd.DataFrame(results.get("per_image", [])).to_csv(per_image_csv_path, index=False)
    _confusion_to_frame(results.get("confusion", {})).to_csv(confusion_csv_path)

    return {
        "json": json_path,
        "summary_csv": summary_csv_path,
        "per_class_csv": per_class_csv_path,
        "per_image_csv": per_image_csv_path,
        "confusion_csv": confusion_csv_path,
    }


def save_evaluation_artifacts(results: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """기존 함수명을 유지하기 위한 호환 alias다."""
    return save_detection_evaluation_artifacts(results, output_dir)


def _build_summary_rows(results: dict[str, Any]) -> list[dict[str, str]]:
    """참조 프로젝트와 비슷한 2열 summary CSV를 만든다."""
    detection_metrics = results.get("detection_metrics", {})
    detailed_statistics = results.get("detailed_statistics", {})
    total_statistics = detailed_statistics.get("total_statistics", {})
    model_complexity = results.get("model_complexity", {})
    per_class = results.get("per_class", [])

    rows: list[dict[str, str]] = [{"Metric": "mAP@0.5", "Value": _fmt(detection_metrics.get("map_50"))}]

    for row in per_class:
        class_name = row.get("class_name", "unknown")
        rows.append(
            {
                "Metric": f"{class_name}_AP@0.5:0.95",
                "Value": _fmt(row.get("ap_50_95")),
            }
        )

    rows.extend(
        [
            {"Metric": "mAP@0.5:0.95", "Value": _fmt(detection_metrics.get("map"))},
            {"Metric": "mAP@0.75", "Value": _fmt(detection_metrics.get("map_75"))},
            {"Metric": "CA-mAP@0.5", "Value": _fmt(detection_metrics.get("ca_map_50"))},
            {"Metric": "CA-mAP@0.5:0.95", "Value": _fmt(detection_metrics.get("ca_map"))},
            {"Metric": "CA-mAP@0.75", "Value": _fmt(detection_metrics.get("ca_map_75"))},
            {"Metric": "Precision", "Value": _fmt(total_statistics.get("overall_precision"))},
            {"Metric": "Recall", "Value": _fmt(total_statistics.get("overall_recall"))},
            {"Metric": "Parameter(M)", "Value": _fmt(model_complexity.get("params_m"), digits=2)},
            {"Metric": "GFLOPs", "Value": _fmt(model_complexity.get("gflops"), digits=2)},
        ]
    )

    return rows


def _fmt(value: Any, *, digits: int = 4) -> str:
    """숫자는 고정 소수점 문자열로, 비어 있으면 `N/A`로 표현한다."""
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _confusion_to_frame(confusion: dict[str, dict[str, int]]) -> pd.DataFrame:
    """중첩 dict 형태 confusion을 CSV로 쓰기 쉬운 표 형태로 바꾼다."""
    if not confusion:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(confusion, orient="index").fillna(0).astype(int)
