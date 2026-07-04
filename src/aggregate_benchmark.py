from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

"""
표의 공통 식별 컬럼. metric 컬럼은 summary_metrics.csv와 동일한 집합을
evaluation_results.json에서 동적으로 구성한다.

현재 benchmark_table의 metric 컬럼 기준:
    mAP@0.5
    fully ripened tomato_AP@0.5
    fully ripened tomato_AP@0.5:0.95
    half ripened tomato_AP@0.5
    half ripened tomato_AP@0.5:0.95
    green tomato_AP@0.5
    green tomato_AP@0.5:0.95
    mAP@0.5:0.95
    mAP@0.75
    CA-mAP@0.5
    CA-mAP@0.5:0.95
    CA-mAP@0.75
    Precision
    Recall
    F1 Score
    Detection Acc
    Classification Acc
    Overall Acc
    Parameter(M)
    GFLOPs
"""

BASE_COLUMNS: list[tuple[str, Any]] = [
    ("model", lambda r: r["evaluation_info"].get("model_name", "?")),
    ("dataset", lambda r: r["evaluation_info"].get("dataset_key", "?")),
]


def _load_summary_row_builder():
    """evaluation_artifact_writer에서 summary row 생성 함수를 직접 로드한다."""
    module_path = Path(__file__).resolve().parent / "evaluation" / "evaluation_artifact_writer.py"
    spec = importlib.util.spec_from_file_location("aggregate_eval_writer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluation artifact writer from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_summary_rows


_build_summary_rows = _load_summary_row_builder()


def _summary_metric_map(results: dict[str, Any]) -> dict[str, str]:
    """summary_metrics.csv와 동일한 Metric/Value 쌍을 dict로 정리한다."""
    return {row["Metric"]: row["Value"] for row in _build_summary_rows(results)}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def collect(eval_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """eval_dir 아래 evaluation_results.json을 읽어 행/헤더를 만든다."""
    rows: list[dict[str, Any]] = []
    metric_headers: list[str] = []
    seen_metric_headers: set[str] = set()

    for json_path in sorted(eval_dir.rglob("evaluation_results.json")):
        try:
            results = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        row = {header: getter(results) for header, getter in BASE_COLUMNS}
        for metric_name, metric_value in _summary_metric_map(results).items():
            row[metric_name] = metric_value
            if metric_name not in seen_metric_headers:
                seen_metric_headers.add(metric_name)
                metric_headers.append(metric_name)

        rows.append(row)

    # dataset → model 순으로 정렬해 표를 읽기 좋게 만든다.
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["model"])))
    headers = [header for header, _ in BASE_COLUMNS] + metric_headers
    return rows, headers


def to_markdown(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [_fmt(row.get(header)) for header in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows: list[dict[str, Any]], path: Path, headers: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="벤치마크 eval JSON들을 모아 비교 표로 만든다.")
    parser.add_argument("--dir", required=True, help="evaluation_results.json들이 들어있는 루트 디렉터리")
    parser.add_argument("--out", help="표 저장 경로 prefix (생략 시 <dir>/benchmark_table)")
    args = parser.parse_args()

    eval_dir = Path(args.dir)
    rows, headers = collect(eval_dir)
    if not rows:
        print(f"[aggregate_benchmark] {eval_dir} 아래 evaluation_results.json이 없다.")
        return

    out_prefix = Path(args.out) if args.out else eval_dir / "benchmark_table"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    md = to_markdown(rows, headers)
    md_path = out_prefix.with_suffix(".md")
    csv_path = out_prefix.with_suffix(".csv")
    md_path.write_text(md + "\n", encoding="utf-8")
    to_csv(rows, csv_path, headers)

    print(md)
    print(f"\n[aggregate_benchmark] {len(rows)}개 행 → {md_path}")
    print(f"[aggregate_benchmark] csv → {csv_path}")


if __name__ == "__main__":
    main()
