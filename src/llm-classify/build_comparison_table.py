"""result/detection_only/<dataset>/*_eval/evaluation_results.json들을 모아
DaR 지표와 llm-classify 지표를 나란히 놓은 비교표(csv)를 만든다.

Example:
    python "src/llm-classify/build_comparison_table.py" \
        --eval-root result/detection_only/ag_rob_tomato --dataset-label ag_rob_tomato \
        --eval-root result/detection_only/little_re --dataset-label little_re \
        --output exp/260911_llm-classify-metrics-comparison/comparison_table.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DaR + llm-classify eval results into one comparison table.")
    parser.add_argument("--eval-root", action="append", required=True, help="*_eval 폴더들이 있는 상위 디렉토리. 여러 번 지정 가능.")
    parser.add_argument("--dataset-label", action="append", required=True, help="각 --eval-root에 대응하는 라벨. --eval-root와 같은 순서/개수.")
    parser.add_argument("--output", type=str, required=True, help="저장할 csv 경로.")
    return parser.parse_args()


def load_row(eval_dir: Path, dataset_label: str) -> dict:
    result = json.loads((eval_dir / "evaluation_results.json").read_text())
    stats = result["detailed_statistics"]["total_statistics"]
    llm = result["llm_classify_style"]
    map_metrics = result.get("map_metrics", {}) or result.get("detection_metrics", {})

    model_name = result["evaluation_info"].get("model_name", eval_dir.name.replace("_eval", ""))
    class_mode = "1cls" if model_name.endswith("1cls") else "2cls" if model_name.endswith("2cls") else "3cls"
    family = model_name.rsplit("_", 1)[0]

    return {
        "dataset": dataset_label,
        "family": family,
        "class_mode": class_mode,
        "model": model_name,
        "dar_detection_acc": stats.get("detection_acc"),
        "dar_classification_acc": stats.get("classification_acc"),
        "dar_overall_acc": stats.get("overall_acc"),
        "dar_precision": stats.get("overall_precision"),
        "dar_recall": stats.get("overall_recall"),
        "mAP@0.5": map_metrics.get("map_50"),
        "llm_classify_acc": llm.get("acc"),
        "llm_classify_macro_f1": llm.get("macro_f1"),
        "n_images": stats.get("num_images"),
        "total_gt": stats.get("total_ground_truths"),
        "total_pred": stats.get("total_predictions"),
    }


def main() -> None:
    args = parse_args()
    if len(args.eval_root) != len(args.dataset_label):
        raise SystemExit("--eval-root와 --dataset-label 개수가 같아야 한다.")

    rows = []
    for eval_root, dataset_label in zip(args.eval_root, args.dataset_label):
        for eval_dir in sorted(Path(eval_root).glob("*_eval")):
            json_path = eval_dir / "evaluation_results.json"
            if not json_path.is_file():
                continue
            rows.append(load_row(eval_dir, dataset_label))

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "family", "class_mode"]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nsaved to {output_path}")


if __name__ == "__main__":
    main()
