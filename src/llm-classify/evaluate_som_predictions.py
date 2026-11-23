"""이미 완료된 SoM predictions.json에서 baseline VLM ablation과 같은 형식의
`evaluation/` 아티팩트(summary/per_class/per_image csv + confusion_matrix.csv)를
만든다. 모델을 다시 안 돌려도 되고, `zeroshot_som.py --icl` 등이 이미 만들어 둔
predictions.json 하나만 있으면 된다 (신규 실행은 이제 이 아티팩트를 자동으로 남기지만,
과거에 이미 끝난 실행 결과를 채워 넣을 때는 이 스크립트를 쓴다).

Example:
    python "src/llm-classify/evaluate_som_predictions.py" \
        --predictions-json exp/260911_vlm-som/ag_rob_tomato/zeroshot/predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import autorootcwd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import som_common

from src.evaluation.llm_classify_artifact_writer import save_llm_classify_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoM predictions.json -> evaluation/ 아티팩트 백필")
    parser.add_argument("--predictions-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None, help="기본: predictions.json과 같은 폴더의 evaluation/")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_json_path = Path(args.predictions_json)
    output_dir = Path(args.output_dir) if args.output_dir else predictions_json_path.parent / "evaluation"

    results = som_common.evaluate_predictions_json_detailed(predictions_json_path)
    results["predictions_json"] = str(predictions_json_path)
    results["iou_threshold"] = args.iou_threshold

    meta_path = predictions_json_path.parent / "evaluation_results.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        results["dataset"] = meta.get("dataset")
        results["backend"] = meta.get("backend")
        results["proposal_source"] = meta.get("proposal_source")

    paths = save_llm_classify_artifacts(results, output_dir)
    print(f"[evaluate_som_predictions] acc={results['acc']:.4f} macro_f1={results['macro_f1']:.4f} -> {paths['json'].parent}")
    for cls, stats in results["per_class"].items():
        print(f"  {cls}: P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}")


if __name__ == "__main__":
    main()
