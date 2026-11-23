#!/usr/bin/env bash
#
# run_qwen3_5_ablation_matrix.sh
#   tomatod / little 두 데이터셋 x 4개 prompt 구성(ablation_qwen3_5_01~04, config/vlm/)을
#   Qwen3.5-27B(vllm backend)로 실행하고, 끝나면 metrics.json을 모아
#   classification_acc 요약표를 출력 + JSON으로 저장한다.
#
#   4개 구성:
#     01 baseline                  — 조정 없음, reasoning 없음
#     02 reasoning                 — + CoT reasoning
#     03 reasoning + reference     — + exemplar
#     04 reasoning + swatch        — + 색상 swatch
#
# 사용:
#   ./scripts/run_qwen3_5_ablation_matrix.sh
#   ABLATION_GPUS="2 4 5 6" ./scripts/run_qwen3_5_ablation_matrix.sh
#   ABLATION_RUN_NAME=260910_my_run PYBIN=/path/to/python ./scripts/run_qwen3_5_ablation_matrix.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYBIN:-$PWD/.venv/bin/python}"
read -ra GPUS <<< "${ABLATION_GPUS:-2 4 5 6}"
RUN_NAME="${ABLATION_RUN_NAME:-$(date +%y%m%d)_vlm-classify-qwen3_5-ablation}"
EXP_DIR="exp/${RUN_NAME}"

DATASETS=(tomatod little)
declare -A DETECTOR_DIR=(
  [tomatod]="result/detection_reasoning/tomatod/rf_detr_1cls_prediction"
  [little]="result/detection_reasoning/little/rf_detr_1cls_prediction"
)

CONFIGS=(
  ablation_qwen3_5_01_baseline
  ablation_qwen3_5_02_reasoning
  ablation_qwen3_5_03_reasoning_reference
  ablation_qwen3_5_04_reasoning_swatch
)

mkdir -p "$EXP_DIR/log"
echo "[qwen3_5_ablation_matrix] run_name=$RUN_NAME gpus=${GPUS[*]} exp_dir=$EXP_DIR"

pids=()
job_labels=()

launch_job() {
  local dataset="$1" config="$2" gpu="$3"
  local out_dir="$EXP_DIR/output/${dataset}/${config}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/vlm_classification_pipeline.py \
    --model rf_detr_1cls --dataset "$dataset" \
    --vlm-config "$config" --split test \
    --detector-prediction-dir "${DETECTOR_DIR[$dataset]}" \
    --output-dir "$out_dir" \
    --stdout-log "$EXP_DIR/log/${dataset}_${config}_stdout.log" \
    --stderr-log "$EXP_DIR/log/${dataset}_${config}_stderr.log" \
    --stderr-only > /dev/null 2>&1 &
  pids+=($!)
  job_labels+=("${dataset}/${config}(gpu${gpu},pid$!)")
}

wait_wave() {
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  pids=()
  job_labels=()
}

gpu_index=0
for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    gpu="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
    launch_job "$dataset" "$config" "$gpu"
    gpu_index=$((gpu_index + 1))
    if (( gpu_index % ${#GPUS[@]} == 0 )); then
      echo "[qwen3_5_ablation_matrix] wave launched: ${job_labels[*]}"
      wait_wave
    fi
  done
done

if (( ${#pids[@]} > 0 )); then
  echo "[qwen3_5_ablation_matrix] final wave launched: ${job_labels[*]}"
  wait_wave
fi

echo "[qwen3_5_ablation_matrix] all jobs complete, summarizing metrics"
"$PY" - "$EXP_DIR" "${DATASETS[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
datasets = sys.argv[2:]
configs = [
    "ablation_qwen3_5_01_baseline",
    "ablation_qwen3_5_02_reasoning",
    "ablation_qwen3_5_03_reasoning_reference",
    "ablation_qwen3_5_04_reasoning_swatch",
]

rows = []
for dataset in datasets:
    for config in configs:
        metrics_path = exp_dir / "output" / dataset / config / "metrics.json"
        if not metrics_path.exists():
            rows.append((dataset, config, None))
            continue
        metrics = json.loads(metrics_path.read_text())
        rows.append((dataset, config, metrics.get("classification_acc")))

print(f"{'dataset':<10} {'config':<40} classification_acc")
for dataset, config, acc in rows:
    acc_str = f"{acc * 100:.2f}%" if acc is not None else "MISSING"
    print(f"{dataset:<10} {config:<40} {acc_str}")

summary_path = exp_dir / "summary_classification_acc.json"
summary_path.write_text(json.dumps(
    [{"dataset": d, "config": c, "classification_acc": a} for d, c, a in rows],
    indent=2,
))
print(f"[qwen3_5_ablation_matrix] summary saved to {summary_path}")
PYEOF
