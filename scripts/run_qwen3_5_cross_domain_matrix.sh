#!/usr/bin/env bash
#
# run_qwen3_5_cross_domain_matrix.sh
#   RF-DETR가 다른 데이터셋으로 학습되어 도메인이 다른 테스트셋에 적용된 예측
#   (exp/260910_laboro-to-tomatod, exp/260910_tomatod-to-laboro) 위에
#   Qwen3.5-27B 5개 prompt 구성(baseline/step-reasoning/schema-reasoning/
#   reference/swatch)을 실행하고 metrics.json을 모아 요약한다.
#
# 사용:
#   ./scripts/run_qwen3_5_cross_domain_matrix.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYBIN:-$PWD/.venv/bin/python}"
read -ra GPUS <<< "${ABLATION_GPUS:-2 4 5 6}"
RUN_NAME="${ABLATION_RUN_NAME:-$(date +%y%m%d)_vlm-classify-qwen3_5-cross-domain}"
EXP_DIR="exp/${RUN_NAME}"

# domain_key -> (eval dataset config, detector prediction dir)
DOMAINS=(laboro_to_tomatod tomatod_to_laboro)
declare -A DATASET=(
  [laboro_to_tomatod]="tomatod"
  [tomatod_to_laboro]="little"
)
declare -A DETECTOR_DIR=(
  [laboro_to_tomatod]="exp/260910_laboro-to-tomatod/prediction"
  [tomatod_to_laboro]="exp/260910_tomatod-to-laboro/prediction"
)

CONFIGS=(
  ablation_qwen3_5_01_baseline
  ablation_qwen3_5_02_reasoning
  ablation_qwen3_5_02b_schema_reasoning
  ablation_qwen3_5_03_reasoning_reference
  ablation_qwen3_5_04_reasoning_swatch
)

mkdir -p "$EXP_DIR/log"
echo "[cross_domain_matrix] run_name=$RUN_NAME gpus=${GPUS[*]} exp_dir=$EXP_DIR"

pids=()
job_labels=()

launch_job() {
  local domain="$1" config="$2" gpu="$3"
  local out_dir="$EXP_DIR/output/${domain}/${config}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/vlm_classification_pipeline.py \
    --model rf_detr_1cls --dataset "${DATASET[$domain]}" \
    --vlm-config "$config" --split test \
    --detector-prediction-dir "${DETECTOR_DIR[$domain]}" \
    --output-dir "$out_dir" \
    --stdout-log "$EXP_DIR/log/${domain}_${config}_stdout.log" \
    --stderr-log "$EXP_DIR/log/${domain}_${config}_stderr.log" \
    --stderr-only > /dev/null 2>&1 &
  pids+=($!)
  job_labels+=("${domain}/${config}(gpu${gpu},pid$!)")
}

wait_wave() {
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  pids=()
  job_labels=()
}

gpu_index=0
for domain in "${DOMAINS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    gpu="${GPUS[$((gpu_index % ${#GPUS[@]}))]}"
    launch_job "$domain" "$config" "$gpu"
    gpu_index=$((gpu_index + 1))
    if (( gpu_index % ${#GPUS[@]} == 0 )); then
      echo "[cross_domain_matrix] wave launched: ${job_labels[*]}"
      wait_wave
    fi
  done
done

if (( ${#pids[@]} > 0 )); then
  echo "[cross_domain_matrix] final wave launched: ${job_labels[*]}"
  wait_wave
fi

echo "[cross_domain_matrix] all jobs complete, summarizing metrics"
"$PY" - "$EXP_DIR" "${DOMAINS[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
domains = sys.argv[2:]
configs = [
    "ablation_qwen3_5_01_baseline",
    "ablation_qwen3_5_02_reasoning",
    "ablation_qwen3_5_02b_schema_reasoning",
    "ablation_qwen3_5_03_reasoning_reference",
    "ablation_qwen3_5_04_reasoning_swatch",
]

rows = []
for domain in domains:
    for config in configs:
        metrics_path = exp_dir / "output" / domain / config / "metrics.json"
        if not metrics_path.exists():
            rows.append((domain, config, None))
            continue
        metrics = json.loads(metrics_path.read_text())
        rows.append((domain, config, metrics.get("classification_acc")))

print(f"{'domain':<20} {'config':<40} classification_acc")
for domain, config, acc in rows:
    acc_str = f"{acc * 100:.2f}%" if acc is not None else "MISSING"
    print(f"{domain:<20} {config:<40} {acc_str}")

summary_path = exp_dir / "summary_classification_acc.json"
summary_path.write_text(json.dumps(
    [{"domain": d, "config": c, "classification_acc": a} for d, c, a in rows],
    indent=2,
))
print(f"[cross_domain_matrix] summary saved to {summary_path}")
PYEOF
