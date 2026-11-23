#!/usr/bin/env bash
#
# Run RF-DETR 1cls + Qwen3-VL full-FT (3cls FFT) classification on the
# four requested detector-prediction pairs.
#
# Execution is batched 2-at-a-time:
#   batch 1: tomatod, little
#   batch 2: merge-tomatod, merge-little
#
# GPU assignment:
#   use BENCH_GPUS=6,7 to pin batch jobs to GPU 6 and 7.

set -uo pipefail

cd "$(dirname "$0")/.."

SPLIT="test"
GPU_SPEC="${BENCH_GPUS:-${BENCH_GPU:-}}"
PY="${PYBIN:-$PWD/.venv/bin/python}"
VLM_CONFIG="qwen3_vl_4b_full_ft_with_exemplars_k2"
FAILURES=0

show_usage() {
  cat <<'EOF'
Usage:
  ./scripts/vlm_rf_detr_1cls_qwen_fft.sh [--split test] [--gpus 6,7] [--py /path/to/python]
                                         [--vlm-config qwen3_vl_4b_full_ft_with_exemplars_k2]

Environment:
  BENCH_GPUS  Comma-separated GPU list, e.g. 6,7
  BENCH_GPU   Backward-compatible single GPU override
  PYBIN       Python binary override
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --gpus) GPU_SPEC="$2"; shift 2 ;;
    --gpu) GPU_SPEC="$2"; shift 2 ;;
    --py) PY="$2"; shift 2 ;;
    --vlm-config) VLM_CONFIG="$2"; shift 2 ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

IFS=, read -r -a GPU_LIST <<< "${GPU_SPEC}"
if [[ ${#GPU_LIST[@]} -gt 0 ]]; then
  for idx in "${!GPU_LIST[@]}"; do
    GPU_LIST[$idx]="${GPU_LIST[$idx]//[[:space:]]/}"
  done
fi

check_env() {
  if [[ ! -x "$PY" ]]; then
    echo "[vlm_rf_detr_1cls_qwen_fft] python not found: $PY" >&2
    exit 1
  fi

  if ! "$PY" -c "import yaml, pandas, torch, transformers; from PIL import Image; import qwen_vl_utils" >/dev/null 2>&1; then
    echo "[vlm_rf_detr_1cls_qwen_fft][경고] '$PY' 에 필요한 패키지(yaml/pandas/torch/transformers/qwen_vl_utils/PIL)가 없습니다." >&2
    echo "  올바른 venv 로 다시 실행하세요. 예:" >&2
    echo "    PYBIN=/home/jovyan/aicon-gamma-datavol-1/hjgoh/detect-and-reason/.venv/bin/python $0" >&2
    exit 1
  fi
}

coco_split_dir_name() {
  local split="${1,,}"
  case "$split" in
    val|validation) echo "valid" ;;
    *) echo "$split" ;;
  esac
}

run_case() {
  local case_name="$1"
  local dataset="$2"
  local pred_dir="$3"
  local output_dir="$4"
  local gpu="${5:-}"
  local split_dir
  split_dir="$(coco_split_dir_name "$SPLIT")"
  local pred_json="${pred_dir}/${split_dir}/predictions_coco.json"

  echo "=================================================================="
  echo "[run] case=${case_name}"
  echo "[run] dataset=${dataset}"
  echo "[run] split=${SPLIT}"
  echo "[run] gpu=${gpu:-unset}"
  echo "[run] prediction_json=${pred_json}"
  echo "[run] output_dir=${output_dir}"
  echo "[run] vlm_config=${VLM_CONFIG}"

  if [[ ! -f "$pred_json" ]]; then
    echo "[fail] prediction json not found: ${pred_json}" >&2
    return 1
  fi

  mkdir -p "$output_dir"

  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/vlm_classification_pipeline.py \
      --model rf_detr_1cls \
      --dataset "$dataset" \
      --vlm-config "$VLM_CONFIG" \
      --split "$SPLIT" \
      --detector-prediction-dir "$pred_dir" \
      --output-dir "$output_dir" \
      --stdout-log "${output_dir}/stdout.log" \
      --stderr-log "${output_dir}/stderr.log" \
      --stderr-only
  else
    "$PY" scripts/vlm_classification_pipeline.py \
      --model rf_detr_1cls \
      --dataset "$dataset" \
      --vlm-config "$VLM_CONFIG" \
      --split "$SPLIT" \
      --detector-prediction-dir "$pred_dir" \
      --output-dir "$output_dir" \
      --stdout-log "${output_dir}/stdout.log" \
      --stderr-log "${output_dir}/stderr.log" \
      --stderr-only
  fi
}

run_case_bg() {
  local gpu="$1"
  shift
  (
    run_case "$@" "$gpu"
  ) &
  RUN_CASE_BG_PID=$!
}

run_pair() {
  local gpu_a="${1:-}"
  local gpu_b="${2:-}"
  shift 2

  local pid1 pid2 status1 status2
  run_case_bg "$gpu_a" "$1" "$2" "$3" "$4"
  pid1="$RUN_CASE_BG_PID"
  run_case_bg "$gpu_b" "$5" "$6" "$7" "$8"
  pid2="$RUN_CASE_BG_PID"

  wait "$pid1"
  status1=$?
  wait "$pid2"
  status2=$?

  if [[ "$status1" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    echo "[fail] batch job 1 exited with status=${status1}" >&2
  fi
  if [[ "$status2" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    echo "[fail] batch job 2 exited with status=${status2}" >&2
  fi
}

pick_gpu() {
  local idx="$1"
  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo ""
    return 0
  fi
  local pos=$((idx % ${#GPU_LIST[@]}))
  echo "${GPU_LIST[$pos]}"
}

check_env

echo "[vlm_rf_detr_1cls_qwen_fft] split=${SPLIT} gpu_spec=${GPU_SPEC:-unset} py=${PY}"
echo "[vlm_rf_detr_1cls_qwen_fft] vlm_config=${VLM_CONFIG}"

gpu0="$(pick_gpu 0)"
gpu1="$(pick_gpu 1)"

run_pair \
  "$gpu0" \
  "$gpu1" \
  "tomatod" \
  "tomatod" \
  "result/detection_reasoning/tomatod/rf_detr_1cls_prediction" \
  "result/detection_reasoning/tomatod/rf_detr_1cls_qwen" \
  "little" \
  "little" \
  "result/detection_reasoning/little/rf_detr_1cls_prediction" \
  "result/detection_reasoning/little/rf_detr_1cls_qwen"

run_pair \
  "$gpu0" \
  "$gpu1" \
  "merge-tomatod" \
  "tomatod" \
  "result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_prediction" \
  "result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_qwen" \
  "merge-little" \
  "little" \
  "result/detection_reasoning/merge/little/rf_detr_1cls_little_prediction" \
  "result/detection_reasoning/merge/little/rf_detr_1cls_little_qwen"

if [[ "$FAILURES" -ne 0 ]]; then
  echo "[vlm_rf_detr_1cls_qwen_fft] failures=${FAILURES}" >&2
  exit 1
fi

echo "[vlm_rf_detr_1cls_qwen_fft] completed"
