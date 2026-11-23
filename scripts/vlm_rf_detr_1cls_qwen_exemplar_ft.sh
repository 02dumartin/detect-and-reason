#!/usr/bin/env bash
#
# Run RF-DETR 1cls + Qwen3-VL full-FT-with-exemplars VLM classification.
#
# Example:
#   BENCH_GPU=7 ./scripts/vlm_rf_detr_1cls_qwen_exemplar_ft.sh
#   BENCH_GPU=7 ./scripts/vlm_rf_detr_1cls_qwen_exemplar_ft.sh --case little
#   BENCH_GPU=7 ./scripts/vlm_rf_detr_1cls_qwen_exemplar_ft.sh --case all

set -uo pipefail

cd "$(dirname "$0")/.."

SPLIT="test"
GPU="${BENCH_GPU:-}"
PY="${PYBIN:-$PWD/.venv/bin/python}"
VLM_CONFIG="qwen3_vl_4b_full_ft_with_exemplars_k2"
CASE_FILTER="all"
OUTPUT_SUFFIX="qwen_exemplar_ft_k2"
FAILURES=0

show_usage() {
  cat <<'EOF'
Usage:
  ./scripts/vlm_rf_detr_1cls_qwen_exemplar_ft.sh [--case all|tomatod|little|merge-tomatod|merge-little|custom_tomato|rob2pheno]
                                                  [--split test] [--gpu 7] [--py /path/to/python]
                                                  [--vlm-config qwen3_vl_4b_full_ft_with_exemplars_k2]

Environment:
  BENCH_GPU   CUDA_VISIBLE_DEVICES override
  PYBIN       Python binary override
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE_FILTER="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --py) PY="$2"; shift 2 ;;
    --vlm-config) VLM_CONFIG="$2"; shift 2 ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

check_env() {
  if [[ ! -x "$PY" ]]; then
    echo "[qwen_exemplar_ft] python not found: $PY" >&2
    exit 1
  fi

  if ! "$PY" -c "import yaml, pandas, torch, transformers; from PIL import Image; import qwen_vl_utils" >/dev/null 2>&1; then
    echo "[qwen_exemplar_ft] required packages are missing in: $PY" >&2
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

should_run() {
  local case_name="$1"
  [[ "$CASE_FILTER" == "all" || "$CASE_FILTER" == "$case_name" ]]
}

run_case() {
  local case_name="$1"
  local dataset="$2"
  local pred_dir="$3"
  local output_dir="$4"
  local split_dir
  split_dir="$(coco_split_dir_name "$SPLIT")"
  local pred_json="${pred_dir}/${split_dir}/predictions_coco.json"

  if ! should_run "$case_name"; then
    return 0
  fi

  echo "=================================================================="
  echo "[run] case=${case_name}"
  echo "[run] dataset=${dataset}"
  echo "[run] split=${SPLIT}"
  echo "[run] prediction_json=${pred_json}"
  echo "[run] output_dir=${output_dir}"
  echo "[run] vlm_config=${VLM_CONFIG}"

  if [[ ! -f "$pred_json" ]]; then
    echo "[fail] prediction json not found: ${pred_json}" >&2
    FAILURES=$((FAILURES + 1))
    return 1
  fi

  mkdir -p "$output_dir"

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

  local status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[fail] case=${case_name} status=${status}" >&2
    FAILURES=$((FAILURES + 1))
    return "$status"
  fi

  echo "[done] case=${case_name}"
}

check_env

echo "[qwen_exemplar_ft] split=${SPLIT} gpu=${GPU:-unset} py=${PY}"
echo "[qwen_exemplar_ft] vlm_config=${VLM_CONFIG}"
echo "[qwen_exemplar_ft] case=${CASE_FILTER}"

run_case \
  "tomatod" \
  "tomatod" \
  "result/detection_reasoning/tomatod/rf_detr_1cls_prediction" \
  "result/detection_reasoning/tomatod/rf_detr_1cls_${OUTPUT_SUFFIX}"

run_case \
  "little" \
  "little" \
  "result/detection_reasoning/little/rf_detr_1cls_prediction" \
  "result/detection_reasoning/little/rf_detr_1cls_${OUTPUT_SUFFIX}"

run_case \
  "merge-tomatod" \
  "tomatod" \
  "result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_prediction" \
  "result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_${OUTPUT_SUFFIX}"

run_case \
  "merge-little" \
  "little" \
  "result/detection_reasoning/merge/little/rf_detr_1cls_little_prediction" \
  "result/detection_reasoning/merge/little/rf_detr_1cls_little_${OUTPUT_SUFFIX}"

run_case \
  "custom_tomato" \
  "custom_tomato" \
  "result/detection_reasoning/custom_tomato/rf_detr_1cls_prediction" \
  "result/detection_reasoning/custom_tomato/rf_detr_1cls_${OUTPUT_SUFFIX}"

run_case \
  "rob2pheno" \
  "rob2pheno" \
  "result/detection_reasoning/rob2pheno/rf_detr_1cls_prediction" \
  "result/detection_reasoning/rob2pheno/rf_detr_1cls_${OUTPUT_SUFFIX}"

if [[ "$FAILURES" -ne 0 ]]; then
  echo "[qwen_exemplar_ft] failures=${FAILURES}" >&2
  exit 1
fi

echo "[qwen_exemplar_ft] completed"
