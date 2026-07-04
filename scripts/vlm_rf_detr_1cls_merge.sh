#!/usr/bin/env bash
#
# vlm_rf_detr_1cls_merge.sh
#   merge 학습으로 만든 RF-DETR 1cls detector prediction 위에
#   VLM classification을 얹어 실행한다.
#
# 고정 실험 조합:
#   1) merge prediction -> tomatod
#   2) merge prediction -> little
#
# 사용 VLM config:
#   config/vlm/qwen3_vl_vllm.yaml
#
# 참고:
#   dataset 은 target dataset(tomatod/little)을 그대로 사용하고,
#   prediction 경로만 merge 결과로 override 한다.
#
# 예시:
#   ./scripts/vlm_rf_detr_1cls_merge.sh
#   BENCH_GPU=0 ./scripts/vlm_rf_detr_1cls_merge.sh
#   PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python ./scripts/vlm_rf_detr_1cls_merge.sh

set -uo pipefail

cd "$(dirname "$0")/.."

SPLIT="test"
GPU="${BENCH_GPU:-}"
PY="${PYBIN:-$PWD/.venv/bin/python}"
VLM_CONFIG="qwen3_vl_vllm"
FAILURES=0

show_usage() {
  cat <<'EOF'
Usage:
  ./scripts/vlm_rf_detr_1cls_merge.sh [--split test] [--gpu 0] [--py /path/to/python]

Options:
  --split   Target split. Default: test
  --gpu     CUDA_VISIBLE_DEVICES override
  --py      Python binary override
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --py) PY="$2"; shift 2 ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

check_env() {
  if [[ ! -x "$PY" ]]; then
    echo "[vlm_rf_detr_1cls_merge] python not found: $PY" >&2
    exit 1
  fi

  if ! "$PY" -c "import yaml, pandas, vllm; from PIL import Image" >/dev/null 2>&1; then
    echo "[vlm_rf_detr_1cls_merge][경고] '$PY' 에 필요한 패키지(yaml/pandas/vllm/PIL)가 없습니다." >&2
    echo "  올바른 venv 로 다시 실행하세요. 예:" >&2
    echo "    PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python $0" >&2
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
  local dataset="$1"
  local pred_dir="$2"
  local output_dir="$3"
  local split_dir
  split_dir="$(coco_split_dir_name "$SPLIT")"
  local pred_json="${pred_dir}/${split_dir}/predictions_coco.json"

  echo "=================================================================="
  echo "[run] dataset=${dataset}"
  echo "[run] split=${SPLIT}"
  echo "[run] prediction_dir=${pred_dir}"
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
    echo "[fail] dataset=${dataset} status=${status}" >&2
    FAILURES=$((FAILURES + 1))
    return "$status"
  fi

  echo "[done] dataset=${dataset}"
}

check_env

echo "[vlm_rf_detr_1cls_merge] split=${SPLIT} gpu=${GPU:-unset} py=${PY}"

run_case \
  "tomatod" \
  "result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_prediction" \
  "result/detection_reasoning/merge/tomatod/rf_detr_1cls_tomatod_qwen"

run_case \
  "little" \
  "result/detection_reasoning/merge/little/rf_detr_1cls_little_prediction" \
  "result/detection_reasoning/merge/little/rf_detr_1cls_little_qwen"

if [[ "$FAILURES" -ne 0 ]]; then
  echo "[vlm_rf_detr_1cls_merge] failures=${FAILURES}" >&2
  exit 1
fi

echo "[vlm_rf_detr_1cls_merge] all runs completed"
