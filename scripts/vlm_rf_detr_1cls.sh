#!/usr/bin/env bash
#
# vlm_rf_detr_1cls.sh
#   RF-DETR 1cls detector prediction 위에 VLM classification을 얹어 실행한다.
#
# 고정 실험 조합:
#   1) tomatod -> tomatod
#   2) little  -> little
#
# 사용 VLM config:
#   config/vlm/qwen3_vl_vllm.yaml
#
# 참고:
#   --detector-prediction-dir 에는 predictions_coco.json 파일이 아니라
#   그 상위 prediction 디렉터리를 넘긴다.
#
# 예시:
#   ./scripts/vlm_rf_detr_1cls.sh
#   BENCH_GPU=0 ./scripts/vlm_rf_detr_1cls.sh
#   PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python ./scripts/vlm_rf_detr_1cls.sh

set -uo pipefail

cd "$(dirname "$0")/.."

SPLIT="test"
GPU="${BENCH_GPU:-}"
PY="${PYBIN:-$PWD/.venv/bin/python}"
VLM_CONFIG="qwen3_vl_4b"
FAILURES=0

show_usage() {
  cat <<'EOF'
Usage:
  ./scripts/vlm_rf_detr_1cls.sh [--split test] [--gpu 0] [--py /path/to/python]

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
    echo "[vlm_rf_detr_1cls] python not found: $PY" >&2
    exit 1
  fi

  if ! "$PY" -c "import yaml, pandas, vllm; from PIL import Image" >/dev/null 2>&1; then
    echo "[vlm_rf_detr_1cls][경고] '$PY' 에 필요한 패키지(yaml/pandas/vllm/PIL)가 없습니다." >&2
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


  local status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[fail] dataset=${dataset} status=${status}" >&2
    FAILURES=$((FAILURES + 1))
    return "$status"
  fi

  echo "[done] dataset=${dataset}"
}

check_env

echo "[vlm_rf_detr_1cls] split=${SPLIT} gpu=${GPU:-unset} py=${PY}"

run_case \
  "tomatod" \
  "result/detection_reasoning/tomatod/rf_detr_1cls_prediction" \
  "result/detection_reasoning/tomatod/rf_detr_1cls_qwen"

run_case \
  "little" \
  "result/detection_reasoning/little/rf_detr_1cls_prediction" \
  "result/detection_reasoning/little/rf_detr_1cls_qwen"

if [[ "$FAILURES" -ne 0 ]]; then
  echo "[vlm_rf_detr_1cls] failures=${FAILURES}" >&2
  exit 1
fi

echo "[vlm_rf_detr_1cls] all runs completed"
