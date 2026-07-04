#!/usr/bin/env bash
#
# pred_eval_overlay_2cls.sh
#   2cls 전용(custom_tomato) prediction + evaluation + overlay 실행 스크립트.
#
# 실행 조합(고정):
#   - custom_tomato -> custom_tomato
#
# 결과 경로 원칙:
#   - 각 모델 config의 dataset_overrides.custom_tomato 가 정의한
#     prediction_dir / eval_dir 를 그대로 사용한다.
#
# 예시:
#   BENCH_GPU=0 ./scripts/pred_eval_overlay_2cls.sh
#   BENCH_GPU=0 ./scripts/pred_eval_overlay_2cls.sh --models "yolo11_2cls rf_detr_2cls"
#   ./scripts/pred_eval_overlay_2cls.sh --no-predict --no-eval
#   ./scripts/pred_eval_overlay_2cls.sh --show-conf --max-images 50

set -uo pipefail

cd "$(dirname "$0")/.."

DATASET="custom_tomato"
SPLIT=test
DO_PREDICT=1
DO_EVAL=1
DO_OVERLAY=1
SHOW_CONF=0
MAX_IMAGES=""
MODELS="yolo11_2cls yolo12_2cls yolo_world_2cls rt_detr_2cls rf_detr_2cls"
GPU="${BENCH_GPU:-}"
PY="${PYBIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --no-predict) DO_PREDICT=0; shift ;;
    --no-eval) DO_EVAL=0; shift ;;
    --no-overlay) DO_OVERLAY=0; shift ;;
    --show-conf) SHOW_CONF=1; shift ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

if ! "$PY" -c "import yaml, torch, rfdetr, pandas; from PIL import Image" >/dev/null 2>&1; then
  echo "[pred_eval_overlay_2cls][경고] '$PY' 에 필요한 패키지(yaml/torch/rfdetr/pandas/PIL)가 없습니다."
  echo "  올바른 venv 로 다시 실행하세요. 예:"
  echo "    PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python $0"
  exit 1
fi

family_of() {
  local model="$1"
  "$PY" - <<'PY' "config/model/${model}.yaml"
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
print(data.get("family", ""))
PY
}

dataset_override_field() {
  local model_cfg="$1"
  local dataset="$2"
  local field="$3"
  "$PY" - <<'PY' "$model_cfg" "$dataset" "$field"
import sys
import yaml

cfg_path, dataset, field = sys.argv[1:4]
with open(cfg_path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
value = ((data.get("dataset_overrides") or {}).get(dataset) or {}).get(field, "")
print(value)
PY
}

weight_path() {
  local dataset="$1"
  local model="$2"
  local family="$3"

  if [[ "$family" == "rf_detr" ]]; then
    for c in \
      "runs/${dataset}/${model}/checkpoint_best_ema.pth" \
      "runs/${dataset}/${model}/checkpoint_best_regular.pth" \
      "runs/${dataset}/${model}/checkpoint.pth"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  else
    for c in \
      "runs/${dataset}/${model}/weights/best.pt" \
      "runs/${dataset}/${model}/weights/last.pt"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  fi

  echo ""
}

render_overlay() {
  local family="$1"
  local model_cfg="$2"

  local overlay_args=()
  if [[ "$SHOW_CONF" -eq 0 ]]; then
    overlay_args+=(--label-only)
  fi
  if [[ -n "$MAX_IMAGES" ]]; then
    overlay_args+=(--max-images "$MAX_IMAGES")
  fi

  if [[ "$family" == "rf_detr" ]]; then
    "$PY" scripts/render_coco_overlays.py \
      --model-config "$model_cfg" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      "${overlay_args[@]}"
  else
    "$PY" scripts/render_yolo_overlays.py \
      --model-config "$model_cfg" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      "${overlay_args[@]}"
  fi
}

echo "[pred_eval_overlay_2cls] dataset=${DATASET} split=${SPLIT} gpu=${GPU:-unset} py=${PY}"
echo "[pred_eval_overlay_2cls] predict=${DO_PREDICT} eval=${DO_EVAL} overlay=${DO_OVERLAY} show_conf=${SHOW_CONF}"
echo "[pred_eval_overlay_2cls] models=${MODELS}"

for model in $MODELS; do
  model_cfg="config/model/${model}.yaml"
  [[ -f "$model_cfg" ]] || { echo "[skip] config 없음: $model"; continue; }

  family="$(family_of "$model")" || { echo "[skip] family 읽기 실패: $model"; continue; }
  wt="$(weight_path "$DATASET" "$model" "$family")"
  if [[ -z "$wt" ]]; then
    echo "[skip] 가중치 없음: dataset=${DATASET} model=${model}"
    continue
  fi

  pred_dir="$(dataset_override_field "$model_cfg" "$DATASET" "prediction_dir")" \
    || { echo "[skip] prediction_dir 읽기 실패: dataset=${DATASET} model=${model}"; continue; }
  eval_dir="$(dataset_override_field "$model_cfg" "$DATASET" "eval_dir")" \
    || { echo "[skip] eval_dir 읽기 실패: dataset=${DATASET} model=${model}"; continue; }
  [[ -n "$pred_dir" && -n "$eval_dir" ]] \
    || { echo "[skip] prediction/eval dir 비어 있음: dataset=${DATASET} model=${model}"; continue; }

  echo "=================================================================="
  echo "[run] dataset=${DATASET} model=${model} family=${family}"
  echo "[run] weights=${wt}"
  echo "[run] prediction_dir=${pred_dir}"
  echo "[run] eval_dir=${eval_dir}"

  if [[ "$DO_PREDICT" -eq 1 ]]; then
    echo "[predict] dataset=${DATASET} model=${model}"
    if [[ "$family" == "rf_detr" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/train_model.py \
        --model-config "$model_cfg" \
        --dataset "$DATASET" \
        --stage predict \
        --weights "$wt" \
        --source "$SPLIT" \
        || { echo "[fail] predict 실패: dataset=${DATASET} model=${model}"; continue; }
    else
      CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/predict_yolo_labels.py \
        --model-config "$model_cfg" \
        --dataset "$DATASET" \
        --weights "$wt" \
        --source "$SPLIT" \
        || { echo "[fail] predict 실패: dataset=${DATASET} model=${model}"; continue; }
    fi
  fi

  if [[ "$DO_EVAL" -eq 1 ]]; then
    echo "[eval] dataset=${DATASET} model=${model}"
    if [[ "$family" == "rf_detr" ]]; then
      "$PY" scripts/evaluate_coco_predictions.py \
        --model-config "$model_cfg" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --output-dir "$eval_dir" \
        || { echo "[warn] eval 실패: dataset=${DATASET} model=${model}"; }
    else
      "$PY" scripts/evaluate_yolo_predictions.py \
        --model-config "$model_cfg" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --output-dir "$eval_dir" \
        || { echo "[warn] eval 실패: dataset=${DATASET} model=${model}"; }
    fi
  fi

  if [[ "$DO_OVERLAY" -eq 1 ]]; then
    echo "[overlay] dataset=${DATASET} model=${model}"
    render_overlay "$family" "$model_cfg" \
      || echo "[warn] overlay 실패: dataset=${DATASET} model=${model}"
  fi
done

echo "=================================================================="
echo "[pred_eval_overlay_2cls] 완료"
