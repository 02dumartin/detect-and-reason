#!/usr/bin/env bash
#
# pred_eval_overlay_zs.sh
#   zero-shot 2cls 전용(custom_tomato, rob2pheno) prediction + evaluation + overlay 실행 스크립트.
#
# 기본 실행 조합:
#   - datasets: custom_tomato, rob2pheno
#   - models  : yolo_world_2cls_zs, grounding_dino_2cls_zs, owl_vit_2cls_zs
#
# 결과 경로 원칙:
#   - 각 zero-shot config의 dataset_overrides.<dataset> 가 정의한
#     prediction_dir / eval_dir 를 그대로 사용한다.

set -uo pipefail

cd "$(dirname "$0")/.."

SPLIT=test
DO_PREDICT=1
DO_EVAL=1
DO_OVERLAY=1
SHOW_CONF=0
MAX_IMAGES=""
MODELS="yolo_world_2cls_zs grounding_dino_2cls_zs owl_vit_2cls_zs"
DATASETS="custom_tomato rob2pheno"
GPU="${BENCH_GPU:-}"
PY="${PYBIN:-python3}"
EVAL_IOU=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --no-predict) DO_PREDICT=0; shift ;;
    --no-eval) DO_EVAL=0; shift ;;
    --no-overlay) DO_OVERLAY=0; shift ;;
    --show-conf) SHOW_CONF=1; shift ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    --eval-iou) EVAL_IOU="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

if ! "$PY" -c "import yaml, torch, pandas; from PIL import Image" >/dev/null 2>&1; then
  echo "[pred_eval_overlay_zs][경고] '$PY' 에 필요한 패키지(yaml/torch/pandas/PIL)가 없습니다."
  echo "  올바른 venv 로 다시 실행하세요. 예:"
  echo "    PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python $0"
  exit 1
fi

config_path_of() {
  local model="$1"
  for candidate in \
    "config/zeroshot/${model}.yaml" \
    "config/model/${model}.yaml"; do
    [[ -f "$candidate" ]] && { echo "$candidate"; return; }
  done
  echo ""
}

family_of() {
  local model_cfg="$1"
  "$PY" - <<'PY' "$model_cfg"
import sys
import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as handle:
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

render_overlay() {
  local model="$1"
  local dataset="$2"
  local model_cfg="$3"
  local img_root=""

  local overlay_args=()
  if [[ "$SHOW_CONF" -eq 0 ]]; then
    overlay_args+=(--label-only)
  fi
  if [[ -n "$MAX_IMAGES" ]]; then
    overlay_args+=(--max-images "$MAX_IMAGES")
  fi
  if [[ "$SPLIT" == "test" ]]; then
    img_root="$(dataset_override_field "$model_cfg" "$dataset" "test_image_source")"
    if [[ -n "$img_root" ]]; then
      overlay_args+=(--img-root "$img_root")
    fi
  fi

  "$PY" scripts/render_yolo_overlays.py \
    --model "$model" \
    --dataset "$dataset" \
    --split "$SPLIT" \
    "${overlay_args[@]}"
}

predict_source_args() {
  if [[ "$SPLIT" == "test" ]]; then
    return 0
  fi
  printf '%s\n' --source "$SPLIT"
}

echo "[pred_eval_overlay_zs] datasets=${DATASETS} split=${SPLIT} gpu=${GPU:-unset} py=${PY}"
echo "[pred_eval_overlay_zs] predict=${DO_PREDICT} eval=${DO_EVAL} overlay=${DO_OVERLAY} show_conf=${SHOW_CONF}"
echo "[pred_eval_overlay_zs] models=${MODELS}"
echo "[pred_eval_overlay_zs] eval_iou=${EVAL_IOU}"

for dataset in $DATASETS; do
  for model in $MODELS; do
    model_cfg="$(config_path_of "$model")"
    [[ -n "$model_cfg" ]] || { echo "[skip] config 없음: $model"; continue; }

    family="$(family_of "$model_cfg")" || { echo "[skip] family 읽기 실패: $model"; continue; }
    pred_dir="$(dataset_override_field "$model_cfg" "$dataset" "prediction_dir")" \
      || { echo "[skip] prediction_dir 읽기 실패: dataset=${dataset} model=${model}"; continue; }
    eval_dir="$(dataset_override_field "$model_cfg" "$dataset" "eval_dir")" \
      || { echo "[skip] eval_dir 읽기 실패: dataset=${dataset} model=${model}"; continue; }
    [[ -n "$pred_dir" && -n "$eval_dir" ]] \
      || { echo "[skip] prediction/eval dir 비어 있음: dataset=${dataset} model=${model}"; continue; }

    echo "=================================================================="
    echo "[run] dataset=${dataset} model=${model} family=${family}"
    echo "[run] prediction_dir=${pred_dir}"
    echo "[run] eval_dir=${eval_dir}"

    if [[ "$DO_PREDICT" -eq 1 ]]; then
      echo "[predict] dataset=${dataset} model=${model}"
      mapfile -t source_args < <(predict_source_args)
      CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/predict_yolo_labels.py \
        --model "$model" \
        --dataset "$dataset" \
        "${source_args[@]}" \
        || { echo "[fail] predict 실패: dataset=${dataset} model=${model}"; continue; }
    fi

    if [[ "$DO_EVAL" -eq 1 ]]; then
      echo "[eval] dataset=${dataset} model=${model}"
      "$PY" scripts/evaluate_yolo_predictions.py \
        --model "$model" \
        --dataset "$dataset" \
        --split "$SPLIT" \
        --iou-threshold "$EVAL_IOU" \
        --output-dir "$eval_dir" \
        || { echo "[warn] eval 실패: dataset=${dataset} model=${model}"; }
    fi

    if [[ "$DO_OVERLAY" -eq 1 ]]; then
      echo "[overlay] dataset=${dataset} model=${model}"
      render_overlay "$model" "$dataset" "$model_cfg" \
        || echo "[warn] overlay 실패: dataset=${dataset} model=${model}"
    fi
  done
done

echo "=================================================================="
echo "[pred_eval_overlay_zs] 완료"
