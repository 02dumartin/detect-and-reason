#!/usr/bin/env bash
#
# pred_eval_overlay_1cls_merge.sh
#   merge 1cls 가중치로 big/little/tomatod test set 에 대해
#   prediction + evaluation + overlay 를 수행한다.
#
# 실행 조합(고정):
#   - merge -> big
#   - merge -> little
#   - merge -> tomatod
#
# 결과 경로:
#   result/detection_reasoning/merge/<target>/<model>_<target>_prediction
#   result/detection_reasoning/merge/<target>/<model>_<target>_eval
#
# 예시:
#   BENCH_GPU=0 ./scripts/pred_eval_overlay_1cls_merge.sh
#   BENCH_GPU=0 ./scripts/pred_eval_overlay_1cls_merge.sh --models "yolo11_1cls rf_detr_1cls"
#   ./scripts/pred_eval_overlay_1cls_merge.sh --no-predict --no-eval
#   ./scripts/pred_eval_overlay_1cls_merge.sh --show-conf --max-images 50

set -uo pipefail

cd "$(dirname "$0")/.."

SPLIT=test
DO_PREDICT=1
DO_EVAL=1
DO_OVERLAY=1
SHOW_CONF=0
MAX_IMAGES=""
MODELS="yolo11_1cls yolo12_1cls yolo_world_1cls rt_detr_1cls rf_detr_1cls"
GPU="${BENCH_GPU:-}"
PY="${PYBIN:-python3}"

TRAIN_DATASET="merge"
TARGET_DATASETS=(
  "big"
  "little"
  "tomatod"
)

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
  echo "[pred_eval_overlay_1cls_merge][경고] '$PY' 에 필요한 패키지(yaml/torch/rfdetr/pandas/PIL)가 없습니다."
  echo "  올바른 venv 로 다시 실행하세요. 예:"
  echo "    PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python $0"
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/pred_eval_overlay_1cls_merge.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

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
  local model="$1"
  local family="$2"

  if [[ "$family" == "rf_detr" ]]; then
    for c in \
      "runs/${TRAIN_DATASET}/${model}/checkpoint_best_ema.pth" \
      "runs/${TRAIN_DATASET}/${model}/checkpoint_best_regular.pth" \
      "runs/${TRAIN_DATASET}/${model}/checkpoint.pth"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  else
    for c in \
      "runs/${TRAIN_DATASET}/${model}/weights/best.pt" \
      "runs/${TRAIN_DATASET}/${model}/weights/last.pt"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  fi

  echo ""
}

make_model_config_for_target() {
  local model="$1"
  local eval_dataset="$2"
  local orig_cfg="config/model/${model}.yaml"
  local out_cfg="${TMP_DIR}/${model}__${TRAIN_DATASET}__to__${eval_dataset}.yaml"
  local pred_dir="result/detection_reasoning/${TRAIN_DATASET}/${eval_dataset}/${model}_${eval_dataset}_prediction"
  local eval_dir="result/detection_reasoning/${TRAIN_DATASET}/${eval_dataset}/${model}_${eval_dataset}_eval"

  "$PY" - <<'PY' "$orig_cfg" "$eval_dataset" "$out_cfg" "$pred_dir" "$eval_dir" "$model"
import sys
from pathlib import Path
import yaml

orig_cfg, eval_dataset, out_cfg, pred_dir, eval_dir, model_name = sys.argv[1:7]

with open(orig_cfg, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

overrides = data.setdefault("dataset_overrides", {})
eval_override = overrides.get(eval_dataset)
merge_override = overrides.get("merge")

if not isinstance(eval_override, dict):
    raise SystemExit(f"eval dataset override not found: {eval_dataset}")
if not isinstance(merge_override, dict):
    merge_override = {}

runs_dir = merge_override.get("runs_dir") or f"runs/merge/{model_name}"
qwen_dir = merge_override.get("qwen_dir") or f"result/detection_reasoning/merge/{model_name}_qwen"

merged_override = dict(eval_override)
merged_override["runs_dir"] = runs_dir
merged_override["prediction_dir"] = pred_dir
merged_override["eval_dir"] = eval_dir
merged_override["qwen_dir"] = qwen_dir
overrides[eval_dataset] = merged_override

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
PY
  local status=$?
  [[ "$status" -eq 0 ]] || return "$status"

  echo "$out_cfg"
}

coco_split_dir_name() {
  local split="${1,,}"
  case "$split" in
    val|validation) echo "valid" ;;
    *) echo "$split" ;;
  esac
}

render_overlay() {
  local family="$1"
  local model_cfg="$2"
  local eval_dataset="$3"
  local pred_dir="$4"

  local overlay_args=()
  if [[ "$SHOW_CONF" -eq 0 ]]; then
    overlay_args+=(--label-only)
  fi
  if [[ -n "$MAX_IMAGES" ]]; then
    overlay_args+=(--max-images "$MAX_IMAGES")
  fi

  if [[ "$family" == "rf_detr" || "$family" == "dino" ]]; then
    local coco_split
    coco_split="$(coco_split_dir_name "$SPLIT")"
    "$PY" scripts/render_coco_overlays.py \
      --model-config "$model_cfg" \
      --dataset "$eval_dataset" \
      --split "$SPLIT" \
      --pred-json "$pred_dir/${coco_split}/predictions_coco.json" \
      --output-dir "$pred_dir/${coco_split}/overlays" \
      "${overlay_args[@]}"
  else
    "$PY" scripts/render_yolo_overlays.py \
      --model-config "$model_cfg" \
      --dataset "$eval_dataset" \
      --split "$SPLIT" \
      --labels-dir "$pred_dir/labels" \
      --output-dir "$pred_dir/overlays" \
      "${overlay_args[@]}"
  fi
}

echo "[pred_eval_overlay_1cls_merge] split=${SPLIT} gpu=${GPU:-unset} py=${PY}"
echo "[pred_eval_overlay_1cls_merge] predict=${DO_PREDICT} eval=${DO_EVAL} overlay=${DO_OVERLAY} show_conf=${SHOW_CONF}"
echo "[run_1cls_merge_only] models=${MODELS}"
echo "[pred_eval_overlay_1cls_merge] train_dataset=${TRAIN_DATASET} targets=${TARGET_DATASETS[*]}"

for eval_dataset in "${TARGET_DATASETS[@]}"; do
  for model in $MODELS; do
    orig_cfg="config/model/${model}.yaml"
    [[ -f "$orig_cfg" ]] || { echo "[skip] config 없음: $model"; continue; }

    family="$(family_of "$model")" || { echo "[skip] family 읽기 실패: $model"; continue; }
    wt="$(weight_path "$model" "$family")"
    if [[ -z "$wt" ]]; then
      echo "[skip] merge 가중치 없음: eval=${eval_dataset} model=${model}"
      continue
    fi

    model_cfg="$(make_model_config_for_target "$model" "$eval_dataset")" \
      || { echo "[skip] 임시 config 생성 실패: eval=${eval_dataset} model=${model}"; continue; }
    [[ -f "$model_cfg" ]] \
      || { echo "[skip] model config 없음: ${model_cfg}"; continue; }

    pred_dir="$(dataset_override_field "$model_cfg" "$eval_dataset" "prediction_dir")" \
      || { echo "[skip] prediction_dir 읽기 실패: eval=${eval_dataset} model=${model}"; continue; }
    eval_dir="$(dataset_override_field "$model_cfg" "$eval_dataset" "eval_dir")" \
      || { echo "[skip] eval_dir 읽기 실패: eval=${eval_dataset} model=${model}"; continue; }
    [[ -n "$pred_dir" && -n "$eval_dir" ]] \
      || { echo "[skip] prediction/eval dir 비어 있음: eval=${eval_dataset} model=${model}"; continue; }

    echo "=================================================================="
    echo "[run] train=${TRAIN_DATASET} eval=${eval_dataset} model=${model} family=${family}"
    echo "[run] weights=${wt}"
    echo "[run] prediction_dir=${pred_dir}"
    echo "[run] eval_dir=${eval_dir}"

    if [[ "$DO_PREDICT" -eq 1 ]]; then
      echo "[predict] train=${TRAIN_DATASET} eval=${eval_dataset} model=${model}"
      if [[ "$family" == "rf_detr" || "$family" == "dino" ]]; then
        mkdir -p "$pred_dir"
        CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/train_model.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --stage predict \
          --weights "$wt" \
          --source "$SPLIT" \
          || { echo "[fail] predict 실패: eval=${eval_dataset} model=${model}"; continue; }
      else
        CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/predict_yolo_labels.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --weights "$wt" \
          --source "$SPLIT" \
          --output-dir "$pred_dir" \
          || { echo "[fail] predict 실패: eval=${eval_dataset} model=${model}"; continue; }
      fi
    fi

    if [[ "$DO_EVAL" -eq 1 ]]; then
      echo "[eval] train=${TRAIN_DATASET} eval=${eval_dataset} model=${model}"
      if [[ "$family" == "rf_detr" || "$family" == "dino" ]]; then
        coco_split="$(coco_split_dir_name "$SPLIT")"
        "$PY" scripts/evaluate_coco_predictions.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --split "$SPLIT" \
          --pred-coco "$pred_dir/${coco_split}/predictions_coco.json" \
          --output-dir "$eval_dir" \
          || { echo "[warn] eval 실패: eval=${eval_dataset} model=${model}"; }
      else
        "$PY" scripts/evaluate_yolo_predictions.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --split "$SPLIT" \
          --pred-labels-dir "$pred_dir/labels" \
          --output-dir "$eval_dir" \
          || { echo "[warn] eval 실패: eval=${eval_dataset} model=${model}"; }
      fi
    fi

    if [[ "$DO_OVERLAY" -eq 1 ]]; then
      echo "[overlay] train=${TRAIN_DATASET} eval=${eval_dataset} model=${model}"
      render_overlay "$family" "$model_cfg" "$eval_dataset" "$pred_dir" \
        || echo "[warn] overlay 실패: eval=${eval_dataset} model=${model}"
    fi
  done
done

echo "=================================================================="
echo "[run_1cls_merge_only] 완료"
