#!/usr/bin/env bash
#
# pred_eval_overlay_3cls.sh
#   3cls self-only prediction + evaluation + overlay 실행 스크립트.
#
# 실행 조합(고정):
#   - big     -> big
#   - little  -> little
#   - tomatod -> tomatod
#
# merge 가중치 cross-dataset 실행은 pred_eval_overlay_3cls_merge.sh 로 분리했다.
#
# 예시:
#   BENCH_GPU=2 ./scripts/pred_eval_overlay_3cls.sh
#   BENCH_GPU=2 ./scripts/pred_eval_overlay_3cls.sh --models "yolo11_3cls rf_detr_3cls"
#   ./scripts/pred_eval_overlay_3cls.sh --no-predict --no-eval
#   ./scripts/pred_eval_overlay_3cls.sh --show-conf --max-images 50

set -uo pipefail

cd "$(dirname "$0")/.."

SPLIT=test
DO_PREDICT=1
DO_EVAL=1
DO_OVERLAY=1
SHOW_CONF=0
MAX_IMAGES=""
MODELS="yolo11_3cls yolo12_3cls yolo_world_3cls rt_detr_3cls rf_detr_3cls"
GPU="${BENCH_GPU:-}"
PY="${PYBIN:-python3}"

PAIRS=(
  "big:big"
  "little:little"
  "tomatod:tomatod"
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
  echo "[pred_eval_overlay_3cls][경고] '$PY' 에 필요한 패키지(yaml/torch/rfdetr/pandas/PIL)가 없습니다."
  echo "  올바른 venv 로 다시 실행하세요. 예:"
  echo "    PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python $0"
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/pred_eval_overlay_3cls.XXXXXX)"
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
  local train_dataset="$1"
  local model="$2"
  local family="$3"

  if [[ "$family" == "rf_detr" ]]; then
    for c in \
      "runs/${train_dataset}/${model}/checkpoint_best_ema.pth" \
      "runs/${train_dataset}/${model}/checkpoint_best_regular.pth" \
      "runs/${train_dataset}/${model}/checkpoint.pth"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  else
    for c in \
      "runs/${train_dataset}/${model}/weights/best.pt" \
      "runs/${train_dataset}/${model}/weights/last.pt"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  fi

  echo ""
}

make_model_config_for_pair() {
  local model="$1"
  local train_dataset="$2"
  local eval_dataset="$3"
  local orig_cfg="config/model/${model}.yaml"

  if [[ "$train_dataset" == "$eval_dataset" ]]; then
    echo "$orig_cfg"
    return
  fi

  local out_cfg="${TMP_DIR}/${model}__${train_dataset}__to__${eval_dataset}.yaml"
  local pred_dir="result/detection_only/${train_dataset}/${eval_dataset}/${model}_${eval_dataset}_prediction"
  local eval_dir="result/detection_only/${train_dataset}/${eval_dataset}/${model}_${eval_dataset}_eval"

  "$PY" - <<'PY' "$orig_cfg" "$train_dataset" "$eval_dataset" "$out_cfg" "$pred_dir" "$eval_dir"
import sys
import yaml

orig_cfg, train_dataset, eval_dataset, out_cfg, pred_dir, eval_dir = sys.argv[1:7]

with open(orig_cfg, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

overrides = data.setdefault("dataset_overrides", {})
train_override = overrides.get(train_dataset)
eval_override = overrides.get(eval_dataset)

if not isinstance(train_override, dict):
    raise SystemExit(f"train dataset override not found: {train_dataset}")
if not isinstance(eval_override, dict):
    raise SystemExit(f"eval dataset override not found: {eval_dataset}")

merged_override = dict(eval_override)
merged_override["runs_dir"] = train_override.get("runs_dir")
merged_override["prediction_dir"] = pred_dir
merged_override["eval_dir"] = eval_dir
overrides[eval_dataset] = merged_override

with open(out_cfg, "w", encoding="utf-8") as handle:
    yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
PY
  local status=$?
  [[ "$status" -eq 0 ]] || return "$status"

  echo "$out_cfg"
}

render_overlay() {
  local family="$1"
  local model_cfg="$2"
  local eval_dataset="$3"

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
      --dataset "$eval_dataset" \
      --split "$SPLIT" \
      "${overlay_args[@]}"
  else
    "$PY" scripts/render_yolo_overlays.py \
      --model-config "$model_cfg" \
      --dataset "$eval_dataset" \
      --split "$SPLIT" \
      "${overlay_args[@]}"
  fi
}

echo "[pred_eval_overlay_3cls] split=${SPLIT} gpu=${GPU:-unset} py=${PY}"
echo "[pred_eval_overlay_3cls] predict=${DO_PREDICT} eval=${DO_EVAL} overlay=${DO_OVERLAY} show_conf=${SHOW_CONF}"
echo "[pred_eval_overlay_3cls] models=${MODELS}"
echo "[pred_eval_overlay_3cls] pairs=${PAIRS[*]}"

for pair in "${PAIRS[@]}"; do
  train_dataset="${pair%%:*}"
  eval_dataset="${pair##*:}"

  for model in $MODELS; do
    orig_cfg="config/model/${model}.yaml"
    [[ -f "$orig_cfg" ]] || { echo "[skip] config 없음: $model"; continue; }

    family="$(family_of "$model")" || { echo "[skip] family 읽기 실패: $model"; continue; }
    wt="$(weight_path "$train_dataset" "$model" "$family")"
    if [[ -z "$wt" ]]; then
      echo "[skip] 가중치 없음: train=${train_dataset} eval=${eval_dataset} model=${model}"
      continue
    fi

    model_cfg="$(make_model_config_for_pair "$model" "$train_dataset" "$eval_dataset")" \
      || { echo "[skip] 임시 config 생성 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
    [[ -f "$model_cfg" ]] \
      || { echo "[skip] model config 없음: ${model_cfg}"; continue; }

    pred_dir="$(dataset_override_field "$model_cfg" "$eval_dataset" "prediction_dir")" \
      || { echo "[skip] prediction_dir 읽기 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
    eval_dir="$(dataset_override_field "$model_cfg" "$eval_dataset" "eval_dir")" \
      || { echo "[skip] eval_dir 읽기 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
    [[ -n "$pred_dir" && -n "$eval_dir" ]] \
      || { echo "[skip] prediction/eval dir 비어 있음: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }

    echo "=================================================================="
    echo "[run] train=${train_dataset} eval=${eval_dataset} model=${model} family=${family}"
    echo "[run] weights=${wt}"
    echo "[run] prediction_dir=${pred_dir}"
    echo "[run] eval_dir=${eval_dir}"

    if [[ "$DO_PREDICT" -eq 1 ]]; then
      echo "[predict] train=${train_dataset} eval=${eval_dataset} model=${model}"
      if [[ "$family" == "rf_detr" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/train_model.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --stage predict \
          --weights "$wt" \
          --source "$SPLIT" \
          || { echo "[fail] predict 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
      else
        CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/predict_yolo_labels.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --weights "$wt" \
          --source "$SPLIT" \
          || { echo "[fail] predict 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
      fi
    fi

    if [[ "$DO_EVAL" -eq 1 ]]; then
      echo "[eval] train=${train_dataset} eval=${eval_dataset} model=${model}"
      if [[ "$family" == "rf_detr" ]]; then
        "$PY" scripts/evaluate_coco_predictions.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --split "$SPLIT" \
          --output-dir "$eval_dir" \
          || { echo "[warn] eval 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; }
      else
        "$PY" scripts/evaluate_yolo_predictions.py \
          --model-config "$model_cfg" \
          --dataset "$eval_dataset" \
          --split "$SPLIT" \
          --output-dir "$eval_dir" \
          || { echo "[warn] eval 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; }
      fi
    fi

    if [[ "$DO_OVERLAY" -eq 1 ]]; then
      echo "[overlay] train=${train_dataset} eval=${eval_dataset} model=${model}"
      render_overlay "$family" "$model_cfg" "$eval_dataset" \
        || echo "[warn] overlay 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"
    fi
  done
done

echo "=================================================================="
echo "[pred_eval_overlay_3cls] 완료"
