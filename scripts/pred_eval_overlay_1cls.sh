#!/usr/bin/env bash
#
# pred_eval_overlay_1cls.sh
#   1cls self-only prediction + evaluation + overlay 실행 스크립트.
#
# 실행 조합(고정):
#   - big           -> big
#   - little        -> little
#   - tomatod       -> tomatod
#   - custom_tomato -> custom_tomato
#
# merge 가중치 cross-dataset 실행은 pred_eval_overlay_1cls.sh 로 분리
# custom_tomato 는 1cls model yaml 에 explicit dataset_override 가 없어도,
# config_loader 의 fallback 경로를 그대로 사용
#
# 예시:
#   BENCH_GPU=0 ./scripts/pred_eval_overlay_1cls.sh
#   BENCH_GPU=0 ./scripts/pred_eval_overlay_1cls.sh --models "yolo11_1cls rf_detr_1cls"
#   ./scripts/pred_eval_overlay_1cls.sh --no-predict --no-eval
#   ./scripts/pred_eval_overlay_1cls.sh --show-conf --max-images 50

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

PAIRS=(
  "big:big"
  "little:little"
  "tomatod:tomatod"
  "custom_tomato:custom_tomato"
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
  echo "[run_1cls_all_datasets][경고] '$PY' 에 필요한 패키지(yaml/torch/rfdetr/pandas/PIL)가 없습니다."
  echo "  올바른 venv 로 다시 실행하세요. 예:"
  echo "    PYBIN=/home/hyeonjin/detect-and-reason/.venv/bin/python $0"
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/run_1cls_all_datasets.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

config_summary() {
  local model_ref="$1"
  local dataset_ref="$2"
  "$PY" - <<'PY' "$model_ref" "$dataset_ref"
import sys
from pathlib import Path
import yaml

model_ref, dataset_ref = sys.argv[1:3]

with open(model_ref, "r", encoding="utf-8") as handle:
    model_cfg = yaml.safe_load(handle) or {}

family = model_cfg.get("family", "")
model_name = model_cfg.get("name") or Path(model_ref).stem
mode = model_cfg.get("mode", "detection_only")
override = ((model_cfg.get("dataset_overrides") or {}).get(dataset_ref) or {})

runs_dir = override.get("runs_dir") or f"runs/{dataset_ref}/{model_name}"
prediction_dir = override.get("prediction_dir") or f"result/{mode}/{dataset_ref}/{model_name}_prediction"
eval_dir = override.get("eval_dir") or f"result/{mode}/{dataset_ref}/{model_name}_eval"

print(family)
print(runs_dir)
print(prediction_dir)
print(eval_dir)
PY
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
  local pred_dir="result/detection_reasoning/${train_dataset}/${eval_dataset}/${model}_${eval_dataset}_prediction"
  local eval_dir="result/detection_reasoning/${train_dataset}/${eval_dataset}/${model}_${eval_dataset}_eval"

  "$PY" - <<'PY' "$orig_cfg" "$train_dataset" "$eval_dataset" "$out_cfg" "$pred_dir" "$eval_dir"
import sys
from pathlib import Path
import yaml

orig_cfg, train_dataset, eval_dataset, out_cfg, pred_dir, eval_dir = sys.argv[1:7]

with open(orig_cfg, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}

overrides = data.setdefault("dataset_overrides", {})
train_override = overrides.get(train_dataset)
eval_override = overrides.get(eval_dataset)
model_name = data.get("name") or Path(orig_cfg).stem

if not isinstance(eval_override, dict):
    raise SystemExit(f"eval dataset override not found: {eval_dataset}")

merged_override = dict(eval_override)
train_runs_dir = (
    train_override.get("runs_dir")
    if isinstance(train_override, dict)
    else f"runs/{train_dataset}/{model_name}"
)
merged_override["runs_dir"] = train_runs_dir
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

weight_path() {
  local runs_dir="$1"
  local family="$2"

  if [[ "$family" == "rf_detr" ]]; then
    for c in \
      "${runs_dir}/checkpoint_best_ema.pth" \
      "${runs_dir}/checkpoint_best_regular.pth" \
      "${runs_dir}/checkpoint.pth"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  else
    for c in \
      "${runs_dir}/weights/best.pt" \
      "${runs_dir}/weights/last.pt"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  fi

  echo ""
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

  if [[ "$family" == "rf_detr" || "$family" == "dino" ]]; then
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

echo "[run_1cls_all_datasets] split=${SPLIT} gpu=${GPU:-unset} py=${PY}"
echo "[run_1cls_all_datasets] predict=${DO_PREDICT} eval=${DO_EVAL} overlay=${DO_OVERLAY} show_conf=${SHOW_CONF}"
echo "[run_1cls_all_datasets] models=${MODELS}"
echo "[run_1cls_all_datasets] pairs=${PAIRS[*]}"

for pair in "${PAIRS[@]}"; do
  train_dataset="${pair%%:*}"
  eval_dataset="${pair##*:}"

  for model in $MODELS; do
    model_cfg="config/model/${model}.yaml"
    [[ -f "$model_cfg" ]] || { echo "[skip] config 없음: $model"; continue; }

    run_cfg="$(make_model_config_for_pair "$model" "$train_dataset" "$eval_dataset")" \
      || { echo "[skip] 실행용 config 생성 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }

    mapfile -t config_info < <(config_summary "$run_cfg" "$eval_dataset") \
      || { echo "[skip] config 해석 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
    [[ "${#config_info[@]}" -ge 4 ]] \
      || { echo "[skip] config info 부족: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }

    family="${config_info[0]}"
    runs_dir="${config_info[1]}"
    pred_dir="${config_info[2]}"
    eval_dir="${config_info[3]}"

    wt="$(weight_path "$runs_dir" "$family")"
    if [[ -z "$wt" ]]; then
      echo "[skip] 가중치 없음: train=${train_dataset} eval=${eval_dataset} model=${model} runs_dir=${runs_dir}"
      continue
    fi

    echo "=================================================================="
    echo "[run] train=${train_dataset} eval=${eval_dataset} model=${model} family=${family}"
    echo "[run] runs_dir=${runs_dir}"
    echo "[run] weights=${wt}"
    echo "[run] prediction_dir=${pred_dir}"
    echo "[run] eval_dir=${eval_dir}"

    if [[ "$DO_PREDICT" -eq 1 ]]; then
      echo "[predict] train=${train_dataset} eval=${eval_dataset} model=${model}"
      if [[ "$family" == "rf_detr" || "$family" == "dino" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/train_model.py \
          --model-config "$run_cfg" \
          --dataset "$eval_dataset" \
          --stage predict \
          --weights "$wt" \
          --source "$SPLIT" \
          || { echo "[fail] predict 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
      else
        CUDA_VISIBLE_DEVICES="${GPU:-}" "$PY" scripts/predict_yolo_labels.py \
          --model-config "$run_cfg" \
          --dataset "$eval_dataset" \
          --weights "$wt" \
          --source "$SPLIT" \
          || { echo "[fail] predict 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; continue; }
      fi
    fi

    if [[ "$DO_EVAL" -eq 1 ]]; then
      echo "[eval] train=${train_dataset} eval=${eval_dataset} model=${model}"
      if [[ "$family" == "rf_detr" || "$family" == "dino" ]]; then
        "$PY" scripts/evaluate_coco_predictions.py \
          --model-config "$run_cfg" \
          --dataset "$eval_dataset" \
          --split "$SPLIT" \
          --output-dir "$eval_dir" \
          || { echo "[warn] eval 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; }
      else
        "$PY" scripts/evaluate_yolo_predictions.py \
          --model-config "$run_cfg" \
          --dataset "$eval_dataset" \
          --split "$SPLIT" \
          --output-dir "$eval_dir" \
          || { echo "[warn] eval 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"; }
      fi
    fi

    if [[ "$DO_OVERLAY" -eq 1 ]]; then
      echo "[overlay] train=${train_dataset} eval=${eval_dataset} model=${model}"
      render_overlay "$family" "$run_cfg" "$eval_dataset" \
        || echo "[warn] overlay 실패: train=${train_dataset} eval=${eval_dataset} model=${model}"
    fi
  done
done

echo "=================================================================="
echo "[run_1cls_all_datasets] 완료"
