#!/usr/bin/env bash
#
# train_3cls_2cls.sh — 3cls + 2cls 모델 학습 (+선택적으로 eval / overlay)
#
# yolo11/12 · yolo_world · rt_detr · rf_detr · dino 를 3cls + 2cls로 학습한다.
# 각 모델 학습이 끝나면 --eval / --overlay 로 평가·시각화까지 이어서 할 수 있다.
# (집계 표는 만들지 않는다 — 모델별 결과만 남긴다. 비교표는 scripts/benchmark.sh 담당.)
#
# 사용 예:
#   ./scripts/train_3cls_2cls.sh                          # 학습만 (기존 동작)
#   ./scripts/train_3cls_2cls.sh --eval --overlay         # 학습 + 평가 + 시각화
#   TRAIN_GPU=0 ./scripts/train_3cls_2cls.sh --eval
#   PYBIN=/home/dongyub/tomato_detect/.venv/bin/python ./scripts/train_3cls_2cls.sh --eval --overlay
#
# 옵션:
#   --gpu N        사용할 GPU (기본 0, 환경변수 TRAIN_GPU)
#   --eval         각 모델 학습 직후 predict + 평가 (모델별 metrics 산출)
#   --overlay      각 모델 학습 직후 예측 overlay 이미지 생성
#   --split NAME   eval/overlay 대상 split (기본 test)
#   --python PATH  인터프리터 지정 (환경변수 PYBIN, 기본 python)
#   --models "..." 이 모델들만 실행 (재학습 범위 제한). 예: --models "dino_3cls dino_2cls"
#
#   * --eval / --overlay 는 torch/pandas/PIL 등이 필요하다. 부족하면 PYBIN 으로 venv 지정.

set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${TRAIN_GPU:-0}"
DO_EVAL=0
DO_OVERLAY=0
SPLIT=test
PY="${PYBIN:-python}"
MODELS_FILTER=""             # 비우면 전체. 지정하면 그 모델들만 (예: "dino_3cls dino_2cls")
DATASETS_FILTER=""           # 비우면 전체. 지정하면 그 데이터셋만 (예: "big little")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)      GPU="$2"; shift 2 ;;
    --eval)     DO_EVAL=1; shift ;;
    --overlay)  DO_OVERLAY=1; shift ;;
    --split)    SPLIT="$2"; shift 2 ;;
    --python)   PY="$2"; shift 2 ;;
    --models)   MODELS_FILTER="$2"; shift 2 ;;
    --datasets) DATASETS_FILTER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# $2(whitelist)에 있는 항목만 배열 $1 에 남긴다. whitelist 가 비면 그대로 둔다.
_filter() {
  [[ -z "$2" ]] && return 0
  local -n arr="$1"
  local kept=() item w
  for item in "${arr[@]}"; do
    for w in $2; do
      if [[ "$item" == "$w" ]]; then kept+=("$item"); break; fi
    done
  done
  arr=("${kept[@]+"${kept[@]}"}")
}

# eval/overlay 를 켰다면 인터프리터에 필요한 패키지가 있는지 먼저 확인한다.
if [[ "$DO_EVAL" -eq 1 || "$DO_OVERLAY" -eq 1 ]]; then
  if ! "$PY" -c "import torch, pandas, PIL" >/dev/null 2>&1; then
    echo "[train_3cls_2cls][경고] '$PY' 에 eval/overlay 용 패키지(torch/pandas/PIL)가 없습니다."
    echo "  PYBIN 으로 풀스택 venv 를 지정하세요. 예:"
    echo "    PYBIN=/home/dongyub/tomato_detect/.venv/bin/python $0 $*"
    exit 1
  fi
fi

MODELS_3CLS=(yolo11_3cls yolo12_3cls yolo_world_3cls rt_detr_3cls rf_detr_3cls dino_3cls)
DATASETS_3CLS=(big little tomatod merge)

MODELS_2CLS=(yolo11_2cls yolo12_2cls yolo_world_2cls rt_detr_2cls rf_detr_2cls dino_2cls)
DATASETS_2CLS=(custom_tomato)

_filter MODELS_3CLS "$MODELS_FILTER"
_filter MODELS_2CLS "$MODELS_FILTER"
# --datasets 는 치환(기본에 없는 merge_big 등도 지정 가능). 안 주면 기본 매트릭스 유지.
if [[ -n "$DATASETS_FILTER" ]]; then
  DATASETS_3CLS=($DATASETS_FILTER)
  DATASETS_2CLS=($DATASETS_FILTER)
fi

family_of() {
  "$PY" -c "import yaml;print((yaml.safe_load(open('config/model/$1.yaml')) or {}).get('family',''))" 2>/dev/null
}

# 학습 직후 predict + (eval / overlay). family 에 따라 yolo 경로 / coco 경로로 분기.
post_job() {
  local gpu="$1" model="$2" dataset="$3"
  [[ "$DO_EVAL" -eq 0 && "$DO_OVERLAY" -eq 0 ]] && return 0

  local family; family="$(family_of "$model")"
  local is_coco=0
  [[ "$family" == "rf_detr" || "$family" == "dino" ]] && is_coco=1

  # 1) predict (eval/overlay 가 공유)
  echo "[post] predict: $model x $dataset (family=$family)"
  if [[ "$is_coco" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_model.py --model "$model" --dataset "$dataset" --stage predict \
      || { echo "[warn] predict 실패: $model x $dataset (eval/overlay 건너뜀)"; return 0; }
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/predict_yolo_labels.py --model "$model" --dataset "$dataset" --source "$SPLIT" \
      || { echo "[warn] predict 실패: $model x $dataset (eval/overlay 건너뜀)"; return 0; }
  fi

  # 2) eval (모델별 metrics 만 — 집계 없음)
  if [[ "$DO_EVAL" -eq 1 ]]; then
    echo "[post] eval: $model x $dataset"
    if [[ "$is_coco" -eq 1 ]]; then
      "$PY" scripts/evaluate_coco_predictions.py --model "$model" --dataset "$dataset" --split "$SPLIT" \
        || echo "[warn] eval 실패: $model x $dataset"
    else
      "$PY" scripts/evaluate_yolo_predictions.py --model "$model" --dataset "$dataset" --split "$SPLIT" \
        || echo "[warn] eval 실패: $model x $dataset"
    fi
  fi

  # 3) overlay
  if [[ "$DO_OVERLAY" -eq 1 ]]; then
    echo "[post] overlay: $model x $dataset"
    if [[ "$is_coco" -eq 1 ]]; then
      "$PY" scripts/render_coco_overlays.py --model "$model" --dataset "$dataset" --split "$SPLIT" \
        || echo "[warn] overlay 실패: $model x $dataset"
    else
      "$PY" scripts/render_yolo_overlays.py --model "$model" --dataset "$dataset" --split "$SPLIT" \
        || echo "[warn] overlay 실패: $model x $dataset"
    fi
  fi
}

run_job() {
  local gpu="$1"
  local model="$2"
  local dataset="$3"

  echo "------------------------------------------------------------------"
  echo "[train] gpu=${gpu} model=${model} dataset=${dataset}"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_model.py --model "$model" --dataset "$dataset" --stage train; then
    echo "[fail] gpu=${gpu} model=${model} dataset=${dataset}"
    return 1
  fi
  echo "[done] train gpu=${gpu} model=${model} dataset=${dataset}"
  post_job "$gpu" "$model" "$dataset"
}

run_matrix() {
  local gpu="$1"
  local label="$2"
  local models_name="$3"
  local datasets_name="$4"
  local failures=0

  local -n models_ref="$models_name"
  local -n datasets_ref="$datasets_name"

  echo "=================================================================="
  echo "[group] ${label} gpu=${gpu}"
  echo "[group] models  : ${models_ref[*]}"
  echo "[group] datasets: ${datasets_ref[*]}"

  for model in "${models_ref[@]}"; do
    for dataset in "${datasets_ref[@]}"; do
      if ! run_job "$gpu" "$model" "$dataset"; then
        failures=1
      fi
    done
  done

  return "$failures"
}

echo "[train_3cls_2cls] gpu=${GPU} eval=${DO_EVAL} overlay=${DO_OVERLAY} split=${SPLIT} py=${PY}"
echo "[train_3cls_2cls] 3cls (6 models x 4 ds = 24) + 2cls (6 models x 1 ds = 6) = 30 jobs"

STATUS=0
run_matrix "$GPU" "3cls" MODELS_3CLS DATASETS_3CLS || STATUS=1
run_matrix "$GPU" "2cls" MODELS_2CLS DATASETS_2CLS || STATUS=1

echo "=================================================================="
if [[ "$STATUS" -eq 0 ]]; then
  echo "[train_3cls_2cls] all train jobs finished successfully"
else
  echo "[train_3cls_2cls] finished with failures"
fi

exit "$STATUS"
