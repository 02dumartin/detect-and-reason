#!/usr/bin/env bash
#
# benchmark.sh — 학습된 모델들을 한 번에 predict + eval 하고 비교 표를 만든다.
#
# 사용법 (프로젝트 루트에서):
#   BENCH_GPU=2 ./scripts/benchmark.sh --classes 3                 # 3cls 전체 (big/little/tomatod/merge)
#   BENCH_GPU=2 ./scripts/benchmark.sh --classes 1                 # +custom_tomato
#   BENCH_GPU=2 ./scripts/benchmark.sh --classes 2                 # custom_tomato (2cls)
#   ./scripts/benchmark.sh --classes 3 --no-predict                # 기존 예측 재사용(빠름)
#   ./scripts/benchmark.sh --classes 3 --models "yolo11_3cls rf_detr_3cls" --datasets "big merge"
#   ./scripts/benchmark.sh --classes 3 --tag rf --models "rf_detr_3cls"   # -> benchmark/rf_3cls/
#
# 옵션:
#   --tag NAME   출력 폴더 이름을 benchmark/NAME_<N>cls 로 고정 (생략 시 타임스탬프)
#
# 동작:
#   - 모델 config의 family를 읽어 predict/eval 경로를 자동 분기
#       * rf_detr / dino : train_model.py --stage predict  +  evaluate_coco_predictions.py (COCO json)
#       * 그 외(yolo/rt_detr/yolo_world) : predict_yolo_labels.py  +  evaluate_yolo_predictions.py
#   - 학습 가중치(best.pt / checkpoint_*.pth)가 없으면 해당 조합은 건너뛴다.
#   - 각 eval 결과(evaluation_results.json)를 한 폴더에 모아 표(md+csv)로 집계한다.

set -uo pipefail
# scripts/ 안에 있으므로 프로젝트 루트로 이동 (config/, runs/, scripts/ 경로 기준)
cd "$(dirname "$0")/.."

# ---- 기본값 ----
CLASSES=3
SPLIT=test
DO_PREDICT=1
MODELS=""
DATASETS=""
TAG=""                       # 출력 폴더 이름 prefix (예: rf -> benchmark/rf_3cls). 생략하면 타임스탬프.
GPU="${BENCH_GPU:-}"
IOU=""                        # 지정 시 모든 eval 에 --iou-threshold 로 강제 (config iou_default 무시)
EVAL_CONF=""                  # 지정 시 coco eval 에 --conf-threshold 로 예측 추가 필터 (DINO 등 미필터 예측 대응)
PREDICT_CONF=""               # 지정 시 predict 를 이 conf 로 실행 (mAP/best-F1 산출용 저conf 예: 0.001)
BESTF1=0                      # 1 이면 evaluate_bestf1.py 사용 (mAP=전체박스, P/R/Acc=best-F1 sweep)
SELECT_VAL=0                  # 1 이면 best-F1 conf 를 val 에서 선택→test 적용 (논문용; val 예측도 뽑음)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --classes)   CLASSES="$2"; shift 2 ;;
    --split)     SPLIT="$2"; shift 2 ;;
    --no-predict) DO_PREDICT=0; shift ;;
    --models)    MODELS="$2"; shift 2 ;;
    --datasets)  DATASETS="$2"; shift 2 ;;
    --tag)       TAG="$2"; shift 2 ;;
    --gpu)       GPU="$2"; shift 2 ;;
    --iou)       IOU="$2"; shift 2 ;;
    --eval-conf) EVAL_CONF="$2"; shift 2 ;;
    --predict-conf) PREDICT_CONF="$2"; shift 2 ;;
    --best-f1)   BESTF1=1; shift ;;
    --select-val) SELECT_VAL=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

# 모든 python 호출에 쓸 인터프리터. 풀스택(torch/rfdetr/ultralytics/pandas/...)을 갖춘
# venv 를 가리켜야 한다. 기본 python3 가 비어있으면 PYBIN 으로 지정:
#   PYBIN=/home/dongyub/tomato_detect/.venv/bin/python ./scripts/benchmark.sh ...
PY="${PYBIN:-python3}"
if ! "$PY" -c "import torch, rfdetr, pandas" >/dev/null 2>&1; then
  echo "[benchmark][경고] '$PY' 에 torch/rfdetr/pandas 가 없습니다."
  echo "  올바른 venv 로 다시 실행하세요. 예:"
  echo "    PYBIN=/home/dongyub/tomato_detect/.venv/bin/python $0 $*"
  echo "  또는:  source /home/dongyub/tomato_detect/.venv/bin/activate  후 재실행"
  exit 1
fi

# ---- 클래스별 기본 매트릭스 ----
if [[ -z "$MODELS" ]]; then
  case "$CLASSES" in
    3) MODELS="yolo11_3cls yolo12_3cls rt_detr_3cls yolo_world_3cls rf_detr_3cls dino_3cls" ;;
    1) MODELS="yolo11_1cls yolo12_1cls rt_detr_1cls yolo_world_1cls rf_detr_1cls dino_1cls" ;;
    2) MODELS="yolo11_2cls rf_detr_2cls dino_2cls" ;;
    *) echo "지원하지 않는 --classes: $CLASSES (3/1/2)"; exit 1 ;;
  esac
fi
if [[ -z "$DATASETS" ]]; then
  case "$CLASSES" in
    3) DATASETS="big little tomatod merge" ;;
    1) DATASETS="big little tomatod merge custom_tomato" ;;
    2) DATASETS="custom_tomato" ;;
  esac
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="benchmark/${TAG:-$TS}_${CLASSES}cls"
mkdir -p "$OUT_ROOT"
echo "[benchmark] classes=${CLASSES} split=${SPLIT} gpu=${GPU:-unset} predict=${DO_PREDICT}"
echo "[benchmark] models  : $MODELS"
echo "[benchmark] datasets: $DATASETS"
echo "[benchmark] out     : $OUT_ROOT"

family_of() {
  "$PY" -c "import yaml;print((yaml.safe_load(open('config/model/$1.yaml')) or {}).get('family',''))" 2>/dev/null
}

weight_path() {  # $1=model $2=dataset $3=family
  if [[ "$3" == "rf_detr" ]]; then
    for c in "runs/$2/$1/checkpoint_best_ema.pth" "runs/$2/$1/checkpoint_best_regular.pth" "runs/$2/$1/checkpoint.pth"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  else
    for c in "runs/$2/$1/weights/best.pt" "runs/$2/$1/weights/last.pt"; do
      [[ -f "$c" ]] && { echo "$c"; return; }
    done
  fi
  echo ""
}

for model in $MODELS; do
  [[ -f "config/model/$model.yaml" ]] || { echo "[skip] config 없음: $model"; continue; }
  family="$(family_of "$model")"
  for dataset in $DATASETS; do
    wt="$(weight_path "$model" "$dataset" "$family")"
    if [[ -z "$wt" ]]; then
      echo "[skip] 가중치 없음: $model x $dataset"
      continue
    fi
    out_dir="$OUT_ROOT/${model}__${dataset}"
    echo "=================================================================="
    echo "[run] $model x $dataset (family=$family) weights=$wt"

    is_coco=0
    [[ "$family" == "rf_detr" || "$family" == "dino" ]] && is_coco=1

    # --- val 예측 (best-F1 conf 를 val 에서 고르기 위함; --select-val 일 때만) ---
    #     coco 는 predict 시 prediction_dir 를 rmtree 하므로 val 을 먼저 뽑아 빼돌린 뒤 test 를 뽑는다.
    sel_args=()
    if [[ "$BESTF1" == "1" && "$SELECT_VAL" == "1" ]]; then
      if [[ "$is_coco" == "1" ]]; then
        if [[ "$DO_PREDICT" == "1" ]]; then
          pred_dir="$("$PY" -c "import autorootcwd; from src.config_loader import resolve_runtime_config as R; print(R(model_ref='$model',dataset_ref='$dataset',stage='predict')['paths']['prediction_dir'])" 2>/dev/null)"
          if "$PY" scripts/train_model.py --model "$model" --dataset "$dataset" --stage predict --source val ${PREDICT_CONF:+--conf "$PREDICT_CONF"}; then
            valjson="$(find "$pred_dir" -name predictions_coco.json 2>/dev/null | head -1)"
            [[ -n "$valjson" ]] && mkdir -p "$out_dir" && cp "$valjson" "$out_dir/valcoco.json"
          else
            echo "[warn] val predict 실패(coco): $model x $dataset — test-tuning 로 대체"
          fi
        fi
        [[ -f "$out_dir/valcoco.json" ]] && sel_args=(--select-split val --select-pred "$out_dir/valcoco.json")
      else
        if [[ "$DO_PREDICT" == "1" ]]; then
          "$PY" scripts/predict_yolo_labels.py --model "$model" --dataset "$dataset" --source val \
            ${PREDICT_CONF:+--conf "$PREDICT_CONF"} --output-dir "$out_dir/valpred" \
            || echo "[warn] val predict 실패(yolo): $model x $dataset — test-tuning 로 대체"
        fi
        [[ -d "$out_dir/valpred/labels" ]] && sel_args=(--select-split val --select-pred "$out_dir/valpred/labels")
      fi
    fi

    # --- test(보고) 예측 (family 별 경로 분기, --predict-conf 로 conf override) ---
    if [[ "$DO_PREDICT" == "1" ]]; then
      if [[ "$is_coco" == "1" ]]; then
        "$PY" scripts/train_model.py --model "$model" --dataset "$dataset" --stage predict \
          ${PREDICT_CONF:+--conf "$PREDICT_CONF"} || { echo "[fail] predict $model x $dataset"; continue; }
      else
        "$PY" scripts/predict_yolo_labels.py --model "$model" --dataset "$dataset" --source "$SPLIT" \
          ${PREDICT_CONF:+--conf "$PREDICT_CONF"} || { echo "[fail] predict $model x $dataset"; continue; }
      fi
    fi

    # --- eval ---
    if [[ "$BESTF1" == "1" ]]; then
      # mAP=전체박스, P/R/Acc=best-F1 (conf 는 val 에서 선택; sel_args 비면 test-tuning fallback)
      "$PY" scripts/evaluate_bestf1.py --model "$model" --dataset "$dataset" --split "$SPLIT" --output-dir "$out_dir" \
        ${IOU:+--iou-threshold "$IOU"} ${sel_args[@]+"${sel_args[@]}"} \
        || { echo "[fail] eval $model x $dataset"; continue; }
    elif [[ "$family" == "rf_detr" || "$family" == "dino" ]]; then
      "$PY" scripts/evaluate_coco_predictions.py --model "$model" --dataset "$dataset" --split "$SPLIT" --output-dir "$out_dir" \
        ${IOU:+--iou-threshold "$IOU"} ${EVAL_CONF:+--conf-threshold "$EVAL_CONF"} \
        || { echo "[fail] eval $model x $dataset"; continue; }
    else
      "$PY" scripts/evaluate_yolo_predictions.py --model "$model" --dataset "$dataset" --split "$SPLIT" --output-dir "$out_dir" \
        ${IOU:+--iou-threshold "$IOU"} \
        || { echo "[fail] eval $model x $dataset"; continue; }
    fi
  done
done

echo "=================================================================="
if [[ "$BESTF1" == "1" ]]; then
  echo "[benchmark] best-F1 모드: bestf1_results.json 들이 $OUT_ROOT 에 생성됨 (별도 집계는 evaluate_bestf1 출력 참고)"
else
  echo "[benchmark] 집계 중..."
  "$PY" src/aggregate_benchmark.py --dir "$OUT_ROOT"
fi
