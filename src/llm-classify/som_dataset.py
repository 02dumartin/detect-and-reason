"""DaR의 config/dataset 컨벤션으로 SoM VLM 학습·평가용 GT를 읽어온다.

원본 llm-classify(`llm-classify/som_common.py`)는 데이터셋마다 경로를 하드코딩한
`DATASETS` dict를 썼는데, 여기선 이미 있는 DaR 데이터(`data/coco/<name>_3cls/`)와
config_loader의 런타임 해석을 그대로 재사용한다 — 새 데이터셋을 추가할 때
이 파일을 안 건드려도 되게 하기 위함(CLAUDE.md: 경로 하드코딩 금지).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config_loader import get_project_root, load_yaml, resolve_dataset_config_path, resolve_runtime_config

_SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


def _xyxy(bbox: list[float]) -> list[int]:
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def resolve_ripeness_class_mode(dataset: str) -> str:
    """이 데이터셋에 3cls(fully-ripe/semi-ripe/unripe) variant가 있으면 "3cls", 없고
    2cls(ripe/unripe, 예: custom_tomato/rob2pheno)만 있으면 "2cls"를 반환한다.

    데이터셋 이름을 하드코딩하지 않고 `config/dataset/<dataset>.yaml`의 실제 variants
    키만 보고 판단한다 (CLAUDE.md: 경로/데이터셋 하드코딩 금지).
    """
    variants = load_yaml(resolve_dataset_config_path(dataset)).get("variants", {})
    if "3cls" in variants:
        return "3cls"
    if "2cls" in variants:
        return "2cls"
    raise ValueError(f"dataset {dataset!r}에 3cls/2cls ripeness variant가 없습니다 (variants: {sorted(variants)})")


def resolve_class_list(dataset: str) -> tuple[str, ...]:
    """`som_common.set_class_list()`에 넘길 canonical ripeness 클래스 튜플."""
    return ("fully-ripe", "semi-ripe", "unripe") if resolve_ripeness_class_mode(dataset) == "3cls" else ("fully-ripe", "unripe")


def resolve_3cls_paths(dataset: str) -> tuple[Path, dict[int, str]]:
    """dataset 이름으로 (rfdetr_dir, {class_id: class_name}) 을 얻는다.

    데이터셋에 3cls variant가 있으면 `rf_detr_3cls`, 2cls만 있으면(custom_tomato/rob2pheno)
    `rf_detr_2cls` 기준으로 runtime_cfg를 풀어서 rfdetr_dir/class_names를 가져온다 —
    COCO GT는 어느 모델을 학습하든 데이터셋당 하나만 존재하므로 안전하다. 이름은
    `resolve_3cls_paths`로 남겨뒀지만(호출부가 많음) 실제로는 2cls도 처리한다.
    """
    class_mode = resolve_ripeness_class_mode(dataset)
    runtime_cfg = resolve_runtime_config(model_ref=f"rf_detr_{class_mode}", dataset_ref=dataset, stage="predict")
    return Path(runtime_cfg["dataset"]["rfdetr_dir"]), runtime_cfg["class_names"]


def resolve_1cls_weight(dataset: str, family: str = "yolo11") -> str:
    """1cls proposal 검출기(YOLO 계열)의 학습된 가중치 경로를 찾는다."""
    runtime_cfg = resolve_runtime_config(model_ref=f"{family}_1cls", dataset_ref=dataset, stage="predict")
    weights = runtime_cfg["paths"]["weights"]
    if not weights:
        raise FileNotFoundError(f"{family}_1cls weight not found for dataset={dataset}")
    return str(weights)


def load_split(dataset: str, split: str) -> list[dict[str, Any]]:
    """COCO GT를 `{"path","boxes","labels","wh"}` 아이템 리스트로 읽는다.

    boxes는 xyxy 정수, labels는 그 데이터셋의 3cls GT 클래스명 문자열
    (예: "fully ripened tomato") 그대로 사용한다 — llm-classify처럼 ripe/semi_ripe/unripe로
    다시 매핑하지 않고 DaR 전체에서 쓰는 이름 그대로 맞춘다.
    """
    import json

    rfdetr_dir, class_names = resolve_3cls_paths(dataset)
    split_dir = rfdetr_dir / _SPLIT_DIR[split]
    ann = json.loads((split_dir / "_annotations.coco.json").read_text())

    images = {img["id"]: img for img in ann["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for a in ann["annotations"]:
        by_image[a["image_id"]].append(a)

    items = []
    for image_id, anns in by_image.items():
        info = images[image_id]
        img_path = split_dir / info["file_name"]
        if not img_path.exists():
            continue
        items.append(
            {
                "path": str(img_path),
                "boxes": [_xyxy(a["bbox"]) for a in anns],
                "labels": [class_names[int(a["category_id"])] for a in anns],
                "wh": (info["width"], info["height"]),
            }
        )
    return items


def _nms_merge(boxes: list[list[int]], thr: float = 0.7) -> list[list[int]]:
    from src.evaluation.llm_classify_metric import iou_xyxy

    ordered = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep: list[list[int]] = []
    for b in ordered:
        if all(iou_xyxy(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def collect_proposals(
    dataset: str,
    items: list[dict[str, Any]],
    cache_path: Path,
    family: str = "yolo11",
    conf: float = 0.1,
    max_n: int = 20,
    device: int | str = 0,
) -> dict[str, list[list[int]]]:
    """학습된 1cls 검출기(YOLO 계열)로 낮은 conf proposal 박스를 뽑는다 (SoM 협업의 dedup 과제용).

    llm-classify의 `collect_yolo1`을 포팅 — 다른 점은 DaR에 이미 학습돼 있는 1cls
    가중치(`resolve_1cls_weight`)를 그대로 쓴다는 것.

    ultralytics(`model.predict(..., device=...)`)가 내부적으로 `os.environ["CUDA_VISIBLE_DEVICES"]`를
    자기가 받은 device 값으로 덮어써버려서(2026-09-13 확인), `CUDA_VISIBLE_DEVICES=N`으로 GPU를
    하나만 보이게 제한해 둔 프로세스에서 이 함수를 부르면 그 뒤로 로드되는 HF 모델이 전혀 다른
    (제한 전) 물리 GPU로 새버리는 문제가 있었다 — sft_som.py/rl_som.py가 학습 전에 이 함수를
    부르고 그다음 VLM을 로드하는 순서라 특히 치명적이었다. YOLO 호출 전후로 env var를
    저장/복원해서 프로세스의 나머지 부분에 영향이 새어나가지 않게 막는다.
    """
    import json
    import os

    import numpy as np
    from ultralytics import YOLO

    from som_common import open_rgb

    cache: dict[str, list[list[int]]] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    need = [it for it in items if it["path"] not in cache]
    if not need:
        return cache

    saved_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        import torch

        model = YOLO(resolve_1cls_weight(dataset, family=family))
        for item in need:
            im = open_rgb(item["path"])
            arr = np.ascontiguousarray(np.array(im)[:, :, ::-1])
            result = model.predict(arr, imgsz=640, verbose=False, conf=conf, iou=0.5, agnostic_nms=True, device=device)[0]
            if result.boxes is None or len(result.boxes) == 0:
                cache[item["path"]] = []
                continue
            xy = result.boxes.xyxy.cpu().tolist()
            cf = result.boxes.conf.cpu().tolist()
            sx = item["wh"][0] / max(1, im.size[0])
            sy = item["wh"][1] / max(1, im.size[1])
            scored = sorted(
                (([int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)], c) for (x1, y1, x2, y2), c in zip(xy, cf)),
                key=lambda pair: -pair[1],
            )
            cache[item["path"]] = _nms_merge([box for box, _ in scored[:max_n]])
        del model
    finally:
        if saved_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda_visible_devices
        # env var 복원만으로는 부족하다 — ultralytics가 predict 중 실제 cudaSetDevice()까지
        # 호출해 버려서 프로세스의 "현재 CUDA device" 자체가 바뀌어 있다(2026-09-13 확인:
        # env var는 복원돼도 mem_get_info(0)이 계속 다른 물리 GPU를 가리킴). 명시적으로
        # 되돌린다.
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    cache_path.write_text(json.dumps(cache))
    return cache


def load_proposals_from_model_predictions(
    dataset: str,
    items: list[dict[str, Any]],
    model: str = "rf_detr_1cls",
    split: str = "test",
    conf_threshold: float = 0.0,
) -> dict[str, list[list[int]]]:
    """이미 predict 돌려둔 1cls COCO 예측(predictions_coco.json)을 proposal로 재사용한다.

    `train_model.py --model {model} --dataset {dataset} --stage predict`로 생성된 예측을
    그대로 쓴다 — YOLO를 라이브로 다시 돌리는 `collect_proposals`의 대안. zero-shot SoM
    실험에서 이미 만들어둔 RF-DETR 1cls 예측을 proposal로 쓰고 싶을 때 사용한다.
    """
    import json

    runtime_cfg = resolve_runtime_config(model_ref=model, dataset_ref=dataset, stage="predict")
    split_dir_name = _SPLIT_DIR[split]
    gt_json_path = Path(runtime_cfg["dataset"]["rfdetr_dir"]) / split_dir_name / "_annotations.coco.json"
    pred_json_path = Path(runtime_cfg["paths"]["prediction_dir"]) / split_dir_name / "predictions_coco.json"
    if not pred_json_path.is_file():
        raise FileNotFoundError(
            f"예측이 없습니다: {pred_json_path}\n"
            f"먼저 predict를 돌리세요: python scripts/train_model.py --model {model} --dataset {dataset} --stage predict --source {split}"
        )

    gt = json.loads(gt_json_path.read_text())
    id_to_name = {img["id"]: Path(img["file_name"]).name for img in gt["images"]}
    name_to_path = {Path(it["path"]).name: it["path"] for it in items}

    preds = json.loads(pred_json_path.read_text())
    boxes_by_path: dict[str, list[list[int]]] = defaultdict(list)
    for pred in preds:
        if pred.get("score", 1.0) < conf_threshold:
            continue
        name = id_to_name.get(int(pred["image_id"]))
        path = name_to_path.get(name)
        if path is None:
            continue
        x, y, w, h = pred["bbox"]
        boxes_by_path[path].append([int(x), int(y), int(x + w), int(y + h)])
    return dict(boxes_by_path)


def load_proposals_from_yolo_txt_predictions(
    dataset: str,
    items: list[dict[str, Any]],
    model: str,
    split: str = "test",
    conf_threshold: float = 0.0,
) -> dict[str, list[list[int]]]:
    """이미 predict 돌려둔 1cls YOLO txt 예측(`labels/*.txt`)을 proposal로 재사용한다.

    `load_proposals_from_model_predictions`의 YOLO 계열(yolo11/yolo12/yolo_world/rt_detr)
    버전 — 이 모델들은 predictions_coco.json이 아니라 `labels/` 아래 txt로 예측을 낸다
    (2026-09-13, family로 자동 분기하는 `load_proposals_from_any_model_predictions` 참고).
    """
    from src.evaluation.yolo_label_io import load_yolo_label_rows, resolve_prediction_labels_dir, yolo_row_to_detection

    runtime_cfg = resolve_runtime_config(model_ref=model, dataset_ref=dataset, stage="predict")
    labels_dir = resolve_prediction_labels_dir(runtime_cfg["paths"]["prediction_dir"])
    wh_by_path = {it["path"]: it["wh"] for it in items}
    stem_to_path = {Path(it["path"]).stem: it["path"] for it in items}

    boxes_by_path: dict[str, list[list[int]]] = defaultdict(list)
    for stem, path in stem_to_path.items():
        rows = load_yolo_label_rows(labels_dir / f"{stem}.txt")
        width, height = wh_by_path[path]
        for row in rows:
            _label, xyxy, score = yolo_row_to_detection(row, width, height)
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = xyxy
            boxes_by_path[path].append([int(x1), int(y1), int(x2), int(y2)])
    return dict(boxes_by_path)


def load_proposals_from_any_model_predictions(
    dataset: str,
    items: list[dict[str, Any]],
    model: str,
    split: str = "test",
    conf_threshold: float = 0.0,
) -> dict[str, list[list[int]]]:
    """모델의 family(COCO json 계열 vs YOLO txt 계열)를 보고 알맞은 로더로 위임한다."""
    runtime_cfg = resolve_runtime_config(model_ref=model, dataset_ref=dataset, stage="predict")
    if runtime_cfg.get("family") in {"rf_detr", "dino"}:
        return load_proposals_from_model_predictions(dataset, items, model=model, split=split, conf_threshold=conf_threshold)
    return load_proposals_from_yolo_txt_predictions(dataset, items, model=model, split=split, conf_threshold=conf_threshold)


def default_output_dir(dataset: str, run_tag: str, exp_dir: str | None = None) -> Path:
    """실험 산출물 저장 위치. DaR의 exp/ 컨벤션을 따른다.

    `exp/{exp_dir}/{dataset}/{run_tag}/` — 데이터셋별로 날짜 폴더를 따로 만들지 않고
    하나의 `{exp_dir}/` 아래 데이터셋 서브폴더로 묶는다. `exp_dir`를 안 주면
    `{오늘 날짜}_vlm-som`(기존 기본값)을 쓴다.
    """
    from datetime import date

    root = get_project_root()
    stamp = date.today().strftime("%y%m%d")
    out = root / "exp" / (exp_dir or f"{stamp}_vlm-som") / dataset / run_tag
    out.mkdir(parents=True, exist_ok=True)
    return out
