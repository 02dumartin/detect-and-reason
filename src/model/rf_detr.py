from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

_SPLIT_ALIASES = {"val": "valid", "validation": "valid"}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
class ModelBuildError(Exception):
    pass


def resolve_rfdetr_resolution(train_cfg: dict) -> int:
    """Match ultralytics ``imgsz`` naming; ``rfdetr`` API uses ``resolution``."""
    value = train_cfg.get("resolution") or train_cfg.get("imgsz")
    return int(value) if value is not None else 640


# 체크포인트 args 에서 읽어 생성자에 그대로 넘길 아키텍처 필드.
#   rfdetr 패키지 버전이 바뀌어 같은 model_size 가 다른 구조를 만들 때
#   (예: 학습 당시 Large=dinov2_windowed_small(384) → 현재 Large=windowed_base(768)),
#   학습된 체크포인트의 실제 구조로 빌드해 로드 실패를 막는다.
_ARCH_SCALAR_KEYS = {
    "encoder": str,
    "hidden_dim": int,
    "sa_nheads": int,
    "ca_nheads": int,
    "dec_layers": int,
    "num_queries": int,
    "num_windows": int,
    "patch_size": int,
    "positional_encoding_size": int,
}
_ARCH_LIST_KEYS = ("out_feature_indexes", "projector_scale")


def rfdetr_arch_from_checkpoint(weight_path: Union[str, Path]) -> dict[str, Any]:
    """rfdetr 체크포인트(.pth=zip)의 pickle 에서 아키텍처 인자를 torch 없이 읽는다.

    실패하면 빈 dict 를 돌려 호출부가 기존 동작(config size)으로 fallback 하게 한다.
    """
    import io
    import pickletools
    import zipfile

    try:
        archive = zipfile.ZipFile(str(weight_path))
        pkl_names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if not pkl_names:
            return {}
        blob = archive.read(pkl_names[0])
    except Exception:
        return {}

    tokens: list[tuple[str, Any]] = []
    try:
        for op, arg, _ in pickletools.genops(io.BytesIO(blob)):
            name = op.name
            if name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "SHORT_BINSTRING"):
                tokens.append(("s", arg))
            elif name in ("BININT", "BININT1", "BININT2", "LONG1"):
                tokens.append(("i", arg))
    except Exception:
        return {}

    out: dict[str, Any] = {}
    total = len(tokens)
    for idx, (kind, value) in enumerate(tokens):
        if kind != "s":
            continue
        if value in _ARCH_SCALAR_KEYS and value not in out and idx + 1 < total:
            try:
                out[value] = _ARCH_SCALAR_KEYS[value](tokens[idx + 1][1])
            except Exception:
                pass
        elif value == "out_feature_indexes" and value not in out:
            vals = []
            j = idx + 1
            while j < total and tokens[j][0] == "i":
                vals.append(int(tokens[j][1]))
                j += 1
            if vals:
                out[value] = vals
        elif value == "projector_scale" and value not in out:
            vals = []
            j = idx + 1
            while j < total and tokens[j][0] == "s" and str(tokens[j][1]).upper().startswith("P") and tokens[j][1][1:].isdigit():
                vals.append(str(tokens[j][1]))
                j += 1
            if vals:
                out[value] = vals
    return out


def build_rf_detr_model(runtime_cfg: dict) -> Any:
    model_cfg = runtime_cfg.get("model", {})
    train_cfg = runtime_cfg.get("train", {})
    size = str(model_cfg.get("model_size", "large")).lower()
    weight_ref = runtime_cfg["paths"].get("weights")
    resolution = resolve_rfdetr_resolution(train_cfg)

    # validate/predict: 학습된 체크포인트의 실제 구조로 빌드해야 가중치가 로드된다.
    arch_override: dict[str, Any] = {}
    if runtime_cfg.get("stage") in {"validate", "predict"} and weight_ref and Path(str(weight_ref)).is_file():
        arch_override = rfdetr_arch_from_checkpoint(weight_ref)
        if arch_override:
            # windowed_small 계열이면 carrier 를 base 로 (large=windowed_base 기본값 회피)
            if "small" in str(arch_override.get("encoder", "")):
                size = "base"
            logger.info("[RF-DETR] 체크포인트 아키텍처로 빌드: size=%s override=%s", size, arch_override)

    return build_model(
        size=size,
        pretrain_weights=weight_ref,
        resolution=resolution,
        arch_override=arch_override or None,
    )


def build_model(
    size: str,
    pretrain_weights: Optional[Union[str, Path]] = None,
    num_classes: Optional[int] = None,
    resolution: Optional[int] = None,
    arch_override: Optional[dict[str, Any]] = None,
) -> Any:
    try:
        from rfdetr import (
            RFDETRBase,
            RFDETRLarge,
            RFDETRMedium,
            RFDETRNano,
            RFDETRSmall,
        )
    except ImportError as exc:
        raise ModelBuildError(f"rfdetr package is not installed: {exc}") from exc

    model_map = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }

    model_class = model_map.get(size.lower())
    if model_class is None:
        raise ModelBuildError(f"Unknown RF-DETR model size: {size}. Use {list(model_map)}")

    kwargs: dict[str, Any] = {}
    if pretrain_weights:
        weight_path = Path(pretrain_weights)
        if weight_path.is_file():
            kwargs["pretrain_weights"] = str(weight_path)
        else:
            logger.warning(
                "RF-DETR weight file not found (%s); using package default checkpoint.",
                weight_path,
            )
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    if resolution is not None:
        kwargs["resolution"] = resolution

    if arch_override:
        # 체크포인트 구조를 명시적으로 강제 — 절대 pretrain_weights 를 떼지 않는다.
        kwargs.update(arch_override)
        try:
            return model_class(**kwargs)
        except TypeError as exc:
            raise ModelBuildError(
                f"RF-DETR 체크포인트 아키텍처로 빌드 실패 (rfdetr 버전이 {sorted(arch_override)} "
                f"인자를 지원하지 않음): {exc}"
            ) from exc

    return _safe_construct(model_class, **kwargs)


def _safe_construct(cls, **kwargs):
    if kwargs.get("num_classes") is None:
        kwargs.pop("num_classes", None)

    try:
        return cls(**kwargs)
    except TypeError:
        kwargs.pop("num_classes", None)
        try:
            return cls(**kwargs)
        except TypeError:
            kwargs.pop("pretrain_weights", None)
            return cls(**kwargs)


# ---------------------------------------------------------------------------
# Inference / COCO helpers (shared by validate & predict)
# ---------------------------------------------------------------------------
def normalize_rfdetr_split(split: str) -> str:
    return _SPLIT_ALIASES.get(split.strip().lower(), split.strip().lower())


def resolve_rfdetr_split_dir(rfdetr_dir: Path, split: str) -> Path:
    return rfdetr_dir / normalize_rfdetr_split(split)


def load_coco_annotations(
    ann_path: Path,
) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], List[Dict]]:
    with ann_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {int(img["id"]): img for img in coco.get("images", [])}
    annotations_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    return images, annotations_by_image, coco.get("categories", [])


def resolve_split_image_path(split_dir: Path, file_name: str) -> Optional[Path]:
    fname = file_name
    if fname.startswith("images/"):
        fname = fname.replace("images/", "", 1)
    for candidate in (split_dir / "images" / fname, split_dir / fname):
        if candidate.is_file():
            return candidate
    return None


def find_rfdetr_checkpoint(runs_dir: Path) -> Optional[Path]:
    for candidate in (
        runs_dir / "checkpoint_best_ema.pth",
        runs_dir / "checkpoint_best_regular.pth",
        runs_dir / "checkpoint.pth",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def get_predict_fn(model: Any):
    if hasattr(model, "predict"):
        return model.predict
    if hasattr(model, "model") and hasattr(model.model, "predict"):
        return model.model.predict
    raise RuntimeError("RF-DETR model has no predict method")


def set_eval_mode(model: Any) -> None:
    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()


def _tensors_to_numpy(boxes, scores, labels):
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().detach().numpy()
    if torch.is_tensor(scores):
        scores = scores.cpu().detach().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().detach().numpy()
    return boxes, scores, labels


def _extract_detection_fields(pred) -> Tuple[Optional[Any], Optional[Any], Optional[Any], bool]:
    if hasattr(pred, "xyxy") and hasattr(pred, "confidence") and hasattr(pred, "class_id"):
        return pred.xyxy, pred.confidence, pred.class_id, True
    if isinstance(pred, dict):
        boxes = pred.get("boxes") or pred.get("pred_boxes") or pred.get("bbox") or pred.get("xyxy")
        scores = (
            pred.get("scores")
            or pred.get("pred_scores")
            or pred.get("score")
            or pred.get("confidence")
        )
        labels = (
            pred.get("labels")
            or pred.get("pred_labels")
            or pred.get("label")
            or pred.get("class_id")
        )
        is_xyxy = "xyxy" in pred or boxes is pred.get("xyxy")
        return boxes, scores, labels, is_xyxy
    if isinstance(pred, (list, tuple)) and len(pred) >= 3:
        return pred[0], pred[1], pred[2], False
    return None, None, None, False


def parse_detection_xyxy(
    pred,
    conf_threshold: float = 0.0,
) -> Tuple[List[Tuple[float, float, float, float]], List[int], List[float]]:
    boxes, scores, labels, is_xyxy = _extract_detection_fields(pred)
    if boxes is None or scores is None or labels is None or len(boxes) == 0:
        return [], [], []

    boxes, scores, labels = _tensors_to_numpy(boxes, scores, labels)
    xyxy_list: List[Tuple[float, float, float, float]] = []
    lab_list: List[int] = []
    score_list: List[float] = []

    for index in range(len(boxes)):
        score = float(scores[index]) if scores[index] is not None else 0.0
        if score < conf_threshold:
            continue
        box = boxes[index]
        if len(box) != 4:
            continue
        if is_xyxy:
            x1, y1, x2, y2 = (float(value) for value in box)
        else:
            cx, cy, bw, bh = (float(value) for value in box)
            x1 = cx - bw * 0.5
            y1 = cy - bh * 0.5
            x2 = cx + bw * 0.5
            y2 = cy + bh * 0.5
        xyxy_list.append((x1, y1, x2, y2))
        lab_list.append(int(labels[index]))
        score_list.append(score)
    return xyxy_list, lab_list, score_list


def convert_predictions_to_coco(
    predictions: List[Any],
    image_ids: List[int],
    conf_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    coco_predictions: List[Dict[str, Any]] = []
    ann_id = 1

    for pred, image_id in zip(predictions, image_ids):
        xyxy_list, lab_list, score_list = parse_detection_xyxy(pred, conf_threshold)
        for (x1, y1, x2, y2), label, score in zip(xyxy_list, lab_list, score_list):
            x = max(0.0, x1)
            y = max(0.0, y1)
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            coco_predictions.append(
                {
                    "id": ann_id,
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [x, y, width, height],
                    "score": score,
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco_predictions


def run_split_inference(
    model: Any,
    split_dir: Path,
    *,
    batch_size: int,
    conf_threshold: float = 0.0,
) -> Tuple[List[Any], List[int], Path]:
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"RF-DETR annotation file not found: {ann_path}")

    images_dict, _, _ = load_coco_annotations(ann_path)
    image_paths: List[str] = []
    image_ids: List[int] = []

    for image_id, info in images_dict.items():
        image_path = resolve_split_image_path(split_dir, info["file_name"])
        if image_path is None:
            continue
        image_paths.append(str(image_path))
        image_ids.append(image_id)

    if not image_paths:
        raise FileNotFoundError(f"No images found under RF-DETR split: {split_dir}")

    predict_fn = get_predict_fn(model)
    predictions: List[Any] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_preds = predict_fn(batch_paths)
        if not isinstance(batch_preds, list):
            batch_preds = [batch_preds]
        predictions.extend(batch_preds)

    if conf_threshold > 0:
        for index, pred in enumerate(predictions):
            xyxy, labels, scores = parse_detection_xyxy(pred, conf_threshold)
            predictions[index] = {"xyxy": xyxy, "class_id": labels, "confidence": scores}

    return predictions, image_ids, ann_path


# ---------------------------------------------------------------------------
# Dataset preprocessing (Roboflow COCO layout for RF-DETR training)
# ---------------------------------------------------------------------------
class DatasetPreprocessor:
    """Prepare Roboflow COCO layout for RF-DETR training."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"RF-DETR dataset directory not found: {self.root}")

    def prepare(self) -> None:
        logger.info("[RF-DETR] Preprocessing dataset: %s", self.root)
        self._ensure_info_field()
        self._ensure_image_paths()
        self._ensure_contiguous_category_ids()

    def _ensure_info_field(self) -> None:
        for split in ("train", "valid", "test"):
            ann_path = self.root / split / "_annotations.coco.json"
            if not ann_path.is_file():
                continue
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            if "info" not in data:
                data["info"] = {"description": "RF-DETR dataset"}
                ann_path.write_text(json.dumps(data), encoding="utf-8")

    def _ensure_image_paths(self) -> None:
        for split in ("train", "valid", "test"):
            ann_path = self.root / split / "_annotations.coco.json"
            if not ann_path.is_file():
                continue

            data = json.loads(ann_path.read_text(encoding="utf-8"))
            images = data.get("images") or []
            if not images:
                continue

            split_root = self.root / split
            sample = str(images[0].get("file_name", ""))
            # rfdetr는 split_root + file_name 으로 이미지를 찾으므로, file_name 그대로
            # 직접 로드 가능한 경우(접두사 이미 포함 or 이미지가 split_root 바로 아래)에만 건너뛴다.
            # 이미지가 images/ 하위에 있는데 file_name에 접두사가 없으면 아래에서 보정한다.
            if (split_root / sample).is_file():
                continue

            prefix = "images"
            if not (split_root / prefix / Path(sample).name).is_file():
                continue

            for image in images:
                name = str(image.get("file_name", "")).replace("\\", "/")
                if not name.startswith(f"{prefix}/"):
                    image["file_name"] = f"{prefix}/{Path(name).name}"

            ann_path.write_text(json.dumps(data), encoding="utf-8")

    def _ensure_contiguous_category_ids(self) -> None:
        for split in ("train", "valid", "test"):
            ann_path = self.root / split / "_annotations.coco.json"
            if not ann_path.is_file():
                continue

            data = json.loads(ann_path.read_text(encoding="utf-8"))
            categories = sorted(data.get("categories", []), key=lambda item: int(item["id"]))
            if not categories:
                continue

            old_to_new = {int(category["id"]): index for index, category in enumerate(categories)}
            if list(old_to_new.values()) == list(range(len(categories))):
                continue

            for ann in data.get("annotations", []):
                ann["category_id"] = old_to_new[int(ann["category_id"])]
            data["categories"] = [
                {
                    "id": index,
                    "name": category["name"],
                    "supercategory": category.get("supercategory", "object"),
                }
                for index, category in enumerate(categories)
            ]
            ann_path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def validate_layout(root: Path) -> None:
        train_ann = root / "train" / "_annotations.coco.json"
        valid_ann = root / "valid" / "_annotations.coco.json"
        if not train_ann.is_file():
            raise FileNotFoundError(f"RF-DETR train annotations missing: {train_ann}")
        if not valid_ann.is_file():
            raise FileNotFoundError(f"RF-DETR valid annotations missing: {valid_ann}")
