from __future__ import annotations

from pathlib import Path

from src.config_loader import resolve_path
from src.model.zeroshot_common import (
    batched_nms,
    iter_source_images,
    load_image_size,
    phrase_to_two_class_id,
    prepare_prediction_dir,
    save_yolo_prediction_txt,
)


def run_grounding_dino_predict(runtime_cfg: dict) -> dict[str, str | int]:
    source_dir = Path(runtime_cfg["predict"]["resolved_source"])
    prediction_dir = prepare_prediction_dir(runtime_cfg["paths"]["prediction_dir"])
    labels_dir = prediction_dir / "labels"
    device = _resolve_device(runtime_cfg)

    config_path = resolve_gdino_config(
        runtime_cfg["model"].get("gdino_config"),
        project_root=runtime_cfg["project_root"],
    )
    if config_path is None:
        raise FileNotFoundError("Could not resolve Grounding DINO config path")

    weights_path = runtime_cfg["paths"]["weights"]
    if not weights_path:
        raise ValueError("Grounding DINO weights are missing")

    prompts = list(runtime_cfg.get("prompts") or [])
    if not prompts:
        raise ValueError("Grounding DINO zero-shot prompts are missing")

    box_threshold = float(runtime_cfg.get("predict", {}).get("conf") or 0.35)
    text_threshold = float(runtime_cfg.get("model", {}).get("text_threshold") or 0.25)
    nms_iou = float(runtime_cfg.get("predict", {}).get("resolved_iou") or 0.5)
    save_conf = bool(runtime_cfg.get("predict", {}).get("save_conf", True))

    model = build_gdino_model(config_path, weights_path, device)
    caption = ". ".join(prompts).strip()
    if caption and not caption.endswith("."):
        caption = f"{caption}."

    num_images = 0
    num_detections = 0
    for image_path in iter_source_images(source_dir):
        boxes_xyxy, scores, labels = predict_grounding_dino_image(
            model=model,
            image_path=image_path,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        boxes_xyxy, scores, labels = batched_nms(boxes_xyxy, scores, labels, nms_iou)

        image_width, image_height = load_image_size(image_path)
        save_yolo_prediction_txt(
            labels_dir=labels_dir,
            image_stem=image_path.stem,
            image_width=image_width,
            image_height=image_height,
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            labels=labels,
            save_conf=save_conf,
        )
        num_images += 1
        num_detections += len(labels)

    return {
        "prediction_dir": str(prediction_dir),
        "labels_dir": str(labels_dir),
        "num_images": num_images,
        "num_detections": num_detections,
    }


def build_gdino_model(config_path: str | Path, weights_path: str | Path, device: str):
    try:
        try:
            from groundeddino_vl.utils.inference import load_model
        except Exception:
            from groundingdino.util.inference import load_model
    except Exception as exc:
        raise RuntimeError(
            "GroundingDINO is not installed. Install groundeddino-vl or groundingdino before running."
        ) from exc

    return load_model(str(config_path), str(weights_path), device=device)


def resolve_gdino_config(user_path: str | None, *, project_root: Path) -> str | None:
    if user_path:
        resolved = resolve_path(user_path, project_root)
        if resolved is not None and resolved.exists():
            return str(resolved)

    for module_name, relative_glob in (
        ("groundeddino_vl", "models/configs/GroundingDINO_SwinT_OGC.py"),
        ("groundingdino", "config/GroundingDINO_SwinT_OGC.py"),
    ):
        try:
            module = __import__(module_name)
        except Exception:
            continue
        package_root = Path(module.__file__).resolve().parent
        matches = list(package_root.rglob(Path(relative_glob).name))
        if matches:
            return str(matches[0])
    return None


def predict_grounding_dino_image(
    *,
    model,
    image_path: str | Path,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> tuple[list[list[float]], list[float], list[int]]:
    try:
        try:
            from groundeddino_vl.utils.inference import load_image, predict
        except Exception:
            from groundingdino.util.inference import load_image, predict
        import torch
        from torchvision.ops import box_convert
    except Exception as exc:
        raise RuntimeError("Grounding DINO inference dependencies are unavailable") from exc

    image_source, image_tensor = load_image(str(image_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=caption,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        device=device,
    )

    if len(phrases) == 0:
        return [], [], []

    image_height, image_width = image_source.shape[:2]
    scale = torch.tensor(
        [image_width, image_height, image_width, image_height],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    boxes = boxes * scale
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

    mapped_boxes: list[list[float]] = []
    mapped_scores: list[float] = []
    mapped_labels: list[int] = []
    boxes_list = boxes.cpu().numpy().tolist()
    scores_list = logits.cpu().numpy().tolist()

    for box, score, phrase in zip(boxes_list, scores_list, phrases):
        class_id = phrase_to_two_class_id(phrase)
        if class_id is None:
            continue
        mapped_boxes.append([float(value) for value in box])
        mapped_scores.append(float(score))
        mapped_labels.append(class_id)

    return mapped_boxes, mapped_scores, mapped_labels


def _resolve_device(runtime_cfg: dict) -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    requested = str(runtime_cfg.get("train", {}).get("device", "cuda")).strip().lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"
