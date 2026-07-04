from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.model.zeroshot_common import (
    batched_nms,
    iter_source_images,
    load_image_size,
    prepare_prediction_dir,
    save_yolo_prediction_txt,
)


def run_owl_vit_predict(runtime_cfg: dict) -> dict[str, str | int]:
    source_dir = Path(runtime_cfg["predict"]["resolved_source"])
    prediction_dir = prepare_prediction_dir(runtime_cfg["paths"]["prediction_dir"])
    labels_dir = prediction_dir / "labels"

    prompts = list(runtime_cfg.get("prompts") or [])
    if not prompts:
        raise ValueError("OWL-ViT zero-shot prompts are missing")

    score_threshold = float(runtime_cfg.get("predict", {}).get("conf") or 0.25)
    nms_iou = float(runtime_cfg.get("predict", {}).get("resolved_iou") or 0.5)
    save_conf = bool(runtime_cfg.get("predict", {}).get("save_conf", True))
    model_name = (
        runtime_cfg.get("model", {}).get("hf_model")
        or runtime_cfg.get("paths", {}).get("weights")
        or "google/owlv2-large-patch14-ensemble"
    )

    device = _resolve_device(runtime_cfg)
    model, processor = build_owl_vit_model(model_name, device)

    num_images = 0
    num_detections = 0
    for image_path in iter_source_images(source_dir):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            boxes_xyxy, scores, labels = predict_owl_vit_image(
                model=model,
                processor=processor,
                image=image,
                prompts=prompts,
                score_threshold=score_threshold,
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


def build_owl_vit_model(model_name: str, device: str):
    try:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except Exception as exc:
        raise RuntimeError(
            "transformers with OWLv2 support is required for OWL-ViT zero-shot inference."
        ) from exc

    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


def predict_owl_vit_image(
    *,
    model,
    processor,
    image: Image.Image,
    prompts: list[str],
    score_threshold: float,
    device: str,
) -> tuple[list[list[float]], list[float], list[int]]:
    import torch

    inputs = processor(text=prompts, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    if hasattr(processor, "post_process_grounded_object_detection"):
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=float(score_threshold),
            text_labels=[prompts],
        )[0]
    else:
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=float(score_threshold),
        )[0]

    boxes = results["boxes"].detach().cpu().numpy().tolist()
    scores = results["scores"].detach().cpu().numpy().tolist()
    prompt_indices = results["labels"].detach().cpu().numpy().tolist()

    mapped_boxes: list[list[float]] = []
    mapped_scores: list[float] = []
    mapped_labels: list[int] = []
    for box, score, prompt_index in zip(boxes, scores, prompt_indices):
        prompt_index = int(prompt_index)
        if prompt_index < 0 or prompt_index >= len(prompts):
            continue
        mapped_boxes.append([float(value) for value in box])
        mapped_scores.append(float(score))
        mapped_labels.append(prompt_index)
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
