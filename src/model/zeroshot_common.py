from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".WEBP",
    ".TIF",
    ".TIFF",
}


def prepare_prediction_dir(prediction_dir: str | Path) -> Path:
    prediction_dir = Path(prediction_dir)
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    (prediction_dir / "labels").mkdir(parents=True, exist_ok=True)
    return prediction_dir.resolve()


def iter_source_images(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    return sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    )


def load_image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def phrase_to_two_class_id(phrase: str) -> int | None:
    raw = str(phrase).strip().lower()
    if not raw:
        return None
    if "green" in raw or "unripe" in raw:
        return 1
    if "red" in raw or "ripe" in raw:
        return 0
    return None


def batched_nms(
    boxes_xyxy: list[list[float]],
    scores: list[float],
    labels: list[int],
    iou_threshold: float,
) -> tuple[list[list[float]], list[float], list[int]]:
    if not boxes_xyxy:
        return [], [], []

    try:
        import torch
        from torchvision.ops import batched_nms as torch_batched_nms
    except Exception:
        return boxes_xyxy, scores, labels

    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    keep = torch_batched_nms(boxes_tensor, scores_tensor, labels_tensor, float(iou_threshold))
    keep_indices = keep.cpu().tolist()
    return (
        [boxes_xyxy[idx] for idx in keep_indices],
        [scores[idx] for idx in keep_indices],
        [labels[idx] for idx in keep_indices],
    )


def save_yolo_prediction_txt(
    *,
    labels_dir: str | Path,
    image_stem: str,
    image_width: int,
    image_height: int,
    boxes_xyxy: list[list[float]],
    scores: list[float],
    labels: list[int],
    save_conf: bool,
) -> Path:
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []

    for box, score, class_id in zip(boxes_xyxy, scores, labels):
        x1, y1, x2, y2 = box
        box_w = max(0.0, float(x2) - float(x1))
        box_h = max(0.0, float(y2) - float(y1))
        if box_w <= 0.0 or box_h <= 0.0:
            continue

        xc = (float(x1) + float(x2)) / 2.0 / float(image_width)
        yc = (float(y1) + float(y2)) / 2.0 / float(image_height)
        norm_w = box_w / float(image_width)
        norm_h = box_h / float(image_height)

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        norm_w = min(max(norm_w, 0.0), 1.0)
        norm_h = min(max(norm_h, 0.0), 1.0)

        if save_conf:
            rows.append(
                f"{int(class_id)} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f} {float(score):.6f}"
            )
        else:
            rows.append(f"{int(class_id)} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

    label_path = labels_dir / f"{image_stem}.txt"
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return label_path
