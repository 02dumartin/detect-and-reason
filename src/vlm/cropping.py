from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_crop(
    record: dict,
    *,
    expand_ratio: float = 0.1,
    min_size: int = 32,
) -> Image.Image:
    image_path = Path(record["image_path"])
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")

    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
    box_w = max(float(min_size), x2 - x1)
    box_h = max(float(min_size), y2 - y1)
    pad_x = box_w * max(0.0, float(expand_ratio))
    pad_y = box_h * max(0.0, float(expand_ratio))

    crop_x1 = max(0, int(round(x1 - pad_x)))
    crop_y1 = max(0, int(round(y1 - pad_y)))
    crop_x2 = min(width, int(round(x2 + pad_x)))
    crop_y2 = min(height, int(round(y2 + pad_y)))

    if crop_x2 <= crop_x1:
        crop_x2 = min(width, crop_x1 + max(1, int(min_size)))
    if crop_y2 <= crop_y1:
        crop_y2 = min(height, crop_y1 + max(1, int(min_size)))
    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

