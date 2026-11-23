from __future__ import annotations

from typing import Any

from pathlib import Path

from PIL import Image, ImageOps

from .image_adjustment import apply_image_adjustment


def load_crop(
    record: dict,
    *,
    expand_ratio: float = 0.1,
    min_size: int = 32,
    adjustment_cfg: dict[str, Any] | str | None = None,
    resize: int | None = None,
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
    crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop = apply_image_adjustment(crop, adjustment_cfg)
    if resize is not None:
        # exemplar 이미지가 ImageOps.contain(..., 224)로 확대돼서 표시되는 것과 달리
        # target crop은 리사이즈가 없어서, 작은 tomato box는 exemplar들보다 훨씬
        # 작게(때론 40px대) 표시돼 multi-image 프롬프트에서 존재감을 잃는다 —
        # target crop을 exemplar와 같거나 더 크게 맞춰서 시각적으로 두드러지게 한다.
        crop = ImageOps.contain(crop, (int(resize), int(resize)))
    return crop
