from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.visualization.yolo_overlay_renderer import (
    _draw_detection,
    _load_font,
    _merge_overlay_options,
    _resolve_label_color,
    _transform_bbox_xyxy_exif,
)


def save_vlm_prediction_overlays(
    *,
    predictions: list[dict[str, Any]],
    output_dir: str | Path,
    overlay_config: dict[str, Any] | None = None,
    font_size: int | None = None,
    box_thickness: int | None = None,
    label_only: bool | None = None,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["image_path"])].append(row)

    options = _merge_overlay_options(
        overlay_config=overlay_config,
        font_path=None,
        font_size=font_size,
        box_thickness=box_thickness,
        label_only=label_only,
    )
    font = _load_font(str(options["font_path"]), int(options["font_size"]))

    rendered = 0
    for image_path_str, rows in grouped.items():
        image_path = Path(image_path_str)
        with Image.open(image_path) as raw_image:
            orientation = int(raw_image.getexif().get(274, 1))
            raw_width, raw_height = raw_image.size
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        draw = ImageDraw.Draw(image)

        for row in rows:
            bbox_xyxy = [float(value) for value in row["bbox_xyxy"]]
            if options["honor_exif"] and options["transform_bbox_with_exif"] and orientation in {3, 6, 8}:
                bbox_xyxy = _transform_bbox_xyxy_exif(bbox_xyxy, raw_width, raw_height, orientation)

            label = str(row.get("predicted_class_name") or "none")
            color = _resolve_label_color(label, options["color_map_rgb"])
            _draw_detection(
                draw=draw,
                bbox_xyxy=bbox_xyxy,
                label=label,
                score=float(row.get("det_score", 0.0)),
                color=color,
                font=font,
                box_thickness=int(options["box_thickness"]),
                label_only=bool(options["label_only"]),
            )

        image.save(output_dir / f"{image_path.stem}_overlay.jpg", quality=95)
        rendered += 1
    return rendered

