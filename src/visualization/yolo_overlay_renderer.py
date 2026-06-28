from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.evaluation.yolo_label_io import (
    build_image_index,
    load_yolo_label_rows,
    normalize_class_name_map,
    yolo_row_to_detection,
)


DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
DEFAULT_COLOR_MAP_RGB = {
    "fully-ripe": (214, 66, 36),
    "semi-ripe": (240, 186, 36),
    "unripe": (106, 254, 152),
    "none": (180, 180, 180),
}


def save_yolo_label_overlays(
    *,
    labels_dir: str | Path,
    img_root: str | Path,
    output_dir: str | Path,
    class_names: dict[int, str] | list[str],
    overlay_config: dict[str, Any] | None = None,
    font_path: str | None = None,
    font_size: int | None = None,
    box_thickness: int | None = None,
    label_only: bool | None = None,
    max_images: int | None = None,
) -> int:
    """YOLO txt 기반 bbox overlay 이미지를 저장한다.

    핵심 목표는 두 가지다.
    1. 예측/정답 txt만 있으면 바로 시각화할 수 있을 것
    2. 색상과 라벨 스타일은 `tomato-detection-agentic`와 같은 감각으로 유지할 것

    따라서 실제 렌더링 옵션은 코드 기본값 + YAML 설정 + CLI override 순서로
    합쳐서 사용한다.
    """
    labels_dir = Path(labels_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_options = _merge_overlay_options(
        overlay_config=overlay_config,
        font_path=font_path,
        font_size=font_size,
        box_thickness=box_thickness,
        label_only=label_only,
    )

    image_index = build_image_index(img_root)
    class_name_map = normalize_class_name_map(class_names)
    font = _load_font(str(overlay_options["font_path"]), int(overlay_options["font_size"]))

    rendered = 0
    for txt_path in sorted(labels_dir.glob("*.txt")):
        if max_images is not None and rendered >= max_images:
            break

        image_info = image_index.get(txt_path.stem)
        if image_info is None:
            continue

        # EXIF 방향을 고려하지 않으면 bbox는 맞는데 이미지가 회전되어 보이거나,
        # 반대로 이미지가 바로 서 있는데 bbox가 틀어진 것처럼 보일 수 있다.
        with Image.open(image_info.path) as raw_image:
            orientation = int(raw_image.getexif().get(274, 1))
            raw_width, raw_height = raw_image.size

        image = ImageOps.exif_transpose(Image.open(image_info.path)).convert("RGB")
        draw = ImageDraw.Draw(image)

        for row in load_yolo_label_rows(txt_path):
            class_id, bbox_xyxy, score = yolo_row_to_detection(row, image_info.width, image_info.height)
            if overlay_options["honor_exif"] and overlay_options["transform_bbox_with_exif"] and orientation in {3, 6, 8}:
                bbox_xyxy = _transform_bbox_xyxy_exif(bbox_xyxy, raw_width, raw_height, orientation)

            label = class_name_map.get(class_id, str(class_id))
            color = _resolve_label_color(label, overlay_options["color_map_rgb"])
            _draw_detection(
                draw=draw,
                bbox_xyxy=bbox_xyxy,
                label=label,
                score=score,
                color=color,
                font=font,
                box_thickness=int(overlay_options["box_thickness"]),
                label_only=bool(overlay_options["label_only"]),
            )

        output_path = output_dir / f"{txt_path.stem}_overlay.jpg"
        image.save(output_path, quality=95)
        rendered += 1

    return rendered


def save_yolo_txt_overlays(**kwargs: Any) -> int:
    """기존 함수명을 유지하기 위한 호환 alias다."""
    return save_yolo_label_overlays(**kwargs)


def _merge_overlay_options(
    *,
    overlay_config: dict[str, Any] | None,
    font_path: str | None,
    font_size: int | None,
    box_thickness: int | None,
    label_only: bool | None,
) -> dict[str, Any]:
    """기본값, YAML 설정, CLI override를 하나의 옵션 dict로 합친다."""
    raw_cfg = overlay_config or {}
    cfg = raw_cfg.get("overlay", raw_cfg) if isinstance(raw_cfg, dict) else {}

    color_map_raw = cfg.get("color_map_rgb", DEFAULT_COLOR_MAP_RGB)
    color_map_rgb = {
        str(key): tuple(int(channel) for channel in value)
        for key, value in color_map_raw.items()
    }

    return {
        "font_path": font_path or cfg.get("font_path", DEFAULT_FONT_PATH),
        "font_size": int(font_size if font_size is not None else cfg.get("font_size", 32)),
        "box_thickness": int(box_thickness if box_thickness is not None else cfg.get("box_thickness", 4)),
        "label_only": bool(label_only if label_only is not None else cfg.get("label_only", False)),
        "honor_exif": bool(cfg.get("honor_exif", True)),
        "transform_bbox_with_exif": bool(cfg.get("transform_bbox_with_exif", True)),
        "prefer_enhanced": bool(cfg.get("prefer_enhanced", False)),
        "color_map_rgb": color_map_rgb,
    }


def _resolve_label_color(label: str, color_map_rgb: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """클래스 이름을 ripeness 그룹으로 정규화해 일관된 bbox 색을 고른다."""
    normalized = _normalize_ripeness_key(label)
    return color_map_rgb.get(normalized, color_map_rgb.get("none", DEFAULT_COLOR_MAP_RGB["none"]))


def _normalize_ripeness_key(label: str) -> str:
    """프로젝트마다 조금씩 다른 클래스 이름 표기를 공통 키로 맞춘다."""
    normalized = str(label).strip().lower().replace("_", " ")

    if normalized.startswith("adj "):
        normalized = normalized[4:]

    if "fully" in normalized or "ripe" in normalized:
        if "half" not in normalized and "semi" not in normalized:
            return "fully-ripe"
    if "half" in normalized or "semi" in normalized:
        return "semi-ripe"
    if "green" in normalized or "unripe" in normalized:
        return "unripe"
    return "none"


def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """지정 폰트가 없을 때도 overlay 저장이 실패하지 않도록 fallback을 둔다."""
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _draw_detection(
    *,
    draw: ImageDraw.ImageDraw,
    bbox_xyxy: list[float],
    label: str,
    score: float,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    box_thickness: int,
    label_only: bool,
) -> None:
    """박스와 라벨 배경을 함께 그려 가독성을 높인다."""
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=box_thickness)

    text = label if label_only else f"{label} {score:.2f}"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    text_x = x1
    text_y = max(0, y1 - text_height - 10)

    draw.rectangle((text_x, text_y, text_x + text_width + 10, text_y + text_height + 6), fill=color)
    draw.text((text_x + 5, text_y + 3), text, font=font, fill=(0, 0, 0))


def _transform_point_exif(x: float, y: float, width: int, height: int, orientation: int) -> tuple[float, float]:
    """EXIF orientation이 적용된 이미지 좌표계로 점을 옮긴다."""
    if orientation == 3:
        return width - x, height - y
    if orientation == 6:
        return height - y, x
    if orientation == 8:
        return y, width - x
    return x, y


def _transform_bbox_xyxy_exif(
    bbox_xyxy: list[float],
    width: int,
    height: int,
    orientation: int,
) -> list[float]:
    """bbox 네 꼭짓점을 각각 회전시켜, 회전 후 최소 외접 사각형을 다시 만든다."""
    x1, y1, x2, y2 = bbox_xyxy
    points = [
        _transform_point_exif(x1, y1, width, height, orientation),
        _transform_point_exif(x1, y2, width, height, orientation),
        _transform_point_exif(x2, y1, width, height, orientation),
        _transform_point_exif(x2, y2, width, height, orientation),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]
