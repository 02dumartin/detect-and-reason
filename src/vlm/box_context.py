from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from src.vlm.image_adjustment import apply_image_adjustment
from src.visualization.yolo_overlay_renderer import DEFAULT_FONT_PATH, _draw_detection, _load_font

NEUTRAL_BOX_COLOR = (255, 255, 255)


def assign_box_numbers(
    records: list[dict[str, Any]],
    *,
    group_key: str,
    id_key: str,
) -> dict[Any, list[dict[str, Any]]]:
    """이미지별 bbox에 안정적인 1-based 번호를 매긴다.

    좌상단부터 읽는 순서(y -> x)로 정렬해, 같은 이미지의 어떤 record를 기준으로
    호출해도 항상 동일한 번호가 나오도록 한다 (학습/추론 양쪽에서 재현 가능해야 함).
    """
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record[group_key]].append(record)

    numbered: dict[Any, list[dict[str, Any]]] = {}
    for key, boxes in grouped.items():
        ordered = sorted(boxes, key=lambda box: (float(box["bbox_xyxy"][1]), float(box["bbox_xyxy"][0])))
        numbered[key] = [
            {"box_number": index, "bbox_xyxy": box["bbox_xyxy"], "id": box[id_key]}
            for index, box in enumerate(ordered, start=1)
        ]
    return numbered


def box_number_for_record(
    numbered_boxes: dict[Any, list[dict[str, Any]]],
    *,
    group_value: Any,
    record_id: Any,
) -> int:
    for box in numbered_boxes[group_value]:
        if box["id"] == record_id:
            return int(box["box_number"])
    raise KeyError(f"record_id={record_id!r} not found under group={group_value!r}")


def render_numbered_overlay_image(
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    *,
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = 36,
    box_thickness: int = 3,
    adjustment_cfg: dict[str, Any] | str | None = None,
) -> Image.Image:
    """탐지된 모든 bbox에 번호만 표기한 전체 이미지를 만든다.

    클래스 이름이나 클래스별 색상 코딩은 절대 넣지 않는다 — 정답 라벨이
    오버레이에 노출되면 학습(정답 유출)과 추론(실제로는 모르는 정보) 양쪽에서
    문제가 된다.

    adjustment_cfg는 crop 생성(load_crop)에 쓰는 것과 동일한 값을 넘겨야 한다 -
    crop만 밝기/대비 보정을 하고 오버레이는 원본 색 그대로 두면, 같은 사물이
    두 이미지에서 다른 색으로 보여 모델이 오히려 혼란스러워진다.
    """
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    image = apply_image_adjustment(image, adjustment_cfg)
    draw = ImageDraw.Draw(image)
    font = _load_font(font_path, font_size)
    for box in sorted(boxes, key=lambda b: b["box_number"]):
        _draw_detection(
            draw=draw,
            bbox_xyxy=[float(value) for value in box["bbox_xyxy"]],
            label=str(box["box_number"]),
            score=0.0,
            color=NEUTRAL_BOX_COLOR,
            font=font,
            box_thickness=box_thickness,
            label_only=True,
        )
    return image


def build_bbox_list_text(boxes: list[dict[str, Any]], *, target_box_number: int) -> str:
    lines = ["All detected tomato boxes in the full image (box_number: bbox):"]
    for box in sorted(boxes, key=lambda b: b["box_number"]):
        x1, y1, x2, y2 = [round(float(value), 2) for value in box["bbox_xyxy"]]
        marker = "  <- this is the target box" if int(box["box_number"]) == target_box_number else ""
        lines.append(f'Box {box["box_number"]}: {{"x1": {x1}, "y1": {y1}, "x2": {x2}, "y2": {y2}}}{marker}')
    return "\n".join(lines)
