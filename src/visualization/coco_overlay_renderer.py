from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.evaluation.yolo_label_io import normalize_class_name_map
# YOLO overlay 와 완전히 동일한 스타일을 쓰기 위해 동일 헬퍼를 재사용한다.
from src.visualization.yolo_overlay_renderer import (
    _draw_detection,
    _load_font,
    _merge_overlay_options,
    _resolve_display_label,
    _resolve_label_color,
)


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_coco_prediction_overlays(
    *,
    gt_json: str | Path,
    img_root: str | Path,
    output_dir: str | Path,
    class_names: dict[int, str] | list[str] | None,
    pred_json: str | Path | None = None,
    overlay_config: dict[str, Any] | None = None,
    font_path: str | None = None,
    font_size: int | None = None,
    box_thickness: int | None = None,
    label_only: bool | None = None,
    max_images: int | None = None,
    conf_threshold: float = 0.25,
    ground_truth: bool = False,
) -> int:
    """COCO 예측(또는 GT) bbox overlay 이미지를 저장한다.

    YOLO overlay(`save_yolo_label_overlays`)와 출력 포맷·색상·라벨 스타일을 동일하게 맞춘다.
    차이는 입력뿐: YOLO txt(정규화 cxcywh) 대신 COCO json(절대픽셀 xywh)을 읽는다.

    - ground_truth=True  : gt_json 의 annotations 를 그린다 (score=1.0).
    - ground_truth=False : pred_json(COCO detection list)을 conf_threshold 로 거른 뒤 그린다.

    COCO 좌표는 저장된 이미지 픽셀 공간(json width/height = 파일 실제 크기)에 정의돼 있으므로
    EXIF transpose 없이 그대로 그린다(박스 정합 보장).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_root = Path(img_root)

    gt = _load_json(gt_json)
    images_by_id = {int(img["id"]): img for img in gt.get("images", [])}
    gt_cat_names = {int(cat["id"]): str(cat["name"]) for cat in gt.get("categories", [])}
    # 표시용 클래스명: 모델 config 의 canonical 이름 우선, 없으면 GT category 이름.
    class_name_map = normalize_class_name_map(class_names) if class_names else gt_cat_names

    overlay_options = _merge_overlay_options(
        overlay_config=overlay_config,
        font_path=font_path,
        font_size=font_size,
        box_thickness=box_thickness,
        label_only=label_only,
    )
    font = _load_font(str(overlay_options["font_path"]), int(overlay_options["font_size"]))

    # image_id -> [(category_id, bbox_xyxy, score), ...]
    by_image: dict[int, list[tuple[int, list[float], float]]] = {}
    if ground_truth:
        for ann in gt.get("annotations", []):
            x, y, w, h = (float(v) for v in ann["bbox"][:4])
            by_image.setdefault(int(ann["image_id"]), []).append(
                (int(ann["category_id"]), [x, y, x + w, y + h], 1.0)
            )
    else:
        if pred_json is None:
            raise ValueError("pred_json is required when ground_truth=False")
        for det in _load_json(pred_json):
            score = float(det.get("score", 1.0))
            if score < conf_threshold:
                continue
            x, y, w, h = (float(v) for v in det["bbox"][:4])
            by_image.setdefault(int(det["image_id"]), []).append(
                (int(det["category_id"]), [x, y, x + w, y + h], score)
            )

    rendered = 0
    for image_id in sorted(images_by_id):
        if max_images is not None and rendered >= max_images:
            break
        info = images_by_id[image_id]
        file_name = str(info["file_name"])
        # GT file_name 은 보통 "images/xxx.jpg". img_root 기준으로 먼저, 안 되면 basename 으로.
        img_path = img_root / file_name
        if not img_path.is_file():
            img_path = img_root / Path(file_name).name
        if not img_path.is_file():
            continue

        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for category_id, bbox_xyxy, score in by_image.get(image_id, []):
            label = _resolve_display_label(category_id, class_name_map)
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

        output_path = output_dir / f"{Path(file_name).stem}_overlay.jpg"
        image.save(output_path, quality=95)
        rendered += 1

    return rendered
