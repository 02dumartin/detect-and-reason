"""COCO GT 기반 VLM 학습/평가용 crop + JSONL 데이터셋 생성.

scripts/vlm_full_ft_with_exemplars.py가 원래 갖고 있던 데이터 생성 로직을 재사용
가능한 모듈로 뺀 것. full-FT SFT 데이터(messages/images jsonl) 생성뿐 아니라,
동일한 crop/exemplar/overlay 생성 로직을 zero-shot 실험 데이터 준비에도 그대로
쓸 수 있게 한다 (실제 zero-shot 추론 시 crop/overlay는 src/vlm/classification.py가
detector 예측을 입력으로 즉석에서 만들지만, GT 기반으로 미리 데이터셋을 굳혀두고
싶을 때는 이 모듈의 build_vlm_dataset()을 그대로 쓰면 된다).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import ImageOps, ImageStat

from src.vlm.box_context import (
    assign_box_numbers,
    box_number_for_record,
    build_bbox_list_text,
    render_numbered_overlay_image,
)
from src.vlm.cropping import load_crop
from src.vlm.prompting import build_instruction_text, build_target_crop_intro_text, FULL_OVERLAY_INTRO_TEXT
from src.vlm.sampling import downsample_class_records
from src.vlm.schema import CANONICAL_CLASS_ID_TO_NAME, normalize_prediction_label

SYSTEM_MESSAGE = (
    "You classify tomato ripeness from a detector crop. "
    "Always return strict JSON only. Do not add markdown or extra text."
)

REASONING_TEMPLATES = {
    "fully-ripe": "Step1: red dominates with no clear green area; Step2: fully-ripe.",
    "semi-ripe": "Step1: red-orange-green transition is visible; Step2: semi-ripe.",
    "unripe": "Step1: green surface remains dominant; Step2: unripe.",
}

TWO_CLASS_MAP = {0: "ripe", 1: "unripe"}


@dataclass
class VlmDatasetConfig:
    """crop/exemplar/overlay 생성에 필요한 실행 파라미터.

    argparse.Namespace 대신 명시적 dataclass로 둬서, CLI 스크립트뿐 아니라 향후
    zero-shot 데이터 준비 스크립트나 노트북에서도 하드코딩 없이 재사용할 수 있게 한다.
    """

    dataset_ref: str
    class_mode: str = "3cls"
    examples_per_class: int = 2
    expand_ratio: float = 0.1
    exemplar_expand_ratio: float | None = None
    min_size: int = 32
    exemplar_min_size: int | None = None
    exemplar_save_size: int | None = None
    exemplar_pick_strategy: str = "brightness_then_area"
    crop_adjustment_cfg: dict[str, Any] | None = None
    unripe_keep_ratio: float = 1.0
    seed: int = 42
    use_full_overlay: bool = False
    overlay_font_size: int = 36
    overlay_box_thickness: int = 3
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    def resolved_exemplar_expand_ratio(self) -> float:
        return float(self.exemplar_expand_ratio) if self.exemplar_expand_ratio is not None else float(self.expand_ratio)

    def resolved_exemplar_min_size(self) -> int:
        return int(self.exemplar_min_size) if self.exemplar_min_size is not None else int(self.min_size)


@dataclass
class VlmDatasetResult:
    dataset_output_root: Path
    train_jsonl: Path
    val_jsonl: Path
    test_jsonl: Path
    manifest_path: Path
    manifest: dict[str, Any]


def resolve_task_class_map(class_mode: str) -> dict[int, str]:
    if class_mode == "2cls":
        return dict(TWO_CLASS_MAP)
    return dict(CANONICAL_CLASS_ID_TO_NAME)


def build_reasoning_templates(class_map: dict[int, str]) -> dict[str, str]:
    class_names = [class_map[class_id] for class_id in sorted(class_map)]
    if class_names == ["ripe", "unripe"]:
        return {
            "ripe": "Step1: red or orange dominates with no clear green area; Step2: ripe.",
            "unripe": "Step1: green surface remains dominant; Step2: unripe.",
        }
    return dict(REASONING_TEMPLATES)


def build_experiment_suffix(config: VlmDatasetConfig) -> str:
    suffix_parts: list[str] = []
    crop_adjustment_type = (config.crop_adjustment_cfg or {}).get("type")
    if crop_adjustment_type:
        suffix_parts.append(str(crop_adjustment_type))
    if float(config.unripe_keep_ratio) < 1.0:
        suffix_parts.append(f"u{int(round(float(config.unripe_keep_ratio) * 100)):02d}")
    exemplar_expand_ratio = config.resolved_exemplar_expand_ratio()
    if abs(exemplar_expand_ratio - float(config.expand_ratio)) > 1e-9:
        suffix_parts.append(f"exr{int(round(exemplar_expand_ratio * 100)):02d}")
    exemplar_min_size = config.resolved_exemplar_min_size()
    if exemplar_min_size != int(config.min_size):
        suffix_parts.append(f"ems{exemplar_min_size}")
    if config.use_full_overlay:
        suffix_parts.append("ovl")
    return "_" + "_".join(suffix_parts) if suffix_parts else ""


def load_coco_records(
    coco_root: Path,
    split_key: str,
    *,
    class_map: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_dir_map = {"train": "train", "val": "valid", "test": "test"}
    split_dir = split_dir_map[split_key]
    ann_path = coco_root / split_dir / "_annotations.coco.json"
    with ann_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images_by_id = {item["id"]: item for item in coco["images"]}
    category_names = {item["id"]: item["name"] for item in coco["categories"]}

    records: list[dict[str, Any]] = []
    for ann in coco["annotations"]:
        image = images_by_id[ann["image_id"]]
        image_path = _resolve_coco_image_path(coco_root, split_dir, image["file_name"])
        x, y, w, h = [float(value) for value in ann["bbox"][:4]]
        bbox_xyxy = [x, y, x + w, y + h]
        class_id, class_name = normalize_prediction_label(
            class_name=category_names.get(ann["category_id"]),
            category_id=ann["category_id"],
            class_name_map=class_map,
        )
        if class_id < 0 or not image_path.exists():
            continue

        records.append(
            {
                "sample_id": f"{split_key}-{ann['id']}",
                "split": split_key,
                "ann_id": ann["id"],
                "image_id": ann["image_id"],
                "image_path": str(image_path.resolve()),
                "file_name": image["file_name"],
                "width": int(image["width"]),
                "height": int(image["height"]),
                "bbox_xyxy": bbox_xyxy,
                "bbox_xywh": [x, y, w, h],
                "category_id": int(ann["category_id"]),
                "class_name": class_name,
            }
        )
    return records, coco


def _resolve_coco_image_path(coco_root: Path, split_dir: str, file_name: str) -> Path:
    candidates = [
        coco_root / split_dir / file_name,
        coco_root / split_dir / "images" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def image_brightness(record: dict[str, Any], *, expand_ratio: float, min_size: int) -> float:
    try:
        crop = load_crop(record, expand_ratio=expand_ratio, min_size=min_size)
    except Exception:
        return float("nan")
    stat = ImageStat.Stat(crop.convert("L"))
    return float(stat.mean[0])


def select_exemplar_records(
    records: list[dict[str, Any]],
    *,
    examples_per_class: int,
    expand_ratio: float,
    min_size: int,
    class_map: dict[int, str],
    pick_strategy: str = "area_then_brightness",
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[tuple[float, float, dict[str, Any]]]] = {class_name: [] for class_name in class_map.values()}
    for record in records:
        class_name = record["class_name"]
        if class_name not in buckets:
            continue
        brightness = image_brightness(record, expand_ratio=expand_ratio, min_size=min_size)
        if brightness != brightness:
            continue
        area = _record_bbox_area(record)
        buckets[class_name].append((brightness, area, record))

    selected: dict[str, list[dict[str, Any]]] = {}
    for class_name, items in buckets.items():
        if not items:
            raise RuntimeError(f"No exemplar candidates found for class: {class_name}")
        if pick_strategy == "brightness":
            ordered = sorted(items, key=lambda item: (item[0], item[1]), reverse=True)
        elif pick_strategy == "area":
            ordered = sorted(items, key=lambda item: item[1], reverse=True)
        elif pick_strategy == "brightness_then_area":
            ordered = sorted(items, key=lambda item: (item[0], item[1]), reverse=True)
        else:
            if pick_strategy == "class_aware":
                if class_name in {"fully-ripe", "semi-ripe"}:
                    ordered = sorted(items, key=lambda item: (item[0], item[1]), reverse=True)
                else:
                    ordered = sorted(items, key=lambda item: (item[1], item[0]), reverse=True)
            else:
                ordered = sorted(items, key=lambda item: (item[1], item[0]), reverse=True)

        unique_records: list[dict[str, Any]] = []
        seen_sample_ids: set[str] = set()
        for _, _, record in ordered:
            if record["sample_id"] in seen_sample_ids:
                continue
            unique_records.append(record)
            seen_sample_ids.add(record["sample_id"])
            if len(unique_records) >= examples_per_class:
                break
        selected[class_name] = unique_records
    return selected


def save_exemplar_crops(
    exemplar_records: dict[str, list[dict[str, Any]]],
    exemplar_dir: Path,
    *,
    expand_ratio: float,
    min_size: int,
    class_map: dict[int, str],
    adjustment_cfg: dict[str, Any] | None = None,
    save_size: int | None = None,
) -> list[dict[str, str]]:
    if exemplar_dir.exists():
        shutil.rmtree(exemplar_dir)
    exemplar_dir.mkdir(parents=True, exist_ok=True)
    exemplars: list[dict[str, str]] = []
    for class_name in class_map.values():
        for index, record in enumerate(exemplar_records[class_name], start=1):
            crop = load_crop(
                record,
                expand_ratio=expand_ratio,
                min_size=min_size,
                adjustment_cfg=adjustment_cfg,
            )
            if save_size is not None:
                crop = ImageOps.contain(crop, (save_size, save_size))
            crop_path = exemplar_dir / f"{class_name}_{index}_{record['sample_id']}.jpg"
            crop.save(crop_path, format="JPEG", quality=95)
            exemplars.append(
                {
                    "class_name": class_name,
                    "sample_id": record["sample_id"],
                    "path": str(crop_path.resolve()),
                }
            )
    return exemplars


def build_user_text(
    record: dict[str, Any],
    exemplars: list[dict[str, str]],
    *,
    class_map: dict[int, str],
    box_number: int | None = None,
    bbox_list_text: str | None = None,
) -> str:
    parts = ["Reference exemplar images with labels:"]
    for exemplar in exemplars:
        parts.append(f'{exemplar["class_name"]} reference: <image>')
    parts.append("")
    if bbox_list_text is not None:
        parts.append(f"{FULL_OVERLAY_INTRO_TEXT} <image>")
        parts.append(bbox_list_text)
        parts.append("")
    parts.append(f"{build_target_crop_intro_text(box_number)} <image>")
    parts.append("")
    parts.append(build_instruction(record, class_map=class_map))
    return "\n".join(parts)


def rounded_bbox_payload(bbox_xyxy: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = [round(float(value), 2) for value in bbox_xyxy]
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def build_instruction(record: dict[str, Any], *, class_map: dict[int, str]) -> str:
    prompt_cfg = {
        "use_reasoning": True,
        "use_color_guide": True,
        "include_bbox_json": True,
        "reasoning_max_words": 35,
    }
    return build_instruction_text(record=record, prompt_cfg=prompt_cfg, class_map=class_map)


def build_assistant_text(record: dict[str, Any], *, reasoning_templates: dict[str, str]) -> str:
    payload = {
        "class_name": record["class_name"],
        "bbox": rounded_bbox_payload(record["bbox_xyxy"]),
        "reasoning": reasoning_templates[record["class_name"]],
    }
    return json.dumps(payload, ensure_ascii=False)


def _record_bbox_area(record: dict[str, Any]) -> float:
    bbox = record.get("bbox_xywh")
    if bbox and len(bbox) >= 4:
        return float(bbox[2]) * float(bbox[3])
    x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def build_training_row(
    record: dict[str, Any],
    crop_path: Path,
    exemplars: list[dict[str, str]],
    *,
    class_map: dict[int, str],
    reasoning_templates: dict[str, str],
    overlay_path: Path | None = None,
    box_number: int | None = None,
    bbox_list_text: str | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_user_text(
                record,
                exemplars,
                class_map=class_map,
                box_number=box_number,
                bbox_list_text=bbox_list_text,
            ),
        },
    ]
    images = [exemplar["path"] for exemplar in exemplars]
    if overlay_path is not None:
        images.append(str(overlay_path.resolve()))
    images.append(str(crop_path.resolve()))
    return {
        "messages": messages + [{"role": "assistant", "content": build_assistant_text(record, reasoning_templates=reasoning_templates)}],
        "images": images,
        "sample_id": record["sample_id"],
        "class_name_target": record["class_name"],
        "exemplar_sample_ids": [exemplar["sample_id"] for exemplar in exemplars],
        "box_number": box_number,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def convert_split(
    records: list[dict[str, Any]],
    split_key: str,
    *,
    crop_root: Path,
    exemplars: list[dict[str, str]],
    expand_ratio: float,
    min_size: int,
    limit: int | None,
    class_map: dict[int, str],
    reasoning_templates: dict[str, str],
    adjustment_cfg: dict[str, Any] | None = None,
    box_groups: dict[int, list[dict[str, Any]]] | None = None,
    overlay_font_size: int = 36,
    overlay_box_thickness: int = 3,
) -> list[dict[str, Any]]:
    selected = records if limit is None else records[:limit]
    crop_dir = crop_root / split_key
    crop_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = crop_dir / "overlays"
    overlay_cache: dict[int, Path] = {}
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(selected, start=1):
        try:
            crop = load_crop(
                record,
                expand_ratio=expand_ratio,
                min_size=min_size,
                adjustment_cfg=adjustment_cfg,
            )
        except Exception as exc:
            print(f"[{split_key}] skipped unreadable sample_id={record['sample_id']} error={exc}", flush=True)
            continue
        crop_path = crop_dir / f"{record['sample_id']}.jpg"
        crop.save(crop_path, format="JPEG", quality=95)

        overlay_path = None
        box_number = None
        bbox_list_text = None
        if box_groups is not None:
            boxes = box_groups[record["image_id"]]
            box_number = box_number_for_record(box_groups, group_value=record["image_id"], record_id=record["ann_id"])
            bbox_list_text = build_bbox_list_text(boxes, target_box_number=box_number)
            overlay_path = overlay_cache.get(record["image_id"])
            if overlay_path is None:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay_image = render_numbered_overlay_image(
                    record["image_path"],
                    boxes,
                    font_size=overlay_font_size,
                    box_thickness=overlay_box_thickness,
                    adjustment_cfg=adjustment_cfg,
                )
                overlay_path = overlay_dir / f"{record['image_id']}_overlay.jpg"
                overlay_image.save(overlay_path, format="JPEG", quality=95)
                overlay_cache[record["image_id"]] = overlay_path

        rows.append(
            build_training_row(
                record,
                crop_path,
                exemplars,
                class_map=class_map,
                reasoning_templates=reasoning_templates,
                overlay_path=overlay_path,
                box_number=box_number,
                bbox_list_text=bbox_list_text,
            )
        )
        if index % 500 == 0 or index == len(selected):
            print(f"[{split_key}] converted {index}/{len(selected)}", flush=True)
    return rows


def build_vlm_dataset(
    config: VlmDatasetConfig,
    *,
    coco_root: Path,
    dataset_output_root: Path,
) -> VlmDatasetResult:
    """GT COCO 라벨로부터 crop + exemplar + (옵션) overlay context가 포함된
    train/val/test jsonl과 manifest.json을 만든다.

    seed(config.seed)는 unripe downsample 샘플링에만 쓰인다 - 나머지(예: exemplar
    선택, box 번호 매김)는 전부 결정론적 정렬 기반이라 시드가 필요 없다. 동일한
    config로 다시 호출하면 항상 같은 데이터셋이 나온다.
    """
    class_map = resolve_task_class_map(config.class_mode)
    reasoning_templates = build_reasoning_templates(class_map)

    crop_root = dataset_output_root / "crops"
    exemplar_dir = dataset_output_root / "exemplars"
    train_jsonl = dataset_output_root / "train.jsonl"
    val_jsonl = dataset_output_root / "val.jsonl"
    test_jsonl = dataset_output_root / "test.jsonl"
    manifest_path = dataset_output_root / "manifest.json"

    train_records, train_coco = load_coco_records(coco_root, "train", class_map=class_map)
    val_records, val_coco = load_coco_records(coco_root, "val", class_map=class_map)
    try:
        test_records, test_coco = load_coco_records(coco_root, "test", class_map=class_map)
    except FileNotFoundError:
        test_records, test_coco = [], {"images": []}

    print(f"coco_root    = {coco_root}")
    print(f"train rows   = {len(train_records)} from {len(train_coco['images'])} images")
    print(f"val rows     = {len(val_records)} from {len(val_coco['images'])} images")
    print(f"test rows    = {len(test_records)} from {len(test_coco['images'])} images")

    exemplar_expand_ratio = config.resolved_exemplar_expand_ratio()
    exemplar_min_size = config.resolved_exemplar_min_size()
    exemplar_records = select_exemplar_records(
        train_records,
        examples_per_class=config.examples_per_class,
        expand_ratio=exemplar_expand_ratio,
        min_size=exemplar_min_size,
        class_map=class_map,
        pick_strategy=config.exemplar_pick_strategy,
    )
    train_records_for_learning = downsample_class_records(
        train_records,
        class_name="unripe",
        keep_ratio=float(config.unripe_keep_ratio),
        seed=int(config.seed),
    )
    print(
        f"train rows after unripe downsample = {len(train_records_for_learning)} "
        f"(raw={len(train_records)}, keep_ratio={float(config.unripe_keep_ratio):.3f}, seed={config.seed})"
    )
    exemplars = save_exemplar_crops(
        exemplar_records,
        exemplar_dir,
        expand_ratio=exemplar_expand_ratio,
        min_size=exemplar_min_size,
        class_map=class_map,
        adjustment_cfg=config.crop_adjustment_cfg,
        save_size=config.exemplar_save_size,
    )

    # box_groups는 downsample 이전의 전체 record 목록으로 계산한다 - unripe crop이
    # 학습에서 빠지더라도 overlay/리스트에는 이미지 안의 실제 박스가 다 보여야 한다.
    def _box_groups(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]] | None:
        if not config.use_full_overlay:
            return None
        return assign_box_numbers(records, group_key="image_id", id_key="ann_id")

    convert_kwargs = dict(
        crop_root=crop_root,
        exemplars=exemplars,
        expand_ratio=config.expand_ratio,
        min_size=config.min_size,
        class_map=class_map,
        reasoning_templates=reasoning_templates,
        adjustment_cfg=config.crop_adjustment_cfg,
        overlay_font_size=config.overlay_font_size,
        overlay_box_thickness=config.overlay_box_thickness,
    )
    train_rows = convert_split(
        train_records_for_learning,
        "train",
        limit=config.max_train_samples,
        box_groups=_box_groups(train_records),
        **convert_kwargs,
    )
    val_rows = convert_split(
        val_records,
        "val",
        limit=config.max_val_samples,
        box_groups=_box_groups(val_records),
        **convert_kwargs,
    )
    test_rows = convert_split(
        test_records,
        "test",
        limit=config.max_test_samples,
        box_groups=_box_groups(test_records),
        **convert_kwargs,
    )
    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)
    write_jsonl(test_jsonl, test_rows)

    manifest = {
        "dataset_ref": config.dataset_ref,
        "class_mode": config.class_mode,
        "prompt_version": "exemplar_full_overlay_v1" if config.use_full_overlay else "exemplar_step_reasoning_v1",
        "examples_per_class": config.examples_per_class,
        "exemplar_expand_ratio": exemplar_expand_ratio,
        "exemplar_min_size": exemplar_min_size,
        "exemplar_save_size": config.exemplar_save_size,
        "exemplar_pick_strategy": config.exemplar_pick_strategy,
        "crop_adjustment": config.crop_adjustment_cfg,
        "unripe_keep_ratio": float(config.unripe_keep_ratio),
        "seed": int(config.seed),
        "use_full_overlay": bool(config.use_full_overlay),
        "overlay_font_size": int(config.overlay_font_size),
        "overlay_box_thickness": int(config.overlay_box_thickness),
        "exemplars": exemplars,
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "test_jsonl": str(test_jsonl),
        "crop_root": str(crop_root),
        "train_rows": len(train_rows),
        "train_rows_raw": len(train_records),
        "train_rows_selected": len(train_records_for_learning),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "class_definition_policy": (
            "ripe favored when red/orange dominates and no clear green remains"
            if config.class_mode == "2cls"
            else "fully-ripe favored when red dominates and no clear green remains"
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved train jsonl = {train_jsonl}")
    print(f"saved val jsonl   = {val_jsonl}")
    print(f"saved test jsonl  = {test_jsonl}")
    print(f"saved manifest    = {manifest_path}")

    return VlmDatasetResult(
        dataset_output_root=dataset_output_root,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        test_jsonl=test_jsonl,
        manifest_path=manifest_path,
        manifest=manifest,
    )
