from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import ImageOps, ImageStat

from src.config_loader import resolve_path
from src.evaluation.yolo_label_io import (
    build_image_index,
    load_yolo_label_rows,
    resolve_yolo_split_dirs,
    yolo_row_to_detection,
)
from src.vlm.cropping import load_crop
from src.vlm.schema import CANONICAL_CLASS_ID_TO_NAME


def load_examples_from_manifest(manifest_path: str | Path, *, project_root: Path) -> list[dict[str, Any]]:
    """Load the exact exemplar crops recorded in a training manifest.json
    (written by scripts/vlm_full_ft_with_exemplars.py's save_exemplar_crops).

    A full-FT checkpoint was trained on these specific images baked into every
    training row; pick_reference_examples() re-derives a fresh (and generally
    different) set from raw dataset files, so it must not be used for a
    full-FT checkpoint's inference/eval.
    """
    resolved = resolve_path(str(manifest_path), project_root) or Path(manifest_path)
    with Path(resolved).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    examples: list[dict[str, Any]] = []
    for exemplar in manifest.get("exemplars", []):
        examples.append(
            {
                "path": exemplar["path"],
                "label": exemplar["class_name"],
                "condition": exemplar.get("sample_id", "unknown"),
            }
        )
    return examples


def pick_reference_examples(
    *,
    dataset_cfg: dict[str, Any],
    project_root: Path,
    target_variant: str,
    picker_cfg: dict[str, Any],
    class_names: dict[int, str] | dict[str, str] | list[str] | None = None,
    adjustment_cfg: dict[str, Any] | str | None = None,
) -> list[dict[str, Any]]:
    variants = dataset_cfg.get("variants", {})
    if target_variant not in variants:
        raise ValueError(f"target variant '{target_variant}' is missing in dataset config")

    source_split = str(picker_cfg.get("source_split") or "train")
    split_path = variants[target_variant].get(source_split)
    resolved_split_path = resolve_path(split_path, project_root)
    if resolved_split_path is None:
        raise ValueError(f"could not resolve split '{source_split}' for target variant '{target_variant}'")

    try:
        images_dir, labels_dir = resolve_yolo_split_dirs(resolved_split_path)
        image_index = build_image_index(images_dir)
    except FileNotFoundError as exc:
        print(
            f"[warn] example_picker skipped for dataset_ref={dataset_cfg.get('name', 'unknown')} "
            f"variant={target_variant}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return []
    except ValueError as exc:
        print(
            f"[warn] example_picker skipped for dataset_ref={dataset_cfg.get('name', 'unknown')} "
            f"variant={target_variant}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return []

    class_map = _resolve_class_map(class_names or variants[target_variant].get("class_names"))
    target_labels = [class_map[class_id] for class_id in sorted(class_map)]

    per_class = max(1, int(picker_cfg.get("per_class", 1)))
    max_total_examples = _parse_optional_positive_int(picker_cfg.get("max_total_examples"))
    expand_ratio = float(picker_cfg.get("expand_ratio", 0.1))
    min_size = int(picker_cfg.get("min_size", 32))
    resize = _parse_optional_positive_int(picker_cfg.get("resize"))
    min_brightness = float(picker_cfg.get("min_brightness", 0.0))

    selected_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for stem, image_info in image_index.items():
        rows = load_yolo_label_rows(labels_dir / f"{stem}.txt")
        for row_index, row in enumerate(rows):
            class_id, bbox_xyxy, _ = yolo_row_to_detection(row, image_info.width, image_info.height)
            label = class_map.get(int(class_id))
            if not label or selected_counts[label] >= per_class:
                continue

            crop_image = load_crop(
                {
                    "image_path": str(image_info.path),
                    "bbox_xyxy": bbox_xyxy,
                },
                expand_ratio=expand_ratio,
                min_size=min_size,
                adjustment_cfg=adjustment_cfg,
            )
            if min_brightness > 0.0:
                brightness = ImageStat.Stat(crop_image.convert("L")).mean[0]
                if brightness < min_brightness:
                    continue
            if resize is not None:
                crop_image = ImageOps.contain(crop_image, (resize, resize))

            examples.append(
                {
                    "image": crop_image,
                    "label": label,
                    "condition": f"{source_split}:{stem}#{row_index}",
                    "source_image": str(image_info.path),
                }
            )
            selected_counts[label] += 1

            if max_total_examples is not None and len(examples) >= max_total_examples:
                return examples
            if all(selected_counts[target_label] >= per_class for target_label in target_labels):
                return examples

    return examples


def summarize_examples(examples: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(example.get("label", "unknown")) for example in examples)
    return dict(sorted(counts.items()))


def _resolve_class_map(raw_class_names: dict[int, str] | dict[str, str] | list[str] | None) -> dict[int, str]:
    if raw_class_names is None:
        return dict(CANONICAL_CLASS_ID_TO_NAME)

    if isinstance(raw_class_names, dict):
        items = sorted(raw_class_names.items(), key=lambda item: int(item[0]))
    else:
        items = list(enumerate(raw_class_names))

    resolved: dict[int, str] = {}
    for raw_class_id, raw_class_name in items:
        class_id = int(raw_class_id)
        label = str(raw_class_name).strip()
        if not label:
            label = str(class_id)
        resolved[class_id] = label
    return resolved


def _parse_optional_positive_int(value: Any) -> int | None:
    if value in {None, "", 0, "0"}:
        return None
    return max(1, int(value))
