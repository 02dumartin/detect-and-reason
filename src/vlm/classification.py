from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from src.config_loader import load_yaml, resolve_dataset_config_path
from src.vlm.backends import build_vlm_backend
from src.vlm.box_context import (
    assign_box_numbers,
    box_number_for_record,
    build_bbox_list_text,
    render_numbered_overlay_image,
)
from src.vlm.color_signal import (
    SWATCH5_TO_3CLS_MAPPING,
    build_attached_swatch_note_with_mapping,
    build_color_signal_text,
    build_swatch_intro_text,
    build_swatch_palette,
    compose_crop_with_vertical_swatch,
    compute_whole_crop_color_index,
    nearest_swatch_label,
    render_swatch_strip_image,
    sample_multi_point_colors,
)
from src.vlm.cropping import load_crop
from src.vlm.example_picker import load_examples_from_manifest, pick_reference_examples, summarize_examples
from src.vlm.json_utils import parse_vlm_response
from src.vlm.normalization import load_or_create_normalized_detections
from src.vlm.prompting import build_messages


def run_vlm_classification(
    *,
    runtime_cfg: dict[str, Any],
    vlm_cfg: dict[str, Any],
    split_override: str | None = None,
    detector_prediction_dir_override: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    split = split_override or vlm_cfg.get("input", {}).get("split") or "test"
    _log(f"[vlm.classification] split={split}")

    prediction_dir = (
        Path(detector_prediction_dir_override).resolve()
        if detector_prediction_dir_override
        else Path(vlm_cfg.get("input", {}).get("prediction_dir") or runtime_cfg["paths"]["prediction_dir"]).resolve()
    )
    _log(f"[vlm.classification] prediction_dir={prediction_dir}")

    normalized_bundle, cache_path, cache_hit = load_or_create_normalized_detections(
        runtime_cfg=runtime_cfg,
        split=split,
        prediction_dir=prediction_dir,
        refresh=bool(vlm_cfg.get("input", {}).get("refresh_normalized", False)),
    )
    _log(f"[vlm.classification] normalized_cache={cache_path} cache_hit={cache_hit}")

    records = list(normalized_bundle.get("records", []))
    _log(f"[vlm.classification] normalized_records={len(records)}")
    max_samples = vlm_cfg.get("input", {}).get("max_samples")
    if max_samples is not None:
        records = records[: int(max_samples)]
        _log(f"[vlm.classification] max_samples_applied={len(records)}")

    backend = build_vlm_backend(
        backend_cfg=vlm_cfg.get("backend", {}),
        generation_cfg=vlm_cfg.get("generation", {}),
    )
    batch_size = max(1, int(vlm_cfg.get("backend", {}).get("batch_size", 1)))
    backend_cfg = dict(vlm_cfg.get("backend", {}))
    crop_cfg = vlm_cfg.get("crop", {})
    prompt_cfg = dict(vlm_cfg.get("prompt", {}))
    task_class_map = vlm_cfg.get("task", {}).get("class_map")
    if task_class_map:
        prompt_cfg["class_map"] = task_class_map
    _log(
        "[vlm.classification] "
        f"backend={backend_cfg.get('type')} "
        f"model={backend_cfg.get('model_name')} "
        f"model_path={backend_cfg.get('model_path')} "
        f"adapter_path={backend_cfg.get('adapter_path')} "
        f"batch_size={batch_size}"
    )

    if prompt_cfg.get("use_examples"):
        picker_cfg = dict(prompt_cfg.get("example_picker") or {})
        manifest_path = picker_cfg.get("manifest_path")
        if manifest_path:
            picked_examples = load_examples_from_manifest(
                manifest_path,
                project_root=runtime_cfg["project_root"],
            )
            prompt_cfg["examples"] = picked_examples
            _log(
                "[vlm.classification] "
                f"example_picker manifest_path={manifest_path} "
                f"picked_examples={summarize_examples(picked_examples)}"
            )
            if not picked_examples:
                _warn("[vlm.classification] manifest_path configured but no exemplars found in it")
        elif picker_cfg.get("enabled"):
            target_variant = str(
                picker_cfg.get("source_class_mode")
                or vlm_cfg.get("task", {}).get("target_class_mode", "3cls")
            )
            example_adjustment_cfg = picker_cfg.get("adjustment", crop_cfg.get("adjustment"))
            _log(
                "[vlm.classification] "
                f"example_picker enabled variant={target_variant} "
                f"split={picker_cfg.get('source_split', 'train')} "
                f"per_class={picker_cfg.get('per_class', 1)}"
            )
            picked_examples = []
            picker_errors: list[str] = []
            for candidate_ref, dataset_cfg in _iter_example_picker_dataset_cfgs(runtime_cfg, picker_cfg):
                try:
                    picked_examples = pick_reference_examples(
                        dataset_cfg=dataset_cfg,
                        project_root=runtime_cfg["project_root"],
                        target_variant=target_variant,
                        picker_cfg=picker_cfg,
                        class_names=task_class_map,
                        adjustment_cfg=example_adjustment_cfg,
                    )
                except (FileNotFoundError, ValueError) as exc:
                    picker_errors.append(f"{candidate_ref}: {exc}")
                    continue

                _log(
                    "[vlm.classification] "
                    f"example_picker dataset_ref={candidate_ref}"
                )
                break

            if picker_errors and not picked_examples:
                _warn("[vlm.classification] example_picker fallback chain exhausted:")
                for message in picker_errors:
                    _warn(f"[vlm.classification]   {message}")

            prompt_cfg["examples"] = picked_examples
            _log(f"[vlm.classification] picked_examples={summarize_examples(picked_examples)}")
            if not picked_examples:
                _warn("[vlm.classification] example_picker enabled but no reference examples were found")
        else:
            static_examples = list(prompt_cfg.get("examples") or [])
            if static_examples:
                _log(f"[vlm.classification] using_static_examples count={len(static_examples)}")
            else:
                _warn("[vlm.classification] use_examples=true but no static examples or example_picker are configured")

    use_full_overlay = bool(prompt_cfg.get("use_full_overlay"))
    box_groups = (
        assign_box_numbers(records, group_key="image_path", id_key="record_id") if use_full_overlay else {}
    )
    overlay_image_cache: dict[str, Any] = {}
    overlay_font_size = int(prompt_cfg.get("overlay_font_size", 36))
    overlay_box_thickness = int(prompt_cfg.get("overlay_box_thickness", 3))
    if use_full_overlay:
        _log(f"[vlm.classification] use_full_overlay=true images={len(box_groups)}")

    def _build_full_overlay_context(record: dict[str, Any]) -> tuple[Any, str | None, int | None]:
        if not use_full_overlay:
            return None, None, None
        image_key = record["image_path"]
        boxes = box_groups[image_key]
        box_number = box_number_for_record(box_groups, group_value=image_key, record_id=record["record_id"])
        bbox_list_text = build_bbox_list_text(boxes, target_box_number=box_number)
        overlay_image = overlay_image_cache.get(image_key)
        if overlay_image is None:
            overlay_image = render_numbered_overlay_image(
                image_key,
                boxes,
                font_size=overlay_font_size,
                box_thickness=overlay_box_thickness,
                adjustment_cfg=crop_cfg.get("adjustment"),
            )
            overlay_image_cache[image_key] = overlay_image
        return overlay_image, bbox_list_text, box_number

    color_signal_cfg = dict(prompt_cfg.get("color_signal") or {})
    color_signal_mode = str(color_signal_cfg.get("mode") or "none")
    swatch_palette: list[dict[str, Any]] = []
    swatch_strip_image: Any = None
    swatch_intro_text: str | None = None
    swatch_attach_to_crop = False
    swatch_width_cfg = 60
    swatch_gap_cfg = 6
    attached_swatch_note: str | None = None
    swatch_debug_dir: Path | None = None
    if color_signal_mode == "swatch_match":
        swatch_cfg = dict(color_signal_cfg.get("swatch") or {})
        swatch_palette = build_swatch_palette(
            count=int(swatch_cfg.get("count", 6)),
            hue_range_deg=tuple(swatch_cfg.get("hue_range_deg", [20.0, 130.0])),
            lightness=float(swatch_cfg.get("lightness", 55.0)),
            chroma=float(swatch_cfg.get("chroma", 55.0)),
        )
        swatch_attach_to_crop = bool(swatch_cfg.get("attach_to_crop", False))
        swatch_width_cfg = int(swatch_cfg.get("swatch_width", 60))
        swatch_gap_cfg = int(swatch_cfg.get("gap", 6))
        if swatch_attach_to_crop:
            attached_swatch_note = build_attached_swatch_note_with_mapping(swatch_palette, SWATCH5_TO_3CLS_MAPPING)
        else:
            swatch_strip_image = render_swatch_strip_image(swatch_palette)
            swatch_intro_text = build_swatch_intro_text(swatch_palette)
        if bool(swatch_cfg.get("save_debug_images", False)) and output_dir is not None:
            swatch_debug_dir = Path(output_dir) / "swatch_debug_images"
            swatch_debug_dir.mkdir(parents=True, exist_ok=True)
        _log(
            "[vlm.classification] "
            f"color_signal=swatch_match labels={[s['label'] for s in swatch_palette]} "
            f"attach_to_crop={swatch_attach_to_crop} debug_dir={swatch_debug_dir}"
        )
    elif color_signal_mode in {"whole_crop_index", "multi_point_sample"}:
        _log(f"[vlm.classification] color_signal={color_signal_mode}")

    def _build_color_signal_context(crop_image: Any, record_id: str) -> dict[str, Any]:
        if color_signal_mode == "swatch_match":
            whole_index = compute_whole_crop_color_index(crop_image, adjustment_cfg=crop_cfg.get("adjustment"))
            log_entry = {
                "mode": color_signal_mode,
                "nearest_swatch_label": nearest_swatch_label(whole_index["hue_deg"], swatch_palette),
                **whole_index,
            }
            if swatch_attach_to_crop:
                combined_crop = compose_crop_with_vertical_swatch(
                    crop_image, swatch_palette, swatch_width=swatch_width_cfg, gap=swatch_gap_cfg
                )
                if swatch_debug_dir is not None:
                    safe_id = record_id.replace("/", "_").replace("#", "_")
                    combined_crop.save(swatch_debug_dir / f"{safe_id}.png")
                return {
                    "crop_image": combined_crop,
                    "color_signal_image": None,
                    "color_signal_intro_text": None,
                    "color_signal_text": None,
                    "crop_intro_note": attached_swatch_note,
                    "log_entry": log_entry,
                }
            return {
                "crop_image": crop_image,
                "color_signal_image": swatch_strip_image,
                "color_signal_intro_text": swatch_intro_text,
                "color_signal_text": None,
                "crop_intro_note": None,
                "log_entry": log_entry,
            }
        if color_signal_mode == "whole_crop_index":
            whole_index = compute_whole_crop_color_index(crop_image, adjustment_cfg=crop_cfg.get("adjustment"))
            return {
                "crop_image": crop_image,
                "color_signal_image": None,
                "color_signal_intro_text": None,
                "color_signal_text": build_color_signal_text(color_signal_mode, whole_index),
                "crop_intro_note": None,
                "log_entry": {"mode": color_signal_mode, **whole_index},
            }
        if color_signal_mode == "multi_point_sample":
            multi_point_cfg = dict(color_signal_cfg.get("multi_point") or {})
            sampled = sample_multi_point_colors(
                crop_image,
                num_points=int(multi_point_cfg.get("num_points", 7)),
                exclude_border_ratio=float(multi_point_cfg.get("exclude_border_ratio", 0.15)),
                patch_radius=int(multi_point_cfg.get("patch_radius", 2)),
                uniformity_thresholds=multi_point_cfg.get("uniformity_thresholds"),
                adjustment_cfg=crop_cfg.get("adjustment"),
            )
            log_entry = {
                "mode": color_signal_mode,
                "median_hue_deg": sampled["median_hue_deg"],
                "median_chroma": sampled["median_chroma"],
                "hue_std_deg": sampled["hue_std_deg"],
                "uniformity_label": sampled["uniformity_label"],
            }
            return {
                "crop_image": crop_image,
                "color_signal_image": None,
                "color_signal_intro_text": None,
                "color_signal_text": build_color_signal_text(color_signal_mode, sampled),
                "crop_intro_note": None,
                "log_entry": log_entry,
            }
        return {
            "crop_image": crop_image,
            "color_signal_image": None,
            "color_signal_intro_text": None,
            "color_signal_text": None,
            "crop_intro_note": None,
            "log_entry": None,
        }

    total_batches = (len(records) + batch_size - 1) // batch_size if records else 0
    _log(f"[vlm.classification] starting total_records={len(records)} total_batches={total_batches}")

    result_rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        batch_number = (batch_start // batch_size) + 1
        if batch_number == 1 or batch_number == total_batches or batch_number % 10 == 0:
            _log(f"[vlm.classification] classifying batch={batch_number}/{total_batches} batch_size={len(batch)}")

        messages_batch = []
        color_signal_logs: list[dict[str, Any] | None] = []
        for record in batch:
            full_overlay_image, bbox_list_text, target_box_number = _build_full_overlay_context(record)
            crop_resize = crop_cfg.get("resize")
            crop_image = load_crop(
                record,
                expand_ratio=float(crop_cfg.get("expand_ratio", 0.1)),
                min_size=int(crop_cfg.get("min_size", 32)),
                adjustment_cfg=crop_cfg.get("adjustment"),
                resize=int(crop_resize) if crop_resize else None,
            )
            color_signal_ctx = _build_color_signal_context(crop_image, record["record_id"])
            color_signal_logs.append(color_signal_ctx["log_entry"])
            messages_batch.append(
                build_messages(
                    record=record,
                    crop_image=color_signal_ctx["crop_image"],
                    prompt_cfg=prompt_cfg,
                    full_overlay_image=full_overlay_image,
                    bbox_list_text=bbox_list_text,
                    target_box_number=target_box_number,
                    color_signal_image=color_signal_ctx["color_signal_image"],
                    color_signal_intro_text=color_signal_ctx["color_signal_intro_text"],
                    color_signal_text=color_signal_ctx["color_signal_text"],
                    crop_intro_note=color_signal_ctx["crop_intro_note"],
                )
            )
        raw_texts = backend.generate_batch(messages_batch)
        for record, raw_text, color_signal_log in zip(batch, raw_texts, color_signal_logs):
            parsed = parse_vlm_response(
                text=raw_text,
                bbox_fallback_xyxy=[float(value) for value in record["bbox_xyxy"]],
                require_reasoning=bool(prompt_cfg.get("use_reasoning", True)),
                class_name_map=task_class_map,
            )
            result_rows.append(
                {
                    **record,
                    "predicted_class_id": int(parsed["class_id"]),
                    "predicted_class_name": str(parsed["class_name"]),
                    "reasoning": str(parsed["reasoning"]),
                    "raw_response": str(parsed["raw_response"]),
                    "parse_error": parsed["parse_error"],
                    "response_ok": bool(parsed["ok"]),
                    "bbox_xyxy": [float(value) for value in parsed["bbox_xyxy"]],
                    "color_signal": color_signal_log,
                }
            )
            if not parsed["ok"]:
                _warn(
                    "[Error] "
                    f"record_id={record['record_id']} "
                    f"file={record.get('file_name')} "
                    f"parse_error={parsed['parse_error']}"
                )

    _log(f"[vlm.classification] complete rows={len(result_rows)}")
    return {
        "split": split,
        "prediction_dir": prediction_dir,
        "normalized_bundle": normalized_bundle,
        "normalized_cache_path": cache_path,
        "normalized_cache_hit": cache_hit,
        "result_rows": result_rows,
        "num_input_detections": len(records),
        "num_vlm_results": len(result_rows),
        "backend": {
            "type": backend_cfg.get("type"),
            "model_name": backend_cfg.get("model_name"),
            "model_path": backend_cfg.get("model_path"),
            "adapter_path": backend_cfg.get("adapter_path"),
            "batch_size": batch_size,
        },
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _log(message: str) -> None:
    print(message, flush=True)


def _warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _infer_example_picker_dataset_ref(runtime_cfg: dict[str, Any]) -> str | None:
    """예시 샘플용 데이터셋이 명시되지 않았을 때 안전한 기본값을 고른다.

    merge_little / merge_tomatod 같은 cross-eval 설정은 평가용 test split은
    각각 little / tomatod 3cls를 쓰지만, 예시 샘플은 실제 존재하는 source
    데이터셋의 train split에서 뽑는 편이 자연스럽다. 이 경우 dataset_ref를
    따로 안 적어도 자동으로 맞춰 준다.
    """
    dataset_key = str(runtime_cfg.get("dataset", {}).get("key") or "")
    if dataset_key == "merge_little":
        return "little"
    if dataset_key == "merge_tomatod":
        return "tomatod"
    return None


def _iter_example_picker_dataset_cfgs(
    runtime_cfg: dict[str, Any],
    picker_cfg: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """예시 선택에 사용할 데이터셋 후보를 순서대로 반환한다.

    1. 설정에서 명시한 dataset_ref
    2. 특수 cross-eval 케이스에 대한 암시적 매핑
    3. 현재 runtime dataset 자체

    앞선 후보가 존재하지 않거나 필요한 split/variant가 없으면 뒤 후보를 시도한다.
    """
    candidates: list[str] = []
    explicit_ref = str(picker_cfg.get("dataset_ref") or "").strip()
    if explicit_ref:
        candidates.append(explicit_ref)

    inferred_ref = _infer_example_picker_dataset_ref(runtime_cfg)
    if inferred_ref and inferred_ref not in candidates:
        candidates.append(inferred_ref)

    runtime_ref = str(runtime_cfg.get("dataset", {}).get("key") or "").strip()
    if runtime_ref and runtime_ref not in candidates:
        candidates.append(runtime_ref)

    resolved: list[tuple[str, dict[str, Any]]] = []
    for candidate_ref in candidates:
        try:
            picker_dataset_path = resolve_dataset_config_path(candidate_ref)
        except FileNotFoundError:
            continue
        resolved.append((candidate_ref, load_yaml(picker_dataset_path)))
    return resolved
