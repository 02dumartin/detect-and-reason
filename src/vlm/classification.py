from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from src.vlm.backends import build_vlm_backend
from src.vlm.cropping import load_crop
from src.vlm.example_picker import pick_reference_examples, summarize_examples
from src.vlm.json_utils import parse_vlm_response
from src.vlm.normalization import load_or_create_normalized_detections
from src.vlm.prompting import build_messages


def run_vlm_classification(
    *,
    runtime_cfg: dict[str, Any],
    vlm_cfg: dict[str, Any],
    split_override: str | None = None,
    detector_prediction_dir_override: str | None = None,
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
    crop_cfg = vlm_cfg.get("crop", {})
    prompt_cfg = dict(vlm_cfg.get("prompt", {}))
    _log(
        "[vlm.classification] "
        f"backend={vlm_cfg.get('backend', {}).get('type')} "
        f"model={vlm_cfg.get('backend', {}).get('model_name')} "
        f"batch_size={batch_size}"
    )

    if prompt_cfg.get("use_examples"):
        dataset_cfg = runtime_cfg["raw"]["dataset"]
        picker_cfg = dict(prompt_cfg.get("example_picker") or {})
        if picker_cfg.get("enabled"):
            target_variant = str(vlm_cfg.get("task", {}).get("target_class_mode", "3cls"))
            _log(
                "[vlm.classification] "
                f"example_picker enabled variant={target_variant} "
                f"split={picker_cfg.get('source_split', 'train')} "
                f"per_class={picker_cfg.get('per_class', 1)}"
            )
            picked_examples = pick_reference_examples(
                dataset_cfg=dataset_cfg,
                project_root=runtime_cfg["project_root"],
                target_variant=target_variant,
                picker_cfg=picker_cfg,
                class_names=vlm_cfg.get("task", {}).get("class_map"),
            )
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

    total_batches = (len(records) + batch_size - 1) // batch_size if records else 0
    _log(f"[vlm.classification] starting total_records={len(records)} total_batches={total_batches}")

    result_rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        batch_number = (batch_start // batch_size) + 1
        if batch_number == 1 or batch_number == total_batches or batch_number % 10 == 0:
            _log(f"[vlm.classification] classifying batch={batch_number}/{total_batches} batch_size={len(batch)}")

        messages_batch = [
            build_messages(
                record=record,
                crop_image=load_crop(
                    record,
                    expand_ratio=float(crop_cfg.get("expand_ratio", 0.1)),
                    min_size=int(crop_cfg.get("min_size", 32)),
                ),
                prompt_cfg=prompt_cfg,
            )
            for record in batch
        ]
        raw_texts = backend.generate_batch(messages_batch)
        for record, raw_text in zip(batch, raw_texts):
            parsed = parse_vlm_response(
                text=raw_text,
                bbox_fallback_xyxy=[float(value) for value in record["bbox_xyxy"]],
                require_reasoning=bool(prompt_cfg.get("use_reasoning", True)),
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
            "type": vlm_cfg.get("backend", {}).get("type"),
            "model_name": vlm_cfg.get("backend", {}).get("model_name"),
            "batch_size": batch_size,
        },
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def _log(message: str) -> None:
    print(message, flush=True)


def _warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
