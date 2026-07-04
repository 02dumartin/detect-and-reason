from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.config_loader import load_yaml, resolve_path
from src.evaluation.vlm_classification_artifact_writer import save_vlm_classification_artifacts
from src.evaluation.vlm_classification_evaluator import evaluate_vlm_classification_predictions
from src.evaluation.yolo_label_io import resolve_yolo_split_dirs
from src.vlm.export import export_family_predictions, write_csv, write_json, write_jsonl, write_yaml
from src.vlm.overlay import save_vlm_prediction_overlays
from src.vlm.schema import CANONICAL_CLASS_ID_TO_NAME


def fmt_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def serialize_paths(payload: Any) -> Any:
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: serialize_paths(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [serialize_paths(value) for value in payload]
    return payload


def emit_classification_mismatch_logs(eval_rows: list[dict[str, Any]], limit: int = 20) -> None:
    mismatches = [
        row
        for row in eval_rows
        if row.get("row_type") == "match" and not bool(row.get("classification_correct"))
    ]
    if not mismatches:
        print("[vlm.pipeline] classification_mismatches=0", flush=True)
        return

    print(
        f"[Error] classification_mismatches={len(mismatches)} showing_up_to={min(limit, len(mismatches))}",
        file=sys.stderr,
        flush=True,
    )
    for row in mismatches[:limit]:
        print(
            "[Error] "
            f"record_id={row.get('record_id')} "
            f"file={row.get('file_name')} "
            f"gt={row.get('gt_class_name')} "
            f"pred={row.get('pred_class_name')} "
            f"iou={fmt_metric(row.get('iou'))} "
            f"det_score={fmt_metric(row.get('det_score'))}",
            file=sys.stderr,
            flush=True,
        )
        reasoning = str(row.get("reasoning") or "").strip()
        raw_response = str(row.get("raw_response") or "").strip()
        if reasoning:
            print(f"Reasoning: {reasoning}", file=sys.stderr, flush=True)
        if raw_response and raw_response != reasoning:
            print(f"Raw response: {raw_response}", file=sys.stderr, flush=True)


def save_classification_outputs(output_dir: Path, classification_result: dict[str, Any]) -> None:
    write_json(output_dir / "normalized_input_detections.json", classification_result["normalized_bundle"])
    write_jsonl(output_dir / "vlm_results.jsonl", classification_result["result_rows"])
    write_json(output_dir / "prediction_class.json", classification_result["result_rows"])
    print(f"[vlm.pipeline] saved_results rows={len(classification_result['result_rows'])}", flush=True)


def run_family_export(
    *,
    runtime_cfg: dict[str, Any],
    predictions: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    return export_family_predictions(
        family=str(runtime_cfg.get("family") or ""),
        predictions=predictions,
        output_dir=output_dir,
    )


def run_evaluation(
    *,
    runtime_cfg: dict[str, Any],
    vlm_cfg: dict[str, Any],
    classification_result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    dataset_cfg = runtime_cfg["raw"]["dataset"]
    target_variant = str(vlm_cfg.get("task", {}).get("target_class_mode", "3cls"))
    split = str(classification_result["split"])
    print(f"[vlm.pipeline] evaluating target_variant={target_variant}", flush=True)

    target_split_path = resolve_target_split_path(
        runtime_cfg=runtime_cfg,
        target_variant=target_variant,
        split=split,
    )
    gt_images_dir, gt_labels_dir = resolve_yolo_split_dirs(target_split_path)
    class_names = (
        vlm_cfg.get("task", {}).get("class_map")
        or dataset_cfg["variants"][target_variant].get("class_names")
        or CANONICAL_CLASS_ID_TO_NAME
    )
    evaluation_output = evaluate_vlm_classification_predictions(
        gt_images_dir=gt_images_dir,
        gt_labels_dir=gt_labels_dir,
        predictions=classification_result["result_rows"],
        class_names=class_names,
        iou_threshold=float(runtime_cfg.get("predict", {}).get("resolved_iou") or 0.5),
    )
    write_json(output_dir / "vlm_eval_rows.json", evaluation_output["eval_rows"])
    write_csv(output_dir / "vlm_eval_rows.csv", evaluation_output["eval_rows"])
    write_json(output_dir / "metrics.json", evaluation_output["metrics"])
    save_vlm_classification_artifacts(evaluation_output, output_dir / "evaluation")
    return evaluation_output


def run_overlay(
    *,
    runtime_cfg: dict[str, Any],
    vlm_cfg: dict[str, Any],
    predictions: list[dict[str, Any]],
    output_dir: Path,
) -> int:
    overlay_cfg_ref = vlm_cfg.get("output", {}).get("overlay_config")
    overlay_cfg = None
    if overlay_cfg_ref:
        overlay_path = resolve_path(overlay_cfg_ref, runtime_cfg["project_root"])
        if overlay_path is None or not overlay_path.exists():
            raise FileNotFoundError(f"overlay config not found: {overlay_cfg_ref}")
        overlay_cfg = load_yaml(overlay_path)

    return save_vlm_prediction_overlays(
        predictions=predictions,
        output_dir=output_dir / "overlay_images",
        overlay_config=overlay_cfg,
    )


def build_runtime_summary(
    *,
    runtime_cfg: dict[str, Any],
    output_dir: Path,
    classification_result: dict[str, Any],
    family_export: dict[str, Any],
    overlay_rendered: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "model_name": runtime_cfg["model_name"],
        "dataset_key": runtime_cfg["dataset"]["key"],
        "split": classification_result["split"],
        "prediction_dir": str(classification_result["prediction_dir"]),
        "output_dir": str(output_dir),
        "normalized_cache_path": str(classification_result["normalized_cache_path"]),
        "normalized_cache_hit": classification_result["normalized_cache_hit"],
        "num_input_detections": classification_result["num_input_detections"],
        "num_vlm_results": classification_result["num_vlm_results"],
        "overlay_rendered": overlay_rendered,
        "family_export": family_export,
        "backend": classification_result["backend"],
        "elapsed_sec": elapsed_sec,
    }


def save_run_metadata(
    *,
    output_dir: Path,
    runtime_cfg: dict[str, Any],
    vlm_cfg: dict[str, Any],
    runtime_summary: dict[str, Any],
) -> None:
    write_yaml(
        output_dir / "run_config_input.yaml",
        {
            "runtime": serialize_paths(runtime_cfg),
            "vlm": vlm_cfg,
        },
    )
    write_json(output_dir / "runtime.json", runtime_summary)


def resolve_target_split_path(
    *,
    runtime_cfg: dict[str, Any],
    target_variant: str,
    split: str,
) -> Path:
    dataset_cfg = runtime_cfg["raw"]["dataset"]
    variants = dataset_cfg.get("variants", {})
    if target_variant not in variants:
        raise ValueError(f"target variant '{target_variant}' is missing in dataset config")

    split_path = variants[target_variant].get(split)
    if not split_path:
        raise ValueError(f"split '{split}' is missing for target variant '{target_variant}'")

    resolved = resolve_path(split_path, runtime_cfg["project_root"])
    if resolved is None:
        raise ValueError(f"could not resolve target split path: {split_path}")
    return resolved
