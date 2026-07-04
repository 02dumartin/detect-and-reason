from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config_loader import load_yaml, resolve_path
from src.vlm.schema import CANONICAL_CLASS_ID_TO_NAME


DEFAULT_VLM_CONFIG: dict[str, Any] = {
    "name": "default",
    "task": {
        "target_class_mode": "3cls",
        "class_map": deepcopy(CANONICAL_CLASS_ID_TO_NAME),
    },
    "backend": {
        "type": "qwen_hf",
        "model_name": "Qwen/Qwen3-VL-4B-Instruct",
        "batch_size": 1,
        "use_4bit": True,
        "compile_model": False,
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": True,
        "api_key_env": "OPENAI_API_KEY",
    },
    "generation": {
        "max_new_tokens": 220,
        "repetition_penalty": 1.1,
        "do_sample": False,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
    },
    "input": {
        "split": "test",
        "prediction_dir": None,
        "refresh_normalized": False,
        "max_samples": None,
    },
    "crop": {
        "expand_ratio": 0.1,
        "min_size": 32,
    },
    "prompt": {
        "use_reasoning": True,
        "use_examples": False,
        "use_color_guide": True,
        "include_bbox_json": True,
        "reasoning_max_words": 25,
        "system_message": None,
        "instruction": None,
        "examples": [],
        "example_picker": {
            "enabled": False,
            "source_split": "train",
            "per_class": 2,
            "max_total_examples": None,
            "resize": 224,
            "expand_ratio": 0.1,
            "min_size": 32,
        },
    },
    "output": {
        "save_overlay": True,
        "save_eval": True,
        "save_family_export": True,
        "overlay_config": "config/visualization/detection_overlay.yaml",
    },
}


def load_vlm_config(config_ref: str | None, *, project_root: Path) -> tuple[dict[str, Any], Path | None]:
    cfg = deepcopy(DEFAULT_VLM_CONFIG)
    if not config_ref:
        return cfg, None

    path = resolve_vlm_config_path(config_ref, project_root=project_root)
    loaded = load_yaml(path)
    merged = merge_dicts(cfg, loaded)
    return merged, path


def resolve_vlm_config_path(config_ref: str, *, project_root: Path) -> Path:
    ref_path = Path(config_ref)
    if ref_path.suffix in {".yaml", ".yml"} or "/" in config_ref or config_ref.startswith("."):
        resolved = resolve_path(ref_path, project_root)
        if resolved is not None and resolved.exists():
            return resolved

    candidate = project_root / "config" / "vlm" / f"{config_ref}.yaml"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"vlm config not found: {config_ref}")


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def derive_output_dir(runtime_cfg: dict[str, Any], output_dir_override: str | None = None) -> Path:
    if output_dir_override:
        return Path(output_dir_override).resolve()

    qwen_dir = runtime_cfg["paths"].get("qwen_dir")
    if qwen_dir:
        return Path(qwen_dir).resolve()

    project_root = runtime_cfg["project_root"]
    pipeline_mode = runtime_cfg.get("pipeline_mode") or "detection_reasoning"
    dataset_key = runtime_cfg["dataset"]["key"]
    model_name = runtime_cfg["model_name"]
    return (project_root / f"result/{pipeline_mode}/{dataset_key}/{model_name}_qwen").resolve()
