from __future__ import annotations

from pathlib import Path

import yaml


def get_project_root() -> Path:
    """현재 파일 위치를 기준으로 프로젝트 루트를 계산한다."""
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path | None, project_root: Path | None = None) -> Path | None:
    """설정값으로 들어온 경로를 프로젝트 루트 기준의 실제 경로로 풀어낸다.

    절대 경로가 들어오면 그대로도 시도하고, `/`를 제거한 뒤 프로젝트 루트
    아래에 같은 구조가 있는지도 함께 확인한다. 상대 경로는 항상 프로젝트
    루트 기준으로 해석한다.
    """
    if value is None:
        return None

    root = project_root or get_project_root()
    raw = Path(value)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
        try:
            candidates.append(root / raw.relative_to("/"))
        except ValueError:
            pass
    else:
        candidates.append(root / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[-1].resolve() if candidates else raw


def resolve_weight_ref(value: str | Path | None, project_root: Path | None = None) -> str | None:
    """가중치 참조값을 문자열 경로로 정규화한다."""
    if value is None:
        return None

    resolved = resolve_path(value, project_root)
    if resolved is not None and resolved.exists():
        return str(resolved)

    return str(value)


def ensure_images_dir(path: Path | None) -> Path | None:
    """데이터셋 split 루트가 들어오면 실제 이미지 폴더까지 내려가 준다."""
    if path is None:
        return None
    if path.is_dir() and (path / "images").exists():
        return (path / "images").resolve()
    return path.resolve()


def load_yaml(path: Path) -> dict:
    """YAML 파일을 읽고, 비어 있으면 빈 dict를 반환한다."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_model_config_path(model_ref: str) -> Path:
    """모델 설정 이름 또는 파일 경로를 실제 YAML 파일 경로로 바꾼다."""
    return _resolve_config_path(model_ref, "config/model")


def resolve_dataset_config_path(dataset_ref: str) -> Path:
    """데이터셋 설정 이름 또는 파일 경로를 실제 YAML 파일 경로로 바꾼다."""
    return _resolve_config_path(dataset_ref, "config/dataset")


def resolve_runtime_config(
    *,
    model_ref: str,
    dataset_ref: str,
    stage: str,
    weights: str | None = None,
    source: str | None = None,
) -> dict:
    """모델 설정과 데이터셋 설정을 합쳐 stage 실행용 런타임 설정을 만든다."""
    project_root = get_project_root()
    model_path = resolve_model_config_path(model_ref)
    dataset_path = resolve_dataset_config_path(dataset_ref)

    model_cfg = load_yaml(model_path)
    dataset_cfg = load_yaml(dataset_path)

    dataset_key = dataset_path.stem
    raw_class_mode = model_cfg.get("class_mode")
    class_mode = str(raw_class_mode).strip() if raw_class_mode is not None else ""
    if not class_mode:
        raise ValueError(f"class_mode is missing in {model_path}")

    variants = dataset_cfg.get("variants", {})
    if class_mode not in variants:
        raise ValueError(
            f"dataset variant '{class_mode}' is missing in {dataset_path}. "
            f"Available variants: {sorted(variants.keys())}"
        )

    variant = variants[class_mode]
    dataset_override = model_cfg.get("dataset_overrides", {}).get(dataset_key, {})

    train_dir = _resolve_variant_path(variant, "train", project_root)
    val_dir = _resolve_variant_path(variant, "val", project_root)
    test_dir = _resolve_variant_path(variant, "test", project_root)
    data_yaml = _resolve_first_existing_like(
        dataset_override.get("data_yaml"),
        variant.get("data_yaml"),
        project_root=project_root,
    )

    train_cfg = dict(model_cfg.get("train", {}))
    val_cfg = dict(model_cfg.get("val", {}))
    predict_cfg = dict(model_cfg.get("predict", {}))

    output_mode = model_cfg.get("mode", "detection_only")
    model_name = model_cfg.get("name", model_path.stem)

    runs_dir = _resolve_first_existing_like(
        dataset_override.get("runs_dir"),
        f"runs/{dataset_key}/{model_name}",
        project_root=project_root,
    )
    eval_dir = _resolve_first_existing_like(
        dataset_override.get("eval_dir"),
        f"result/{output_mode}/{dataset_key}/{model_name}_eval",
        project_root=project_root,
    )
    prediction_dir = _resolve_first_existing_like(
        dataset_override.get("prediction_dir"),
        f"result/{output_mode}/{dataset_key}/{model_name}_prediction",
        project_root=project_root,
    )
    qwen_dir = _resolve_optional_path(
        dataset_override.get("qwen_dir"),
        project_root=project_root,
    )

    resolved_weights = _resolve_runtime_weights(
        stage=stage,
        cli_weights=weights,
        init_weight=model_cfg.get("model", {}).get("init_weight"),
        runs_dir=runs_dir,
        model_name=model_name,
        project_root=project_root,
    )
    resolved_val_split = _resolve_split_name(variant, val_cfg.get("split", "test"))
    resolved_predict_source = _resolve_predict_source(
        variant=variant,
        dataset_override=dataset_override,
        source=source,
        project_root=project_root,
    )

    runtime = {
        "project_root": project_root,
        "stage": stage,
        "model_config_path": model_path,
        "dataset_config_path": dataset_path,
        "model_name": model_name,
        "family": model_cfg.get("family"),
        "framework": model_cfg.get("framework"),
        "class_mode": class_mode,
        "pipeline_mode": output_mode,
        "model": dict(model_cfg.get("model", {})),
        "prompts": list(model_cfg.get("prompts", [])),
        "class_names": dict(model_cfg.get("class_names", {})),
        "metrics": list(model_cfg.get("metrics", [])),
        "train": train_cfg,
        "val": {
            **val_cfg,
            "resolved_iou": _resolve_iou(val_cfg, dataset_key),
            "resolved_split": resolved_val_split,
        },
        "predict": {
            **predict_cfg,
            "resolved_iou": _resolve_iou(predict_cfg, dataset_key),
            "resolved_source": resolved_predict_source,
        },
        "dataset": {
            "key": dataset_key,
            "name": dataset_cfg.get("name", dataset_key),
            "variant": class_mode,
            "path": _resolve_optional_path(variant.get("path"), project_root=project_root),
            "train_dir": train_dir,
            "val_dir": val_dir,
            "test_dir": test_dir,
            "data_yaml": data_yaml,
            "cache": dataset_override.get("cache"),
            "class_names": dict(variant.get("class_names", {})),
        },
        "paths": {
            "runs_dir": runs_dir,
            "eval_dir": eval_dir,
            "prediction_dir": prediction_dir,
            "qwen_dir": qwen_dir,
            "weights": resolved_weights,
        },
        "raw": {
            "model": model_cfg,
            "dataset": dataset_cfg,
            "dataset_override": dataset_override,
        },
    }

    _validate_runtime(runtime)
    return runtime


def _resolve_config_path(ref: str, base_dir: str) -> Path:
    """설정 이름 또는 파일 경로를 받아 실제 설정 파일을 찾는다."""
    project_root = get_project_root()
    ref_path = Path(ref)

    if ref_path.suffix in {".yaml", ".yml"} or "/" in ref or ref.startswith("."):
        candidate = resolve_path(ref_path, project_root)
        if candidate is not None and candidate.exists():
            return candidate

    candidate = project_root / base_dir / f"{ref}.yaml"
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve config: {ref}")


def _resolve_variant_path(variant: dict, key: str, project_root: Path) -> Path | None:
    """variant dict에서 split 경로를 꺼내 실제 경로 객체로 바꾼다."""
    return _resolve_optional_path(variant.get(key), project_root=project_root)


def _resolve_optional_path(value: str | Path | None, *, project_root: Path) -> Path | None:
    """선택값 형태의 경로를 안전하게 해석한다."""
    if not value:
        return None
    return resolve_path(value, project_root)


def _resolve_first_existing_like(*values: str | Path | None, project_root: Path) -> Path:
    """여러 후보 중 먼저 존재하는 경로를 고르고, 없으면 마지막 후보를 반환한다."""
    last_resolved: Path | None = None
    for value in values:
        if not value:
            continue
        resolved = resolve_path(value, project_root)
        if resolved is None:
            continue
        last_resolved = resolved
        if resolved.exists():
            return resolved

    if last_resolved is None:
        raise ValueError("Expected at least one path-like config value")
    return last_resolved


def _resolve_split_name(variant: dict, preferred: str) -> str:
    """원하는 split이 없을 때 사용할 대체 split 이름을 결정한다."""
    aliases = [preferred]
    if preferred == "test":
        aliases.extend(["val", "train"])
    elif preferred == "val":
        aliases.extend(["test", "train"])
    else:
        aliases.extend(["val", "test"])

    for split_name in aliases:
        if variant.get(split_name):
            return split_name

    raise ValueError("Dataset variant does not define train/val/test splits")


def _resolve_runtime_weights(
    *,
    stage: str,
    cli_weights: str | Path | None,
    init_weight: str | Path | None,
    runs_dir: Path,
    model_name: str,
    project_root: Path,
) -> str | None:
    """stage에 맞는 weight 우선순위를 정해 실제 모델 로드 대상을 고른다."""
    if cli_weights is not None:
        return _resolve_weight_input(cli_weights, project_root)

    if stage in {"validate", "predict"}:
        trained_weight = _find_trained_weight(runs_dir)
        if trained_weight is not None:
            return str(trained_weight)
        trained_weight = _find_shared_trained_weight(project_root, model_name)
        if trained_weight is not None:
            return str(trained_weight)

    return resolve_weight_ref(init_weight, project_root)


def _resolve_weight_input(value: str | Path, project_root: Path) -> str:
    """사용자가 직접 넘긴 weight 값은 가능한 한 엄격하게 해석한다."""
    raw = Path(value)
    resolved = resolve_path(raw, project_root)

    if raw.is_absolute() or "/" in str(value) or str(value).startswith("."):
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(f"weight file not found: {value}")
        return str(resolved)

    if resolved is not None and resolved.exists():
        return str(resolved)

    return str(value)


def _find_trained_weight(runs_dir: Path) -> Path | None:
    """학습 산출물에서 재사용할 checkpoint를 우선순위대로 찾는다."""
    for candidate in (runs_dir / "weights" / "best.pt", runs_dir / "weights" / "last.pt"):
        if candidate.exists():
            return candidate.resolve()
    return None


def _find_shared_trained_weight(project_root: Path, model_name: str) -> Path | None:
    """현재 dataset용 checkpoint가 없을 때 동일 모델의 공용 checkpoint를 찾는다."""
    runs_root = project_root / "runs"
    if not runs_root.is_dir():
        return None

    best_candidates = sorted(runs_root.glob(f"*/{model_name}/weights/best.pt"))
    if len(best_candidates) == 1:
        return best_candidates[0].resolve()
    if len(best_candidates) > 1:
        raise ValueError(
            f"Multiple trained checkpoints found for model '{model_name}': "
            f"{[str(path) for path in best_candidates]}. Pass --weights explicitly."
        )

    last_candidates = sorted(runs_root.glob(f"*/{model_name}/weights/last.pt"))
    if len(last_candidates) == 1:
        return last_candidates[0].resolve()
    if len(last_candidates) > 1:
        raise ValueError(
            f"Multiple fallback checkpoints found for model '{model_name}': "
            f"{[str(path) for path in last_candidates]}. Pass --weights explicitly."
        )
    return None


def _resolve_predict_source(
    *,
    variant: dict,
    dataset_override: dict,
    source: str | None,
    project_root: Path,
) -> Path:
    """predict 단계에서 실제 입력 이미지 경로를 정한다."""
    if source and source not in {"train", "val", "test"}:
        resolved = resolve_path(source, project_root)
        if resolved is None:
            raise ValueError(f"Could not resolve predict source: {source}")
        return ensure_images_dir(resolved)

    if source in {"train", "val", "test"}:
        split_name = _resolve_split_name(variant, source)
        split_path = _resolve_variant_path(variant, split_name, project_root)
        if split_path is None:
            raise ValueError(f"Split path is missing for predict source '{source}'")
        return ensure_images_dir(split_path)

    override_source = dataset_override.get("test_image_source")
    if override_source:
        resolved = resolve_path(override_source, project_root)
        if resolved is not None:
            return ensure_images_dir(resolved)

    split_name = _resolve_split_name(variant, "test")
    split_path = _resolve_variant_path(variant, split_name, project_root)
    if split_path is None:
        raise ValueError("Could not resolve a default predict source")
    return ensure_images_dir(split_path)


def _resolve_iou(stage_cfg: dict, dataset_key: str) -> float | None:
    """데이터셋별 예외값이 있으면 그것을, 없으면 기본 iou 값을 선택한다."""
    iou_cfg = stage_cfg.get("iou_default")
    if not isinstance(iou_cfg, dict):
        return stage_cfg.get("iou")

    if dataset_key in iou_cfg:
        return iou_cfg[dataset_key]
    if dataset_key == "little" and "small" in iou_cfg:
        return iou_cfg["small"]
    return iou_cfg.get("default")


def _validate_runtime(runtime: dict) -> None:
    """실행 전에 반드시 존재해야 하는 파일/디렉터리를 확인한다."""
    data_yaml = runtime["dataset"]["data_yaml"]
    if data_yaml is None or not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    stage = runtime["stage"]
    if stage == "predict":
        source = runtime["predict"]["resolved_source"]
        if not source.exists():
            raise FileNotFoundError(f"predict source not found: {source}")
