from __future__ import annotations

from typing import Any


CANONICAL_CLASS_ID_TO_NAME = {
    0: "fully-ripe",
    1: "semi-ripe",
    2: "unripe",
}

CANONICAL_CLASS_NAME_TO_ID = {name: class_id for class_id, name in CANONICAL_CLASS_ID_TO_NAME.items()}

CLASS_NAME_ALIASES = {
    "fully-ripe": "fully-ripe",
    "fully ripe": "fully-ripe",
    "fully ripened tomato": "fully-ripe",
    "fully ripened": "fully-ripe",
    "red tomato": "fully-ripe",
    "ripe": "fully-ripe",
    "semi-ripe": "semi-ripe",
    "semi ripe": "semi-ripe",
    "half-ripe": "semi-ripe",
    "half ripe": "semi-ripe",
    "half ripened tomato": "semi-ripe",
    "half ripened": "semi-ripe",
    "partially ripe": "semi-ripe",
    "unripe": "unripe",
    "green tomato": "unripe",
    "green": "unripe",
    "none": "none",
    "unknown": "none",
}


def normalize_class_name(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("_", " ")
    raw = " ".join(raw.split())
    if raw in CLASS_NAME_ALIASES:
        return CLASS_NAME_ALIASES[raw]

    if "unripe" in raw or "green" in raw:
        return "unripe"
    if "semi" in raw or "half" in raw:
        return "semi-ripe"
    if "fully" in raw or "ripe" in raw or "red" in raw:
        return "fully-ripe"
    return "none"


def normalize_class_id(value: Any) -> int:
    try:
        class_id = int(value)
    except (TypeError, ValueError):
        return -1
    return class_id if class_id in CANONICAL_CLASS_ID_TO_NAME else -1


def normalize_prediction_label(
    *,
    class_name: Any = None,
    category_id: Any = None,
    class_name_map: dict[int, str] | list[str] | None = None,
) -> tuple[int, str]:
    """예측 라벨을 `(class_id, class_name)`로 정규화한다.

    `class_name_map`이 주어지면 그 맵을 우선적으로 사용해서 2cls/3cls
    같은 task-specific label space를 그대로 유지한다. 맵이 없을 때만
    기존 canonical 3cls 공간으로 떨어진다.
    """
    if class_name_map is not None:
        resolved_map = _coerce_class_name_map(class_name_map)
        normalized_lookup = {
            normalize_class_name(raw_name): (class_id, raw_name)
            for class_id, raw_name in resolved_map.items()
        }
        raw_lookup = {str(raw_name).strip().lower(): (class_id, raw_name) for class_id, raw_name in resolved_map.items()}

        if class_name is not None:
            candidate = str(class_name).strip().lower()
            normalized_candidate = normalize_class_name(candidate)
            if candidate in raw_lookup:
                return raw_lookup[candidate]
            if normalized_candidate in normalized_lookup:
                return normalized_lookup[normalized_candidate]

        class_id = normalize_class_id(category_id)
        if class_id in resolved_map:
            return class_id, resolved_map[class_id]

        return -1, "none"

    class_id = normalize_class_id(category_id)
    if class_id >= 0:
        return class_id, CANONICAL_CLASS_ID_TO_NAME[class_id]

    normalized_name = normalize_class_name(class_name)
    if normalized_name in CANONICAL_CLASS_NAME_TO_ID:
        resolved_id = CANONICAL_CLASS_NAME_TO_ID[normalized_name]
        return resolved_id, normalized_name
    return -1, "none"


def _coerce_class_name_map(class_name_map: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(class_name_map, dict):
        return {int(key): str(value).strip() for key, value in sorted(class_name_map.items(), key=lambda item: int(item[0]))}
    return {idx: str(name).strip() for idx, name in enumerate(class_name_map)}
