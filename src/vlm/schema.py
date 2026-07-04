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


def normalize_prediction_label(*, class_name: Any = None, category_id: Any = None) -> tuple[int, str]:
    class_id = normalize_class_id(category_id)
    if class_id >= 0:
        return class_id, CANONICAL_CLASS_ID_TO_NAME[class_id]

    normalized_name = normalize_class_name(class_name)
    if normalized_name in CANONICAL_CLASS_NAME_TO_ID:
        resolved_id = CANONICAL_CLASS_NAME_TO_ID[normalized_name]
        return resolved_id, normalized_name
    return -1, "none"

