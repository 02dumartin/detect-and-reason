from __future__ import annotations

import json
import re
from typing import Any

from .schema import normalize_class_name, normalize_prediction_label


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_first_json(text: str) -> str:
    payload = (text or "").strip()
    if not payload:
        raise ValueError("empty response")

    for candidate in _candidate_json_strings(payload):
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue
    raise ValueError(f"no valid json found in response: {payload[:240]}")


def parse_vlm_response(
    *,
    text: str,
    bbox_fallback_xyxy: list[float],
    require_reasoning: bool,
    class_name_map: dict[int, str] | list[str] | None = None,
) -> dict[str, Any]:
    raw = (text or "").strip()
    parsed_payload: dict[str, Any] | None = None
    parse_error: str | None = None

    try:
        parsed_payload = json.loads(extract_first_json(raw))
    except Exception as exc:
        parse_error = str(exc)
        parsed_payload = _recover_payload_from_text(raw, class_name_map=class_name_map)

    class_id, class_name = normalize_prediction_label(
        class_name=(parsed_payload or {}).get("class_name") or (parsed_payload or {}).get("tomato"),
        class_name_map=class_name_map,
    )

    bbox_payload = (parsed_payload or {}).get("bbox")
    bbox_xyxy = _normalize_bbox_payload(bbox_payload, fallback=bbox_fallback_xyxy)
    reasoning = str((parsed_payload or {}).get("reasoning") or "").strip()
    if not require_reasoning and not reasoning:
        reasoning = ""

    return {
        "ok": class_id >= 0,
        "class_id": class_id,
        "class_name": class_name,
        "bbox_xyxy": bbox_xyxy,
        "reasoning": reasoning,
        "raw_response": raw,
        "parse_error": parse_error,
        "parsed_payload": parsed_payload,
    }


def _candidate_json_strings(text: str) -> list[str]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    candidates = [cleaned]
    candidates.extend(match.group(0) for match in _JSON_RE.finditer(cleaned))
    return candidates


def _recover_payload_from_text(text: str, *, class_name_map: dict[int, str] | list[str] | None = None) -> dict[str, Any]:
    lowered = (text or "").lower()
    label = None
    if class_name_map is not None:
        resolved_map = _coerce_class_name_map(class_name_map)
        for class_id, class_label in sorted(resolved_map.items(), key=lambda item: len(str(item[1])), reverse=True):
            normalized_label = normalize_class_name(class_label)
            candidates = {
                str(class_label).strip().lower(),
                normalized_label,
            }
            if any(candidate and _contains_token(lowered, candidate) for candidate in candidates):
                label = class_label
                break

    if label is None:
        if "fully-ripe" in lowered or "fully ripe" in lowered or "red tomato" in lowered:
            label = "fully-ripe"
        elif "semi-ripe" in lowered or "semi ripe" in lowered or "half ripe" in lowered:
            label = "semi-ripe"
        elif "unripe" in lowered or "green tomato" in lowered or re.search(r"\bgreen\b", lowered):
            label = "unripe"

    reasoning = _short_reasoning(text)
    return {
        "class_name": label or "none",
        "reasoning": reasoning,
    }


def _coerce_class_name_map(class_name_map: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(class_name_map, dict):
        return {int(key): str(value).strip() for key, value in sorted(class_name_map.items(), key=lambda item: int(item[0]))}
    return {idx: str(name).strip() for idx, name in enumerate(class_name_map)}


def _contains_token(text: str, candidate: str) -> bool:
    pattern = rf"(?<!\w){re.escape(candidate)}(?!\w)"
    return re.search(pattern, text) is not None


def _short_reasoning(text: str) -> str:
    compact = " ".join((text or "").split())
    words = compact.split()
    return " ".join(words[:25]).strip()


def _normalize_bbox_payload(payload: Any, *, fallback: list[float]) -> list[float]:
    if isinstance(payload, dict):
        try:
            values = [float(payload[key]) for key in ("x1", "y1", "x2", "y2")]
            return values
        except Exception:
            return list(fallback)
    if isinstance(payload, list) and len(payload) >= 4:
        try:
            return [float(payload[idx]) for idx in range(4)]
        except Exception:
            return list(fallback)
    return list(fallback)
