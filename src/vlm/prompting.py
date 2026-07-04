from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .schema import CANONICAL_CLASS_ID_TO_NAME


def build_messages(
    *,
    record: dict[str, Any],
    crop_image: Image.Image,
    prompt_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_message = prompt_cfg.get("system_message") or _default_system_message()
    messages.append(
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        }
    )

    if prompt_cfg.get("use_examples"):
        example_messages = _build_example_messages(prompt_cfg.get("examples") or [])
        messages.extend(example_messages)

    user_text = prompt_cfg.get("instruction") or _build_instruction(record=record, prompt_cfg=prompt_cfg)
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Target crop image of one detected tomato."},
                {"type": "image", "image": crop_image},
                {"type": "text", "text": user_text},
            ],
        }
    )
    return messages


def _build_example_messages(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for example in examples:
        image = example.get("image")
        if image is None:
            path = example.get("path")
            if not path:
                continue
            with Image.open(Path(path)) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
        else:
            image = image.convert("RGB")
        label = example.get("label", "unknown")
        condition = example.get("condition", "unspecified")
        text = f"Reference example. condition={condition}, label={label}."
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image", "image": image},
                ],
            }
        )
    return messages


def _default_system_message() -> str:
    return (
        "You classify tomato ripeness from a detector crop. "
        "Always return strict JSON only. Do not add markdown or extra text."
    )


def _build_instruction(*, record: dict[str, Any], prompt_cfg: dict[str, Any]) -> str:
    use_reasoning = bool(prompt_cfg.get("use_reasoning", True))
    include_bbox_json = bool(prompt_cfg.get("include_bbox_json", True))
    use_color_guide = bool(prompt_cfg.get("use_color_guide", True))
    reasoning_max_words = int(prompt_cfg.get("reasoning_max_words", 25))

    bbox_block = ""
    if include_bbox_json:
        x1, y1, x2, y2 = [round(float(value), 2) for value in record["bbox_xyxy"]]
        bbox_block = (
            "\nInput bbox JSON (copy these values exactly into output.bbox): "
            f'{{"x1": {x1}, "y1": {y1}, "x2": {x2}, "y2": {y2}}}\n'
        )

    color_guide = ""
    if use_color_guide:
        color_guide = (
            "\nRipeness guide:\n"
            "- fully-ripe: red surface is dominant and continuous, with no residual green shoulder\n"
            "- semi-ripe: heterogeneous red-orange-green transition remains visible\n"
            "- unripe: green surface remains dominant\n"
            "- If illumination, shadow, blur, or occlusion makes fully-ripe vs semi-ripe ambiguous, choose semi-ripe.\n"
        )

    reasoning_rule = (
        f'- reasoning: one short sentence, {reasoning_max_words} words or fewer\n'
        if use_reasoning
        else '- reasoning: return an empty string ""\n'
    )

    enum_names = '", "'.join(CANONICAL_CLASS_ID_TO_NAME.values())
    return (
        "Classify only the tomato shown in the crop image.\n"
        "Use the crop appearance as primary evidence.\n"
        f"{bbox_block}"
        f"{color_guide}"
        "Return JSON that follows this schema exactly:\n"
        "{\n"
        f'  "class_name": one of ["{enum_names}"],\n'
        '  "bbox": {"x1": number, "y1": number, "x2": number, "y2": number},\n'
        '  "reasoning": string\n'
        "}\n"
        "Rules:\n"
        "- output valid JSON only\n"
        "- do not add extra keys\n"
        "- bbox must exactly copy the input bbox JSON when it is provided\n"
        f"{reasoning_rule}"
    )
