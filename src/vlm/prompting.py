from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .schema import CANONICAL_CLASS_ID_TO_NAME

FULL_OVERLAY_INTRO_TEXT = (
    "Full image with all detected tomato boxes numbered. Use this only for spatial "
    "context (position and size relative to neighboring tomatoes) — do not use it to "
    "judge exact color, since the crop below is closer and clearer for that."
)


def build_target_crop_intro_text(box_number: int | None) -> str:
    """Text shown right before the target crop image.

    Shared by training-data generation and inference so both describe the crop's
    relationship to the full-image overlay identically.
    """
    if box_number is None:
        return (
            "Target crop image to classify (shown below, after any reference images above). "
            "This is the ONLY image your final answer is about — the reference images above "
            "are for color comparison only and must not be classified."
        )
    return (
        f"Target crop image to classify — this is a close-up of Box {box_number} from "
        "the full image above. Use this crop as the primary evidence for color and ripeness. "
        "This is the ONLY image your final answer is about — any reference images shown "
        "earlier are for color comparison only and must not be classified."
    )


def build_messages(
    *,
    record: dict[str, Any],
    crop_image: Image.Image,
    prompt_cfg: dict[str, Any],
    full_overlay_image: Image.Image | None = None,
    bbox_list_text: str | None = None,
    target_box_number: int | None = None,
    color_signal_image: Image.Image | None = None,
    color_signal_intro_text: str | None = None,
    color_signal_text: str | None = None,
    crop_intro_note: str | None = None,
) -> list[dict[str, Any]]:
    """
    Mirrors scripts/vlm_full_ft_with_exemplars.py's build_user_text/build_instruction
    (single user turn, exemplars inline before the target crop) so inference matches
    the exact conversation layout the full-FT checkpoints were trained on.

    full_overlay_image/bbox_list_text are optional extra context (whole image with
    numbered boxes + a text list of all box coordinates) — mirrors the same
    use_full_overlay path in the training script.

    color_signal_image/color_signal_intro_text/color_signal_text are optional
    auxiliary color-evidence blocks (see src/vlm/color_signal.py) shown right after
    the target crop — either a reference swatch strip image, or a text summary of a
    measured color index. They do not change the output JSON schema.

    crop_intro_note is an optional sentence shown right before the crop image itself
    (e.g. noting that a reference swatch strip is attached to the crop via
    color_signal.compose_crop_with_vertical_swatch, so the model knows what it's
    looking at before reading the image).
    """
    class_map = _resolve_class_map(prompt_cfg.get("class_map") or CANONICAL_CLASS_ID_TO_NAME)
    system_message = prompt_cfg.get("system_message") or _default_system_message()
    has_examples = bool(prompt_cfg.get("use_examples")) and bool(prompt_cfg.get("examples"))
    if has_examples:
        system_message = (
            f"{system_message} Reference exemplar images may be shown first for context only — "
            "the one image explicitly marked as the TARGET IMAGE (with a red border and the text "
            "'*** TARGET IMAGE ***' right before it) is the only one you classify."
        )

    user_content: list[dict[str, Any]] = []
    if has_examples:
        examples = prompt_cfg.get("examples") or []
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Reference exemplar images with labels ({len(examples)} total). "
                    "These are ONLY for learning what each ripeness class looks like. "
                    "They are NOT the image you need to classify."
                ),
            }
        )
        for index, example in enumerate(examples, start=1):
            label = example.get("label", "unknown")
            user_content.append(
                {"type": "text", "text": f"[Example {index}/{len(examples)}] label = {label}:"}
            )
            user_content.append({"type": "image", "image": _load_example_image(example)})
        user_content.append(
            {
                "type": "text",
                "text": "--- End of reference examples. None of the images above are to be classified. ---",
            }
        )

    if full_overlay_image is not None:
        user_content.append({"type": "text", "text": FULL_OVERLAY_INTRO_TEXT})
        user_content.append({"type": "image", "image": full_overlay_image})
        if bbox_list_text:
            user_content.append({"type": "text", "text": bbox_list_text})

    user_content.append({"type": "text", "text": build_target_crop_intro_text(target_box_number)})
    if crop_intro_note:
        user_content.append({"type": "text", "text": crop_intro_note})
    if has_examples:
        user_content.append({"type": "text", "text": "*** TARGET IMAGE — classify this one: ***"})
        crop_image = _bordered(crop_image)
    user_content.append({"type": "image", "image": crop_image})

    if color_signal_image is not None:
        if color_signal_intro_text:
            user_content.append({"type": "text", "text": color_signal_intro_text})
        user_content.append({"type": "image", "image": color_signal_image})
    if color_signal_text:
        user_content.append({"type": "text", "text": color_signal_text})

    instruction = prompt_cfg.get("instruction") or _build_instruction(
        record=record,
        prompt_cfg=prompt_cfg,
        class_map=class_map,
    )
    user_content.append({"type": "text", "text": instruction})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": user_content},
    ]


def build_instruction_text(*, record: dict[str, Any], prompt_cfg: dict[str, Any], class_map: dict[int, str]) -> str:
    """Return the canonical instruction text shared by inference and FFT data generation."""
    return _build_instruction(record=record, prompt_cfg=prompt_cfg, class_map=class_map)


def _bordered(image: Image.Image, *, width: int = 10, color: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """Add a solid border around the target crop so it is visually distinct from
    exemplar images even if the model does not fully attend to the text markers."""
    return ImageOps.expand(image, border=width, fill=color)


def _load_example_image(example: dict[str, Any]) -> Image.Image:
    image = example.get("image")
    if image is not None:
        return image.convert("RGB")
    path = example.get("path")
    if not path:
        raise ValueError(f"exemplar example is missing both 'image' and 'path': {example}")
    with Image.open(Path(path)) as handle:
        return ImageOps.exif_transpose(handle).convert("RGB")


def _default_system_message() -> str:
    return (
        "You classify tomato ripeness from a detector crop. "
        "Always return strict JSON only. Do not add markdown or extra text."
    )


def _build_instruction(*, record: dict[str, Any], prompt_cfg: dict[str, Any], class_map: dict[int, str]) -> str:
    """
    Byte-for-byte matches build_instruction() in
    scripts/vlm_full_ft_with_exemplars.py so full-FT checkpoints see the same
    instruction text at inference time that they were trained on.
    """
    use_reasoning = bool(prompt_cfg.get("use_reasoning", True))
    include_bbox_json = bool(prompt_cfg.get("include_bbox_json", True))
    use_color_guide = bool(prompt_cfg.get("use_color_guide", True))
    reasoning_max_words = int(prompt_cfg.get("reasoning_max_words", 35))

    class_names = [str(class_map[class_id]) for class_id in sorted(class_map)]
    enum_names = '", "'.join(class_names)

    ambiguous_tie_break = str(prompt_cfg.get("ambiguous_tie_break", "none"))
    class_definition_block = (
        _build_class_definition_block(class_names, ambiguous_tie_break=ambiguous_tie_break)
        if use_color_guide
        else ""
    )

    text = "Classify only the tomato shown in the crop image.\n"
    text += "Use the crop appearance as primary evidence.\n\n"
    if class_definition_block:
        text += f"Class definitions:\n{class_definition_block}\n"
    if include_bbox_json:
        x1, y1, x2, y2 = [round(float(value), 2) for value in record["bbox_xyxy"]]
        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        text += f"Input bbox JSON (copy these values exactly into output.bbox): {json.dumps(bbox)}\n"
    text += "\n"
    reasoning_style = str(prompt_cfg.get("reasoning_style", "step"))
    swatch_reasoning_hint = bool(prompt_cfg.get("color_signal", {}).get("swatch_reasoning_hint", False))
    if use_reasoning and swatch_reasoning_hint:
        text += (
            "Reasoning instruction:\n"
            "Step1: Compare the tomato's color to the reference swatches attached beside the crop, using "
            "the swatch-to-class guide given with them, and note which swatch it is closest to.\n"
            "Step2: Decide ripeness class from the visual evidence.\n\n"
        )
    elif use_reasoning and reasoning_style == "schema":
        text += _schema_reasoning_instruction()
    elif use_reasoning:
        text += _step_reasoning_instruction()
    text += (
        "Return JSON that follows this schema exactly:\n"
        "{\n"
        f'  "class_name": one of ["{enum_names}"],\n'
        '  "bbox": {"x1": number, "y1": number, "x2": number, "y2": number},\n'
        '  "reasoning": string\n'
        "}\n"
        "Rules:\n"
        "- output valid JSON only\n"
        "- do not add extra keys\n"
        "- bbox must exactly copy the input bbox JSON\n"
    )
    if use_reasoning and reasoning_style == "schema":
        text += (
            "- reasoning must fill in every field of the checklist above, as one line in exactly this "
            'format: "Dominant color: X | Secondary color: X | Green coverage: X | Yellow/orange '
            'coverage: X | Red coverage: X | Color uniformity: X | Transition patches: X"\n'
        )
    elif use_reasoning:
        text += f"- reasoning must contain Step1 and Step2 in one concise sentence, {reasoning_max_words} words or fewer\n"
    else:
        text += '- reasoning: return an empty string ""\n'
    return text


def _step_reasoning_instruction() -> str:
    """Free-form 2-step chain of thought: describe, then decide."""
    return (
        "Reasoning instruction:\n"
        "Step1: Describe visible color pattern and surface cues.\n"
        "Step2: Decide ripeness class from the visual evidence.\n\n"
    )


def _schema_reasoning_instruction() -> str:
    """Structured color checklist the model must fill in before deciding — forces it to
    commit to concrete color-coverage estimates instead of a loose free-text description,
    so the final class follows from stated evidence rather than a vague impression."""
    return (
        "Reasoning instruction:\n"
        "Before deciding, fill in this color checklist for the target tomato:\n"
        "- Dominant color\n"
        "- Secondary color (or \"none\")\n"
        "- Green coverage (0%, 1-20%, 20-50%, or 50%+)\n"
        "- Yellow/orange coverage (0%, 1-20%, 20-50%, or 50%+)\n"
        "- Red coverage (0%, 1-20%, 20-50%, or 50%+)\n"
        "- Color uniformity (high, medium, or low)\n"
        "- Transition patches (present or absent)\n"
        "Then decide the ripeness class from the class definitions above, using the checklist "
        "values as your evidence — do not decide before filling in the checklist.\n\n"
    )


def _build_class_definition_block(class_names: list[str], *, ambiguous_tie_break: str = "none") -> str:
    if class_names == ["ripe", "unripe"]:
        return (
            "- ripe: red or orange dominates overall, even with small shadow, highlight, or dark red regions\n"
            "- unripe: green surface remains dominant\n"
            "- If red or orange is dominant and no clear green area remains, choose ripe.\n"
        )
    if class_names == ["fully-ripe", "semi-ripe", "unripe"]:
        if ambiguous_tie_break == "favor_semi_ripe":
            tie_break_line = (
                "- If any meaningful green, yellow, or orange patch remains anywhere on the surface, even "
                "small, choose semi-ripe over fully-ripe.\n"
            )
        elif ambiguous_tie_break == "favor_fully_ripe":
            tie_break_line = "- If red is dominant and no clear green area remains, choose fully-ripe.\n"
        else:
            # ambiguous_tie_break == "none": 애매한 경우의 강제 기본값을 두지 않는다 —
            # "불분명하면 fully-ripe로" 같은 편향된 tie-break 지시를 프롬프트에서 뺀다.
            tie_break_line = ""
        return (
            "- fully-ripe: red is dominant overall, even with small shadow, highlight, or dark red regions\n"
            "- semi-ripe: clear green/yellow/orange transition remains over a meaningful area\n"
            "- unripe: green surface remains dominant\n"
            f"{tie_break_line}"
        )
    return _build_color_guide(class_names).replace("\nRipeness guide:\n", "").rstrip()


def _build_color_guide(class_names: list[str]) -> str:
    if len(class_names) == 2:
        first, second = class_names
        return (
            "\nRipeness guide:\n"
            f"- {first}: red/orange surface is dominant\n"
            f"- {second}: green surface is dominant\n"
            f"- If the crop is ambiguous, prefer the label whose color is more dominant.\n"
        )

    if len(class_names) == 3:
        first, second, third = class_names
        return (
            "\nRipeness guide:\n"
            f"- {first}: red surface is dominant and continuous, with no residual green shoulder\n"
            f"- {second}: heterogeneous red-orange-green transition remains visible\n"
            f"- {third}: green surface remains dominant\n"
            f"- If illumination, shadow, blur, or occlusion makes {first} vs {second} ambiguous, choose {second}.\n"
        )

    if not class_names:
        return ""
    items = "\n".join(f"- {name}" for name in class_names)
    return f"\nClass guide:\n{items}\n"


def _resolve_class_map(raw_class_names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(raw_class_names, dict):
        return {int(key): str(value).strip() for key, value in sorted(raw_class_names.items(), key=lambda item: int(item[0]))}
    return {idx: str(name).strip() for idx, name in enumerate(raw_class_names)}
