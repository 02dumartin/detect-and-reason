"""crop 색상을 VLM 프롬프트에 보조 신호로 넣기 위한 3가지 ablation 모드.

- whole_crop_index: crop 전체 평균을 CIELAB a*/b*/hue로 요약해 텍스트로 삽입
- multi_point_sample: crop 내부 여러 지점을 스포이딩하듯 샘플링해 median 색 +
  지점 간 색 편차(균일도)를 텍스트로 삽입
- swatch_match: 실제 익음 궤적(hue 0~360)을 따라 만든 참조 색상 스와치
  이미지를 함께 보여줘 상대 비교를 유도 (CPT의 fill-in-the-blank 아이디어 참고)

세 모드 모두 최종 판정 스키마는 건드리지 않고, build_messages에 들어가는
추가 evidence 블록(텍스트/이미지)만 제공한다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.vlm.image_adjustment import apply_image_adjustment
from src.visualization.yolo_overlay_renderer import DEFAULT_FONT_PATH, _load_font

try:
    from skimage import color as skcolor
except Exception:  # pragma: no cover
    skcolor = None


def compute_whole_crop_color_index(
    image: Image.Image,
    *,
    adjustment_cfg: dict[str, Any] | str | None = None,
) -> dict[str, float]:
    """crop 전체 평균 색을 CIELAB a*/b*/hue로 요약한다."""
    adjusted = apply_image_adjustment(image.convert("RGB"), adjustment_cfg)
    l_val, a_val, b_val = _lab_of_patch(np.asarray(adjusted))
    hue_deg, chroma = _hue_and_chroma(a_val, b_val)
    return {
        "l": round(l_val, 2),
        "a": round(a_val, 2),
        "b": round(b_val, 2),
        "hue_deg": round(hue_deg, 1),
        "chroma": round(chroma, 2),
    }


def sample_multi_point_colors(
    image: Image.Image,
    *,
    num_points: int = 7,
    exclude_border_ratio: float = 0.15,
    patch_radius: int = 2,
    uniformity_thresholds: dict[str, float] | None = None,
    adjustment_cfg: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """crop 내부 여러 지점을 스포이딩하듯 샘플링해 median 색과 색 균일도를 계산한다.

    각 지점은 단일 픽셀이 아니라 patch_radius 반경의 작은 이웃 평균으로 뜬다
    (실제 스포이드 도구의 aperture와 비슷하게, 반사광 등 단일 픽셀 노이즈에
    덜 흔들리도록). 지점 간 hue의 원형 표준편차를 색 균일도(uniformity) 지표로
    같이 계산해, "이 토마토가 색이 고르게 익었는지 얼룩덜룩한지"까지 텍스트로
    전달한다.
    """
    thresholds = uniformity_thresholds or {"low_max_hue_std": 5.0, "high_min_hue_std": 15.0}
    adjusted = apply_image_adjustment(image.convert("RGB"), adjustment_cfg)
    arr = np.asarray(adjusted)
    height, width = arr.shape[:2]

    points = _interior_grid_points(
        width=width,
        height=height,
        num_points=num_points,
        exclude_border_ratio=exclude_border_ratio,
    )

    samples: list[dict[str, Any]] = []
    for px, py in points:
        xa, xb = max(0, px - patch_radius), min(width, px + patch_radius + 1)
        ya, yb = max(0, py - patch_radius), min(height, py + patch_radius + 1)
        l_val, a_val, b_val = _lab_of_patch(arr[ya:yb, xa:xb])
        hue_deg, chroma = _hue_and_chroma(a_val, b_val)
        samples.append(
            {
                "x": px,
                "y": py,
                "l": round(l_val, 2),
                "a": round(a_val, 2),
                "b": round(b_val, 2),
                "hue_deg": round(hue_deg, 1),
                "chroma": round(chroma, 2),
            }
        )

    median_l = float(np.median([s["l"] for s in samples]))
    median_a = float(np.median([s["a"] for s in samples]))
    median_b = float(np.median([s["b"] for s in samples]))
    median_hue_deg, median_chroma = _hue_and_chroma(median_a, median_b)
    hue_std_deg = _circular_std_deg(np.array([s["hue_deg"] for s in samples], dtype=np.float32))

    if hue_std_deg <= thresholds["low_max_hue_std"]:
        uniformity_label = "uniform (low color variation across the surface)"
    elif hue_std_deg >= thresholds["high_min_hue_std"]:
        uniformity_label = "blotchy (high color variation across the surface)"
    else:
        uniformity_label = "moderate color variation across the surface"

    return {
        "points": samples,
        "median_l": round(median_l, 2),
        "median_a": round(median_a, 2),
        "median_b": round(median_b, 2),
        "median_hue_deg": round(median_hue_deg, 1),
        "median_chroma": round(median_chroma, 2),
        "hue_std_deg": round(hue_std_deg, 1),
        "uniformity_label": uniformity_label,
    }


def build_color_signal_text(mode: str, signal: dict[str, Any]) -> str:
    if mode == "whole_crop_index":
        return (
            "Measured crop color (CIELAB, whole-crop average): "
            f"hue={signal['hue_deg']}°, chroma={signal['chroma']}, a*={signal['a']}, b*={signal['b']}. "
            "Lower hue (roughly 0-60°) indicates more red, higher hue (roughly 100-140°) indicates more "
            "green. Use this as a supporting numeric cue alongside the visual appearance."
        )
    if mode == "multi_point_sample":
        return (
            f"Measured crop color from {len(signal['points'])} sampled surface points (median CIELAB): "
            f"hue={signal['median_hue_deg']}°, chroma={signal['median_chroma']}, "
            f"a*={signal['median_a']}, b*={signal['median_b']}. "
            f"Surface color variation across sampled points: {signal['uniformity_label']} "
            f"(hue std={signal['hue_std_deg']}°). "
            "Lower hue indicates more red, higher hue indicates more green. Use this as a supporting "
            "numeric cue alongside the visual appearance."
        )
    raise ValueError(f"unsupported color_signal mode for text: {mode!r}")


def build_swatch_palette(
    *,
    count: int = 6,
    hue_range_deg: tuple[float, float] = (20.0, 130.0),
    lightness: float = 55.0,
    chroma: float = 55.0,
) -> list[dict[str, Any]]:
    """토마토 익음 hue 궤적(빨강~초록)을 따라 균등 간격 참조 swatch를 만든다.

    hue_range_deg는 낮을수록 빨강, 높을수록 초록 쪽 (CIELAB hue angle 기준).
    """
    if skcolor is None:
        raise RuntimeError("skimage is required to build color swatches (pip install scikit-image)")
    low, high = hue_range_deg
    hues = np.linspace(float(low), float(high), int(count))
    swatches: list[dict[str, Any]] = []
    for index, hue_deg in enumerate(hues):
        label = chr(ord("A") + index)
        a_val = chroma * math.cos(math.radians(hue_deg))
        b_val = chroma * math.sin(math.radians(hue_deg))
        lab = np.array([[[lightness, a_val, b_val]]], dtype=np.float32)
        rgb = np.clip(skcolor.lab2rgb(lab), 0.0, 1.0)[0, 0]
        swatches.append(
            {
                "label": label,
                "hue_deg": round(float(hue_deg), 1),
                "rgb": tuple(int(round(channel * 255)) for channel in rgb),
            }
        )
    return swatches


def render_swatch_strip_image(
    swatches: list[dict[str, Any]],
    *,
    swatch_size: tuple[int, int] = (90, 90),
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = 28,
) -> Image.Image:
    swatch_w, swatch_h = swatch_size
    image = Image.new("RGB", (swatch_w * len(swatches), swatch_h + 40), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = _load_font(font_path, font_size)
    for index, swatch in enumerate(swatches):
        x0 = index * swatch_w
        x1 = x0 + swatch_w
        draw.rectangle([x0, 0, x1, swatch_h], fill=swatch["rgb"], outline=(0, 0, 0))
        draw.text((x0 + swatch_w // 2 - font_size // 3, swatch_h + 5), swatch["label"], fill=(0, 0, 0), font=font)
    return image


def render_vertical_swatch_strip_image(
    swatches: list[dict[str, Any]],
    *,
    height: int,
    swatch_width: int = 60,
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = 20,
) -> Image.Image:
    """swatch를 세로로 쌓아, crop과 정확히 같은 높이로 렌더링한다 (crop 옆에 붙이기 위함).

    strip 전체 높이는 항상 요청한 height와 같다 — swatch가 crop보다 커지면
    작은 crop에서 swatch가 화면을 압도해 정확도가 더 떨어지는 게 확인됐기
    때문에, 칸 최소 높이를 강제하지 않는다. 대신 칸이 글자보다 작아지면
    글자 크기를 칸에 맞춰 줄여서, 라벨이 옆 칸을 침범하는 것만 막는다
    (아주 작은 crop에서는 글자가 안 보일 정도로 작아질 수 있음 — 의도된 트레이드오프).
    """
    count = max(1, len(swatches))
    row_height = max(1, height // count)
    strip_height = row_height * count
    effective_font_size = max(1, min(font_size, row_height - 2))
    image = Image.new("RGB", (swatch_width, strip_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = _load_font(font_path, effective_font_size)
    for index, swatch in enumerate(swatches):
        y0 = index * row_height
        y1 = y0 + row_height
        draw.rectangle([0, y0, swatch_width, y1], fill=swatch["rgb"], outline=(0, 0, 0))
        label = swatch["label"]
        text_color = (255, 255, 255) if _is_dark_rgb(swatch["rgb"]) else (0, 0, 0)
        draw.text(
            (swatch_width // 2 - effective_font_size // 4, y0 + row_height // 2 - effective_font_size // 2),
            label,
            fill=text_color,
            font=font,
        )
    return image


def compose_crop_with_vertical_swatch(
    crop_image: Image.Image,
    swatches: list[dict[str, Any]],
    *,
    swatch_width: int = 60,
    gap: int = 6,
) -> Image.Image:
    """crop 오른쪽에 세로 swatch strip을 붙여 하나의 이미지로 합친다.

    이 함수의 반환값이 그대로 VLM에 보여줄 crop_image로 쓰인다 — swatch가
    별도 이미지 블록이 아니라 crop 자체에 나란히 붙어서 들어간다. swatch strip은
    항상 crop과 같은 높이로 렌더링되므로 (render_vertical_swatch_strip_image 참고)
    swatch가 crop보다 커져서 화면을 압도하는 일은 없다.
    """
    swatch_strip = render_vertical_swatch_strip_image(swatches, height=crop_image.height, swatch_width=swatch_width)
    canvas_height = max(crop_image.height, swatch_strip.height)
    canvas = Image.new(
        "RGB",
        (crop_image.width + gap + swatch_strip.width, canvas_height),
        (255, 255, 255),
    )
    canvas.paste(crop_image.convert("RGB"), (0, (canvas_height - crop_image.height) // 2))
    canvas.paste(swatch_strip, (crop_image.width + gap, (canvas_height - swatch_strip.height) // 2))
    return canvas


def _is_dark_rgb(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 140.0


def build_swatch_intro_text(swatches: list[dict[str, Any]]) -> str:
    labels = ", ".join(swatch["label"] for swatch in swatches)
    return (
        f"Reference color swatches ({labels}, left to right) spanning typical tomato ripening colors. "
        "Compare the target crop's color to these swatches only as a coarse anchor for where it falls on "
        "the color spectrum — make your final judgment mainly from the crop image itself."
    )


def build_attached_swatch_note(swatches: list[dict[str, Any]]) -> str:
    """crop 오른쪽에 세로로 붙인 swatch strip용 안내 문구 (compose_crop_with_vertical_swatch와 짝)."""
    labels = ", ".join(swatch["label"] for swatch in swatches)
    return (
        f"Reference color swatches ({labels}, top to bottom) are attached to the right side of this "
        "image, spanning typical tomato ripening colors. Compare the tomato's color to these swatches "
        "only as a coarse anchor for where it falls on the color spectrum — make your final judgment "
        "mainly from the tomato crop itself."
    )


# 5-swatch(A~E) / 3-class(fully-ripe, semi-ripe, unripe) 조합 전용 고정 매핑.
# swatch 개수나 class 구성이 바뀌면 이 상수도 같이 바꿔야 한다 (일반화된 자동 계산이 아님 —
# 이 특정 ablation 조합에 한정해서 의도적으로 하드코딩).
SWATCH5_TO_3CLS_MAPPING: dict[str, str] = {
    "A": "fully-ripe",
    "B": "fully-ripe",
    "C": "semi-ripe",
    "D": "unripe",
    "E": "unripe",
}


def build_attached_swatch_note_with_mapping(
    swatches: list[dict[str, Any]],
    class_mapping: dict[str, str] = SWATCH5_TO_3CLS_MAPPING,
) -> str:
    """swatch 문자 -> ripeness class 매핑을 명시한 안내 문구.

    build_attached_swatch_note와 달리 "이 swatch가 어떤 클래스에 해당하는지"를
    직접 알려준다 — 매핑을 숨기면 모델이 스스로 애매한 매핑을 만들어내고, 특히
    중간 swatch들을 전부 "semi-ripe"로 뭉뚱그리는 경향이 관찰됐기 때문
    (260909_vlm-classify-swatch-attach-ablation-tomatod 실험 참고).
    """
    mapping_text = ", ".join(f"{swatch['label']}→{class_mapping[swatch['label']]}" for swatch in swatches)
    return (
        f"Reference color swatches are attached to the right side of this image, top to bottom, spanning "
        f"typical tomato ripening colors from red to green. Swatch-to-class guide: {mapping_text}. "
        "Use this only as a coarse anchor for where the tomato's color falls — make your final judgment "
        "mainly from the tomato crop itself."
    )


def nearest_swatch_label(hue_deg: float, swatches: list[dict[str, Any]]) -> str | None:
    """분석용 — 모델에게 강제로 출력시키지는 않지만, whole-crop hue와 가장 가까운
    swatch를 로그에 남겨 나중에 예측과의 상관관계를 볼 수 있게 한다."""
    if not swatches:
        return None
    nearest = min(swatches, key=lambda swatch: _hue_distance(hue_deg, swatch["hue_deg"]))
    return str(nearest["label"])


def _hue_distance(hue_a: float, hue_b: float) -> float:
    diff = abs(hue_a - hue_b) % 360.0
    return min(diff, 360.0 - diff)


def _interior_grid_points(
    *,
    width: int,
    height: int,
    num_points: int,
    exclude_border_ratio: float,
) -> list[tuple[int, int]]:
    x0 = int(width * exclude_border_ratio)
    x1 = max(x0 + 1, width - x0)
    y0 = int(height * exclude_border_ratio)
    y1 = max(y0 + 1, height - y0)

    grid_side = max(2, math.ceil(math.sqrt(num_points)))
    xs = np.linspace(x0, x1 - 1, grid_side)
    ys = np.linspace(y0, y1 - 1, grid_side)
    candidates = [(int(round(x)), int(round(y))) for y in ys for x in xs]
    candidates.sort(key=lambda point: (point[0] - width / 2) ** 2 + (point[1] - height / 2) ** 2)
    return candidates[:num_points]


def _lab_of_patch(rgb_patch: np.ndarray) -> tuple[float, float, float]:
    arr = rgb_patch.astype(np.float32) / 255.0
    if skcolor is not None:
        lab = skcolor.rgb2lab(arr).reshape(-1, 3)
        l_val, a_val, b_val = (float(v) for v in lab.mean(axis=0))
        return l_val, a_val, b_val
    # skimage 없을 때는 a*/b*를 추정할 수 없으니 명도만 근사치로 반환한다.
    return float(arr.mean() * 100.0), 0.0, 0.0


def _hue_and_chroma(a_val: float, b_val: float) -> tuple[float, float]:
    hue_deg = math.degrees(math.atan2(b_val, a_val)) % 360.0
    chroma = math.hypot(a_val, b_val)
    return hue_deg, chroma


def _circular_std_deg(hues_deg: np.ndarray) -> float:
    radians = np.radians(hues_deg)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    resultant = min(max(math.hypot(sin_mean, cos_mean), 1e-9), 1.0)
    return math.degrees(math.sqrt(-2.0 * math.log(resultant))) + 0.0
