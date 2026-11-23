from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

try:
    from skimage import color, exposure
except Exception:  # pragma: no cover
    color = None
    exposure = None


def apply_image_adjustment(image: Image.Image, adjustment_cfg: dict[str, Any] | str | None) -> Image.Image:
    cfg = _normalize_adjustment_cfg(adjustment_cfg)
    if not cfg:
        return image

    adjustment_type = str(cfg.get("type") or cfg.get("name") or "").strip().lower()
    if adjustment_type == "clahe":
        return apply_clahe(
            image,
            clip_limit=float(cfg.get("clip_limit", 0.03)),
            kernel_size=cfg.get("kernel_size"),
        )
    if adjustment_type in {"brightness_contrast", "brightness-contrast", "bc"}:
        return apply_brightness_contrast(
            image,
            brightness=float(cfg.get("brightness", 1.0)),
            contrast=float(cfg.get("contrast", 1.0)),
        )
    if adjustment_type in {"none", "off", "false"}:
        return image
    raise ValueError(f"unsupported image adjustment type: {adjustment_type!r}")


def apply_clahe(
    image: Image.Image,
    *,
    clip_limit: float = 0.03,
    kernel_size: int | tuple[int, int] | None = None,
) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if color is None or exposure is None:
        # Fallback is intentionally conservative: if skimage is unavailable,
        # still improve contrast a bit so the pipeline does not fail.
        return ImageOps.autocontrast(rgb)

    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(arr)
    l_channel = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    equalized = exposure.equalize_adapthist(l_channel, clip_limit=clip_limit, kernel_size=kernel_size)
    lab[..., 0] = np.clip(equalized * 100.0, 0.0, 100.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Conversion from CIE-LAB, via XYZ to sRGB color space resulted in",
            category=UserWarning,
        )
        adjusted = np.clip(color.lab2rgb(lab), 0.0, 1.0)
    return Image.fromarray((adjusted * 255.0).astype(np.uint8), mode="RGB")


def apply_brightness_contrast(
    image: Image.Image,
    *,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    adjusted = ImageEnhance.Brightness(rgb).enhance(max(0.0, brightness))
    adjusted = ImageEnhance.Contrast(adjusted).enhance(max(0.0, contrast))
    return adjusted


def _normalize_adjustment_cfg(adjustment_cfg: dict[str, Any] | str | None) -> dict[str, Any]:
    if adjustment_cfg is None:
        return {}
    if isinstance(adjustment_cfg, str):
        return {"type": adjustment_cfg}
    return dict(adjustment_cfg)
