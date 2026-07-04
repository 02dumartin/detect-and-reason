from __future__ import annotations

import base64
import io
import os
from typing import Any

from PIL import Image

from .base import VlmBackend


class OpenAIGpt4oBackend(VlmBackend):
    def __init__(self, *, backend_cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> None:
        super().__init__(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("openai package is required for the GPT-4o backend") from exc

        api_key_env = backend_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"{api_key_env} is not set")

        base_url = backend_cfg.get("base_url")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(self, messages: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.backend_cfg.get("model_name", "gpt-4o"),
            messages=_to_openai_messages(messages),
            max_tokens=int(self.generation_cfg.get("max_new_tokens", 220)),
            temperature=float(self.generation_cfg.get("temperature", 0.2))
            if self.generation_cfg.get("do_sample")
            else 0.0,
            top_p=float(self.generation_cfg.get("top_p", 0.9)),
        )
        return str(response.choices[0].message.content or "").strip()


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for message in messages:
        content_items = []
        for item in message.get("content", []):
            if item.get("type") == "text":
                content_items.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image":
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _pil_to_data_url(item["image"])},
                    }
                )
        formatted.append({"role": message["role"], "content": content_items})
    return formatted


def _pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

