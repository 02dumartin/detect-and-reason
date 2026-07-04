from __future__ import annotations

from typing import Any

from .base import VlmBackend


class QwenVllmBackend(VlmBackend):
    supports_batch = True

    def __init__(self, *, backend_cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> None:
        super().__init__(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
        try:
            from vllm import LLM, SamplingParams
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("vllm package is required for the qwen_vllm backend") from exc

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=backend_cfg["model_name"],
            trust_remote_code=bool(backend_cfg.get("trust_remote_code", True)),
            limit_mm_per_prompt={"image": int(backend_cfg.get("max_images_per_prompt", 16))},
        )

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self.generate_batch([messages])[0]

    def generate_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        requests = [_to_vllm_request(messages) for messages in messages_batch]
        outputs = self.llm.generate(requests, sampling_params=_sampling_params(self.SamplingParams, self.generation_cfg))
        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]


def _sampling_params(cls, generation_cfg: dict[str, Any]):
    return cls(
        max_tokens=int(generation_cfg.get("max_new_tokens", 220)),
        temperature=float(generation_cfg.get("temperature", 0.2))
        if generation_cfg.get("do_sample")
        else 0.0,
        top_p=float(generation_cfg.get("top_p", 0.9)),
        repetition_penalty=float(generation_cfg.get("repetition_penalty", 1.1)),
    )


def _to_vllm_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    text_parts: list[str] = []
    images = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        for item in message.get("content", []):
            if item.get("type") == "text":
                text_parts.append(f"{role}: {item['text']}")
            elif item.get("type") == "image":
                images.append(item["image"])
    request = {"prompt": "\n\n".join(text_parts)}
    if images:
        request["multi_modal_data"] = {"image": images}
    return request

