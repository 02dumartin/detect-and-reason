from __future__ import annotations

import inspect
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover
    process_vision_info = None
# 이게 뭐 하는 거지 

from .base import VlmBackend


class QwenHfBackend(VlmBackend):
    supports_batch = True

    def __init__(self, *, backend_cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> None:
        super().__init__(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
        self.model = None
        self.processor = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        model_name = self.backend_cfg["model_name"] # Qwen/Qwen3-VL-4B-Instruct
        kwargs: dict[str, Any] = {
            "device_map": self.backend_cfg.get("device_map", "auto"),
            "trust_remote_code": bool(self.backend_cfg.get("trust_remote_code", True)),
        }
        torch_dtype = self.backend_cfg.get("torch_dtype", "auto")
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, str(torch_dtype))
        if self.backend_cfg.get("use_4bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        if self.backend_cfg.get("compile_model"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=kwargs["trust_remote_code"])
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    @torch.inference_mode()
    def generate(self, messages: list[dict[str, Any]]) -> str:
        self._load()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor_inputs(messages=messages, text=text)
        outputs = self.model.generate(**inputs, **_generation_kwargs(self.generation_cfg))
        return _decode_response(self.processor, outputs[0])

    @torch.inference_mode()
    def generate_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        self._load()
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        inputs = self._processor_inputs(messages_batch=messages_batch, text=texts)
        outputs = self.model.generate(**inputs, **_generation_kwargs(self.generation_cfg))
        return [_decode_response(self.processor, outputs[idx]) for idx in range(outputs.shape[0])]

    def _processor_inputs(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        messages_batch: list[list[dict[str, Any]]] | None = None,
        text: str | list[str],
    ):
        proc_sig = inspect.signature(self.processor.__call__)
        kwargs: dict[str, Any] = {
            "text": text,
            "return_tensors": "pt",
        }
        if isinstance(text, list):
            kwargs["padding"] = True

        if process_vision_info is not None:
            if messages_batch is not None:
                flat_images = []
                flat_videos = []
                for message_set in messages_batch:
                    image_inputs, video_inputs = process_vision_info(message_set)
                    if image_inputs:
                        flat_images.extend(image_inputs)
                    if video_inputs:
                        flat_videos.extend(video_inputs)
                if flat_images and "images" in proc_sig.parameters:
                    kwargs["images"] = flat_images
                if flat_videos and "video" in proc_sig.parameters:
                    kwargs["video"] = flat_videos
            elif messages is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                if image_inputs is not None and "images" in proc_sig.parameters:
                    kwargs["images"] = image_inputs
                if video_inputs is not None and "video" in proc_sig.parameters:
                    kwargs["video"] = video_inputs
        inputs = self.processor(**kwargs)
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)
        return inputs


def _generation_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(cfg.get("max_new_tokens", 220)),
        "repetition_penalty": float(cfg.get("repetition_penalty", 1.1)),
        "do_sample": bool(cfg.get("do_sample", False)),
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(cfg.get("temperature", 0.2))
        kwargs["top_p"] = float(cfg.get("top_p", 0.9))
        kwargs["top_k"] = int(cfg.get("top_k", 40))
    return kwargs


def _decode_response(processor: Any, tokens) -> str:
    decoded = processor.decode(tokens, skip_special_tokens=True)
    return decoded.split("assistant")[-1].strip() if "assistant" in decoded else decoded.strip()
