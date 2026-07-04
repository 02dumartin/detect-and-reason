from __future__ import annotations

from .llava_hf import LlavaHfBackend
from .openai_gpt4o import OpenAIGpt4oBackend
from .qwen_hf import QwenHfBackend
from .qwen_vllm import QwenVllmBackend


def build_vlm_backend(*, backend_cfg: dict, generation_cfg: dict):
    backend_type = str(backend_cfg.get("type", "qwen_hf")).strip().lower()
    if backend_type == "qwen_hf":
        return QwenHfBackend(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
    if backend_type == "qwen_vllm":
        return QwenVllmBackend(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
    if backend_type == "openai_gpt4o":
        return OpenAIGpt4oBackend(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
    if backend_type == "llava_hf":
        return LlavaHfBackend(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
    raise ValueError(f"unsupported vlm backend type: {backend_type}")

