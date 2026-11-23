from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover
    process_vision_info = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

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

        model_name = self.backend_cfg["model_name"]
        kwargs: dict[str, Any] = {
            "device_map": self.backend_cfg.get("device_map", "auto"),
            "trust_remote_code": bool(self.backend_cfg.get("trust_remote_code", True)),
        }
        torch_dtype = self.backend_cfg.get("torch_dtype", "auto")
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, str(torch_dtype))
        if self.backend_cfg.get("use_4bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        load_mode = str(self.backend_cfg.get("load_mode", "auto")).strip().lower()
        model_path = _existing_local_path(self.backend_cfg.get("model_path"))
        adapter_path = _existing_local_path(self.backend_cfg.get("adapter_path"))
        processor_ref = self.backend_cfg.get("processor_path")

        if load_mode not in {"auto", "full", "adapter"}:
            raise ValueError(f"unsupported qwen_hf load_mode: {load_mode}")

        if model_path is not None and model_path.is_dir() and not _looks_like_hf_model_dir(model_path):
            resolved_model_path = _resolve_training_output_model_dir(model_path)
            if resolved_model_path is not None:
                model_path = resolved_model_path

        adapter_like_model_path = _looks_like_adapter_dir(model_path)
        if load_mode == "adapter" or adapter_path is not None or adapter_like_model_path:
            if PeftModel is None:
                raise ImportError("peft is required when loading a Qwen LoRA adapter")
            base_model_ref = self.backend_cfg.get("base_model_name") or model_name
            adapter_ref = adapter_path or model_path
            if adapter_ref is None:
                raise ValueError("adapter_path or adapter-like model_path is required for adapter mode")
            self.model = AutoModelForImageTextToText.from_pretrained(base_model_ref, **kwargs)
            self.model = PeftModel.from_pretrained(self.model, str(adapter_ref))
            if bool(self.backend_cfg.get("merge_adapter", False)):
                merge_and_unload = getattr(self.model, "merge_and_unload", None)
                if callable(merge_and_unload):
                    self.model = merge_and_unload()
            if processor_ref is None:
                processor_ref = base_model_ref
        else:
            load_ref = model_path or model_name
            self.model = AutoModelForImageTextToText.from_pretrained(load_ref, **kwargs)
            if processor_ref is None:
                processor_ref = load_ref

        if self.backend_cfg.get("compile_model"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        self.processor = AutoProcessor.from_pretrained(
            processor_ref,
            trust_remote_code=kwargs["trust_remote_code"],
        )
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
                images_batch = []
                videos_batch = []
                for message_set in messages_batch:
                    image_inputs, video_inputs = process_vision_info(message_set)
                    if image_inputs:
                        images_batch.extend(image_inputs)
                    if video_inputs:
                        videos_batch.extend(video_inputs)
                if images_batch and "images" in proc_sig.parameters:
                    kwargs["images"] = images_batch
                if videos_batch and "videos" in proc_sig.parameters:
                    kwargs["videos"] = videos_batch
            elif messages is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                if image_inputs is not None and "images" in proc_sig.parameters:
                    kwargs["images"] = image_inputs
                if video_inputs is not None and "videos" in proc_sig.parameters:
                    kwargs["videos"] = video_inputs
        inputs = self.processor(**kwargs)
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)
        return inputs


def _existing_local_path(value: Any) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    return path.resolve() if path.exists() else None


def _looks_like_adapter_dir(path: Path | None) -> bool:
    if path is None:
        return False
    return (path / "adapter_config.json").exists()


def _looks_like_hf_model_dir(path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    expected_files = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
    ]
    return any((path / filename).exists() for filename in expected_files)


def _resolve_training_output_model_dir(path: Path) -> Path | None:
    if path.is_dir():
        best_hint = _read_best_checkpoint_hint(path)
        if best_hint is not None:
            return best_hint

    if _looks_like_hf_model_dir(path):
        return path.resolve()

    trainer_states = sorted(path.rglob("trainer_state.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for trainer_state in trainer_states:
        try:
            payload = json.loads(trainer_state.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        for key in ("best_model_checkpoint", "last_model_checkpoint"):
            checkpoint_ref = payload.get(key)
            if not checkpoint_ref:
                continue
            checkpoint_path = Path(str(checkpoint_ref))
            if checkpoint_path.exists():
                return checkpoint_path.resolve()
        if _looks_like_hf_model_dir(trainer_state.parent):
            return trainer_state.parent.resolve()

    checkpoint_dirs = [candidate for candidate in path.rglob("checkpoint-*") if candidate.is_dir()]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda candidate: (_checkpoint_step(candidate), candidate.stat().st_mtime), reverse=True)
        return checkpoint_dirs[0].resolve()
    return None


def _read_best_checkpoint_hint(path: Path) -> Path | None:
    hint_path = path / "best_model_checkpoint.txt"
    if not hint_path.exists():
        return None

    try:
        checkpoint_ref = hint_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    if not checkpoint_ref:
        return None

    checkpoint_path = Path(checkpoint_ref)
    if checkpoint_path.exists():
        return checkpoint_path.resolve()
    return None


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


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
