from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from .base import VlmBackend

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _ensure_venv_bin_on_path() -> None:
    """flashinfer 등 vLLM 내부 JIT 컴파일이 `ninja`를 subprocess로 찾는데,
    `.venv/bin/python`을 직접 호출하면(활성화 없이) 그 venv의 bin이 PATH에
    없어서 못 찾는 문제가 있다 — 여기서 명시적으로 넣어준다.

    주의: `sys.executable`을 `.resolve()`하면 안 된다 — uv venv의
    `.venv/bin/python`은 실제 인터프리터로 가는 심볼릭 링크라, resolve()하면
    venv의 bin이 아니라 base interpreter의 bin(거기엔 ninja가 없음)으로
    빠져서 이 함수가 무의미해진다."""
    venv_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if venv_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([venv_bin, *path_entries])


class QwenVllmBackend(VlmBackend):
    """vLLM 안에서 직접 모델을 띄워 생성한다 (외부 API 서버 호출이 아니라 in-process LLM).

    Qwen3.5/3.8처럼 하나의 체크포인트가 thinking/instruct 두 모드를 겸하는 모델을 위해
    `backend_cfg.thinking`으로 `chat_template_kwargs={"enable_thinking": ...}`를 넘긴다.
    thinking=true면 응답에 <think>...</think> 블록이 섞여 나오는데, 최종 JSON 파싱을
    방해하지 않도록 이 블록을 잘라내고 반환한다.
    """

    supports_batch = True

    def __init__(self, *, backend_cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> None:
        super().__init__(backend_cfg=backend_cfg, generation_cfg=generation_cfg)
        _ensure_venv_bin_on_path()
        try:
            from vllm import LLM, SamplingParams
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("vllm package is required for the qwen_vllm backend") from exc

        self.SamplingParams = SamplingParams
        self.thinking = bool(backend_cfg.get("thinking", False))

        llm_kwargs: dict[str, Any] = {
            "model": backend_cfg["model_name"],
            "trust_remote_code": bool(backend_cfg.get("trust_remote_code", True)),
            "limit_mm_per_prompt": {"image": int(backend_cfg.get("max_images_per_prompt", 16))},
            # 이 환경의 vllm_flash_attn/quack/cutlass 조합이 깨져 있어
            # (cutlass.cute.core에 ThrMma가 없음) ViT용 flash-attn 커널을 못 씀 —
            # SDPA로 강제 전환해서 우회한다.
            "mm_encoder_attn_backend": backend_cfg.get("mm_encoder_attn_backend", "TORCH_SDPA"),
        }
        if backend_cfg.get("tensor_parallel_size"):
            llm_kwargs["tensor_parallel_size"] = int(backend_cfg["tensor_parallel_size"])
        if backend_cfg.get("gpu_memory_utilization"):
            llm_kwargs["gpu_memory_utilization"] = float(backend_cfg["gpu_memory_utilization"])
        if backend_cfg.get("max_model_len"):
            llm_kwargs["max_model_len"] = int(backend_cfg["max_model_len"])
        # continuous batching/chunked prefill은 배치 구성에 따라 부동소수점 연산 순서가
        # 달라져서 greedy(temperature=0)여도 실행마다 살짝 다른 토큰이 나올 수 있다.
        # 완전한 재현성이 필요하면 이 둘을 꺼서 그 변동 요인을 없앤다 (처리량은 떨어짐).
        if "enforce_eager" in backend_cfg:
            llm_kwargs["enforce_eager"] = bool(backend_cfg["enforce_eager"])
        if "enable_chunked_prefill" in backend_cfg:
            llm_kwargs["enable_chunked_prefill"] = bool(backend_cfg["enable_chunked_prefill"])
        self.llm = LLM(**llm_kwargs)

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self.generate_batch([messages])[0]

    def generate_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        chat_messages_batch = [_to_vllm_chat_messages(messages) for messages in messages_batch]
        outputs = self.llm.chat(
            chat_messages_batch,
            sampling_params=_sampling_params(self.SamplingParams, self.generation_cfg),
            chat_template_kwargs={"enable_thinking": self.thinking},
            use_tqdm=False,
        )
        texts = [output.outputs[0].text if output.outputs else "" for output in outputs]
        return [_strip_thinking(text) for text in texts]


def _sampling_params(cls, generation_cfg: dict[str, Any]):
    do_sample = bool(generation_cfg.get("do_sample", False))
    kwargs: dict[str, Any] = dict(
        max_tokens=int(generation_cfg.get("max_new_tokens", 220)),
        temperature=float(generation_cfg.get("temperature", 0.2)) if do_sample else 0.0,
        top_p=float(generation_cfg.get("top_p", 0.9)) if do_sample else 1.0,
        repetition_penalty=float(generation_cfg.get("repetition_penalty", 1.1)),
    )
    if generation_cfg.get("seed") is not None:
        kwargs["seed"] = int(generation_cfg["seed"])
    return cls(**kwargs)


def _to_vllm_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chat_messages = []
    for message in messages:
        content = []
        for item in message.get("content", []):
            if item.get("type") == "text":
                content.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image":
                content.append({"type": "image_pil", "image_pil": item["image"]})
        chat_messages.append({"role": message.get("role", "user"), "content": content})
    return chat_messages


def _strip_thinking(text: str) -> str:
    """<think>...</think> 블록을 잘라낸다.

    vLLM의 chat template가 `add_generation_prompt`로 "<think>\n"을 프롬프트 쪽에
    미리 붙여주기 때문에, 실제로 생성되는 텍스트에는 여는 태그 없이 닫는
    "</think>"만 나오는 경우가 있다 (thinking=True일 때 항상 이 케이스였음).
    그래서 여는/닫는 태그가 둘 다 있으면 그 블록만 지우고, 닫는 태그만 있으면
    거기까지 전부(=thinking 부분)를 잘라낸다.
    """
    raw = text or ""
    if "<think>" in raw and "</think>" in raw:
        return _THINK_BLOCK_RE.sub("", raw).strip()
    if "</think>" in raw:
        return raw.split("</think>", 1)[1].strip()
    return raw.strip()
