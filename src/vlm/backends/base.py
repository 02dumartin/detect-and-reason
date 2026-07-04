from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VlmBackend(ABC):
    supports_batch = False

    def __init__(self, *, backend_cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> None:
        self.backend_cfg = backend_cfg
        self.generation_cfg = generation_cfg

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]]) -> str:
        raise NotImplementedError

    def generate_batch(self, messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        return [self.generate(messages) for messages in messages_batch]

