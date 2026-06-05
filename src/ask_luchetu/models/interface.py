from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ask_luchetu.context import AIMessage, Message


class ModelSettings(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    extra: dict[str, Any] = Field(default_factory=dict)


class Model(Protocol):
    model_name: str
    settings: ModelSettings

    def invoke(self, messages: list[Message]) -> AIMessage: ...
