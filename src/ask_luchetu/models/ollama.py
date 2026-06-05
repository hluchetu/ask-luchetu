from __future__ import annotations

import json
import urllib.request
from typing import Any

from ask_luchetu.context import (
    AIMessage,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolCallResponse,
)
from ask_luchetu.models.interface import ModelSettings


class OllamaClient:
    def __init__(
        self,
        model_name: str,
        settings: ModelSettings = ModelSettings(),
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.settings = settings
        self._base_url = base_url.rstrip("/")

    def invoke(self, messages: list[Message]) -> AIMessage:
        body = json.dumps({
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "temperature": self.settings.temperature,
            "stream": False,
        }).encode()

        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            data = json.loads(response.read().decode())

        content = data.get("message", {}).get("content", "")
        return AIMessage(content=content)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                result.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                result.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                result.append({"role": "assistant", "content": message.content})
            elif isinstance(message, ToolCallResponse):
                result.append({"role": "tool", "content": message.content})

        return result
