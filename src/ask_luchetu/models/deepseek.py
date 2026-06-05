from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from ask_luchetu.context import (
    AIMessage,
    HumanMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolCallResponse,
)
from ask_luchetu.models.interface import ModelSettings


class DeepSeekClient:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        settings: ModelSettings = ModelSettings(),
        base_url: str = "https://api.deepseek.com/v1",
    ) -> None:
        self.model_name = model_name
        self.settings = settings
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def invoke(self, messages: list[Message]) -> AIMessage:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=self._convert_messages(messages),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            timeout=self.settings.timeout_seconds,
        )

        message = response.choices[0].message
        tool_calls = self._parse_tool_calls(message.tool_calls or [])

        return AIMessage(
            content=message.content or "",
            tool_calls=tool_calls,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                result.append({"role": "system", "content": message.content})

            elif isinstance(message, HumanMessage):
                result.append({"role": "user", "content": message.content})

            elif isinstance(message, AIMessage):
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if message.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                        for tc in message.tool_calls
                    ]
                result.append(msg)

            elif isinstance(message, ToolCallResponse):
                result.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                })

        return result

    def _parse_tool_calls(self, raw: list) -> list[ToolCall]:
        tool_calls = []
        for tc in raw:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args,
                )
            )
        return tool_calls
