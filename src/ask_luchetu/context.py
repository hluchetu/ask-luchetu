from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    type: Literal["system"] = "system"
    content: str


class HumanMessage(BaseModel):
    type: Literal["human"] = "human"
    content: str


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class AIMessage(BaseModel):
    type: Literal["ai"] = "ai"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolCallResponse(BaseModel):
    type: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    status: Literal["success", "error"] = "success"
    metadata: dict[str, Any] = Field(default_factory=dict)


Message = Union[SystemMessage, HumanMessage, AIMessage, ToolCallResponse]


class ChatContext:
    def __init__(self, max_messages: int = 40) -> None:
        self.max_messages = max_messages
        self.items: list[Message] = []

    def _append(self, message: Message) -> None:
        self.items.append(message)
        self._compact()

    def add_system_message(self, content: str) -> SystemMessage:
        message = SystemMessage(content=content)
        self._append(message)
        return message

    def add_human_message(self, content: str) -> HumanMessage:
        message = HumanMessage(content=content)
        self._append(message)
        return message

    def add_ai_message(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> AIMessage:
        message = AIMessage(content=content, tool_calls=tool_calls or [])
        self._append(message)
        return message

    def add_tool_message(
        self,
        tool_call_id: str,
        content: str,
        status: Literal["success", "error"] = "success",
        metadata: dict[str, Any] | None = None,
    ) -> ToolCallResponse:
        message = ToolCallResponse(
            tool_call_id=tool_call_id,
            content=content,
            status=status,
            metadata=metadata or {},
        )
        self._append(message)
        return message

    def _compact(self) -> None:
        if len(self.items) <= self.max_messages:
            return
        from ask_luchetu.utils.messages import trim_messages

        self.items = trim_messages(self.items, max_messages=self.max_messages)
