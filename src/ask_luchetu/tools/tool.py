from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ask_luchetu.context import ToolCall, ToolCallResponse


class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolExecutionResult(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error"] = "success"


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: ToolInputSchema
    run: Callable[[dict[str, Any]], ToolExecutionResult]


class ToolRegistry(BaseModel):
    tools: dict[str, Tool] = Field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def is_empty(self) -> bool:
        return len(self.tools) == 0

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema.model_dump(),
                },
            }
            for tool in self.tools.values()
        ]

    def execute(self, tool_call: ToolCall) -> ToolCallResponse:
        tool = self.get(tool_call.name)

        if tool is None:
            return ToolCallResponse(
                tool_call_id=tool_call.id,
                content=f"Error: Tool '{tool_call.name}' not found.",
                status="error",
                metadata={"recoverable": True, "reason": "unknown_tool"},
            )

        try:
            result = tool.run(tool_call.args)
            return ToolCallResponse(
                tool_call_id=tool_call.id,
                content=result.content,
                status=result.status,
                metadata=result.metadata,
            )
        except Exception as exc:
            return ToolCallResponse(
                tool_call_id=tool_call.id,
                content=f"Error executing '{tool_call.name}': {exc}",
                status="error",
                metadata={"recoverable": True, "reason": "tool_exception"},
            )
