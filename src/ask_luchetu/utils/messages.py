from __future__ import annotations

from pydantic import TypeAdapter

from ask_luchetu.context import (
    HumanMessage,
    Message,
    SystemMessage,
    ToolCallResponse,
)

MessageAdapter = TypeAdapter(list[Message])


def trim_messages(
    messages: list[Message],
    max_messages: int = 40,
    include_system: bool = True,
    start_on: type = HumanMessage,
) -> list[Message]:
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]

    rest = rest[-max_messages:]

    while rest and not isinstance(rest[0], start_on):
        rest = rest[1:]

    if include_system:
        return system + rest
    return rest
