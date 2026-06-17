from typing import List
from pydantic import BaseModel
from enum import StrEnum


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: MessageRole
    content: str

class Answer(BaseModel):
    name: str
    arguments: dict

class Tool(BaseModel):
    name: str
    description: str

    def __hash__(self):
        return hash(self.name)

class ToolWithParameters(Tool):
    parameters: dict

class FunctionCallingExample(BaseModel):
    id: int
    query: str
    answers: List[Answer]
    tools: List[ToolWithParameters]

class Tool(BaseModel):
    name: str
    description: str

class FunctionCallingExample(BaseModel):
    id: int
    query: str
    answers: List[Answer]
    tools: List[ToolWithParameters]

class ToolSelectionExample(BaseModel):
    id: int
    messages: List[Message]
    tools: List[Tool]
    answer: List[str]
    scenario: str | None = None

    def __repr__(self) -> str:

        lines = [
            f"ToolSelectionExample(id={self.id})"
        ]

        if self.scenario:
            lines.append(
                f"Scenario: {self.scenario}"
            )

        lines.append("")
        lines.append("Messages:")

        for message in self.messages:
            lines.append(
                f"  [{message.role.upper()}]"
            )
            lines.append(
                f"  {message.content}"
            )
            lines.append("")

        lines.append("Tools:")

        for tool in self.tools:
            marker = "✓" if tool.name in self.answer else " "
            lines.append(
                f"  [{marker}] {tool.name}"
            )

        lines.append("")
        lines.append(
            f"Answer: {self.answer}"
        )

        return "\n".join(lines)