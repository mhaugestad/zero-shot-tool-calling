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


class Tool(BaseModel):
    name: str
    description: str


class ToolSelectionExample(BaseModel):
    id: int
    messages: List[Message]
    tools: List[Tool]
    answer: List[str]