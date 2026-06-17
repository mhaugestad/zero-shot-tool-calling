from pydantic import BaseModel
from src.domain import Message
from typing import List

class GeneratedConversation(BaseModel):
    messages: List[Message]