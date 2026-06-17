from src.scenarios.base import Scenario
from src.domain import Tool, ToolSelectionExample, FunctionCallingExample, Message, MessageRole
from typing import List
import random
from src.scenarios.constants import SYSTEM_PROMPTS

class SystemPromptScenario(Scenario):
    """
    Simulate single turn conversation where the system prompt is dynamically selected from a list of predefined prompts.
    """
    name = "system_prompt"

    def __init__(self):
        self.system_prompts = SYSTEM_PROMPTS

    def generate(self, example: FunctionCallingExample) -> ToolSelectionExample:
        prompt = random.choice(self.system_prompts)
        messages = [
            Message(role=MessageRole.SYSTEM, content=prompt),
            Message(role=MessageRole.USER, content=example.query)
        ]
        answer = [answer.name for answer in example.answers]
        return ToolSelectionExample(
            id=example.id,
            messages=messages,
            tools=[Tool(name=t.name, description=t.description) for t in example.tools],
            answer=answer,
            scenario=self.name,
        )