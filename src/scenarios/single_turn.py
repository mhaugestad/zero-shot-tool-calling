from src.scenarios.base import Scenario
from src.domain import Tool, ToolSelectionExample, FunctionCallingExample, Message, MessageRole

class SingleTurnScenario(Scenario):
    """
    Simulate single turn conversation where the user query is directly followed by tool selection without any additional interactions.
    """
    name = "single_turn"

    def generate(self, example: FunctionCallingExample) -> ToolSelectionExample:
        messages = [
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