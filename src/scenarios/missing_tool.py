from src.scenarios.base import Scenario
from src.domain import Tool, ToolSelectionExample, FunctionCallingExample, Message, MessageRole

class MissingToolScenario(Scenario):
    """
    Simulate conversation where the correct tool is not mentioned at all, testing the model's ability to handle missing tools and select "none" or a default option.
    """
    name = "missing_tool"

    def generate(self, example: FunctionCallingExample) -> ToolSelectionExample:
        messages = [
            Message(role=MessageRole.USER, content=example.query)
        ]
        tools =[Tool(name = tool.name, description = tool.description) for tool in example.tools if tool.name in [answer.name for answer in example.answers]]
        answer = []
        return ToolSelectionExample(
            id=example.id,
            messages=messages,
            tools=tools,
            answer=answer,
            scenario=self.name,
        )