from src.scenarios.base import Scenario
from src.domain import Tool, ToolSelectionExample, FunctionCallingExample
from src.scenarios.domain import GeneratedConversation
import json
from openai import OpenAI

class FollowUpScenario(Scenario):
    """
    Simulate multi turn conversation where the assistant asks follow-up questions before tool selection.
    """
    name = "follow_up"

    def __init__(self, client: OpenAI = None):
        self.client = client if client else OpenAI()
        self.system_prompt = """\
        You are generating synthetic training data for a tool-selection model.

        Rewrite the request as a natural conversation where the user first introduces the topic and later asks the actual question.

        Requirements:
        - Use exactly three messages.
        - The first user message should establish context.
        - The assistant should ask a brief follow-up question.
        - The final user message should contain the actual request.
        - Preserve the original intent.
        - The correct tool selection must remain unchanged.
        - Return only valid JSON.\
        """
        self.prompt = """\
        Original request:

        query: {query}\
        answers: {answers}
        """

    def generate(self, example: FunctionCallingExample) -> ToolSelectionExample:
        response = self.client.beta.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.prompt.format(
                        query=example.query,
                        answers=json.dumps([answer.model_dump() for answer in example.answers]),
                    ),
                },
            ],
            response_format=GeneratedConversation,
        )

        conversation = response.choices[0].message.parsed

        return ToolSelectionExample(
            id=example.id,
            messages=conversation.messages,
            tools=[Tool(name=t.name, description=t.description) for t in example.tools],
            answer=[answer.name for answer in example.answers],
            scenario=self.name,
        )
