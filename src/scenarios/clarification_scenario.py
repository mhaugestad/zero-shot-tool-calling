from src.scenarios.base import Scenario
from src.domain import ToolSelectionExample, FunctionCallingExample, Tool
from src.scenarios.domain import GeneratedConversation
import json
from openai import OpenAI

class ClarificationScenario(Scenario):
    """
    Simulate multi turn conversation where the assistant asks for clarification before tool selection. 
    """

    name = "clarification"

    def __init__(self, client: OpenAI = None):
        self.client = client if client else OpenAI()
        self.system_prompt = """\
            You are generating synthetic training data for a tool-selection model.

            Your task is to rewrite a single user request into a short multi-turn conversation.

            Requirements:
            - Introduce one assistant clarification question.
            - The final user message must provide the missing information.
            - Preserve the original intent.
            - The correct tool selection must remain unchanged.
            - Generate realistic conversations.
            - Return only valid JSON.

            Output schema:

            {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "..."}
            ]
            }\
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
                        answers=json.dumps([answer.dict() for answer in example.answers]),
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
