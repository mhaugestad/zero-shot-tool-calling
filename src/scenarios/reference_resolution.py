from src.scenarios.base import Scenario
from src.domain import Tool, ToolSelectionExample
from src.scenarios.domain import GeneratedConversation
from openai import OpenAI

class ReferenceResolutionScenario(Scenario):
    """
    Simulate multi turn conversation where the assistant needs to resolve references in the user query before tool selection.
    """
    name = "reference_resolution"

    def __init__(self, client: OpenAI = None):
        self.client = client if client else OpenAI()
        self.system_prompt = """\
        You are generating synthetic training data for a tool-selection model.

        Rewrite the request as a conversation that requires reference resolution.

        Requirements:
        - The final user message should contain a pronoun or reference such as "it", "they", "that company", "those flights", etc.
        - The conversation must remain unambiguous.
        - Preserve the original intent.
        - The correct tool selection must remain unchanged.
        - Return only valid JSON.
        """
        self.prompt = """\
        Original request:

        query: {query}\
        answers: {answers}
        """

    def generate(self, example) -> ToolSelectionExample:
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
                        answers=[answer.name for answer in example.answers]
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
