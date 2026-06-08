from typing import Any, List

from transformers import PreTrainedTokenizerBase
from src.domain import Message, MessageRole, Tool, ToolSelectionExample


class GLiClassPreprocessor:

    ROLE_TO_TOKEN = {
        MessageRole.SYSTEM: "[SYSTEM]",
        MessageRole.USER: "[USER]",
        MessageRole.ASSISTANT: "[ASSISTANT]",
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        tool_token: str = "[TOOL]",
        system_token: str = "[SYSTEM]",
        user_token: str = "[USER]",
        assistant_token: str = "[ASSISTANT]",
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.tool_token = tool_token
        self.tool_token_id = tokenizer.convert_tokens_to_ids(
            tool_token
        )

        self.system_token = system_token
        self.user_token = user_token
        self.assistant_token = assistant_token

    def __call__(
        self,
        example: dict[str, Any],
    ) -> dict[str, Any]:

        example = ToolSelectionExample.model_validate(
            example
        )

        serialized = self.serialize(
            messages=example.messages,
            tools=example.tools,
        )

        tokenized = self.tokenize(
            serialized
        )

        labels = self.build_labels(
            tools=example.tools,
            answers=example.answer,
        )

        return {
            **tokenized,
            "labels": labels,
            "tool_names": [
                tool.name
                for tool in example.tools
            ],
        }

    def serialize(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> str:

        parts = []

        #
        # Conversation
        #

        for message in messages:

            role = message["role"]

            role_token = self.ROLE_TO_TOKEN[
                    message.role
                ]

            parts.append(
                f"{role_token}\n"
                f"{message.content}"
            )

        #
        # Candidate tools
        #

        for tool in tools:

            parts.append(
                (
                    f"{self.tool_token}\n"
                    f"{tool.name}\n"
                    f"{tool.description}"
                )
            )

        return "\n\n".join(parts)

    def tokenize(
        self,
        text: str,
    ) -> dict[str, list[int]]:

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids = encoding["input_ids"]

        tool_token_mask = [
            token_id == self.tool_token_id
            for token_id in input_ids
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "tool_token_mask": tool_token_mask,
        }

    def build_labels(
        self,
        tools: list[Tool],
        answers: list[str],
    ) -> list[float]:

        answer_set = set(answers)

        return [
            float(tool.name in answer_set)
            for tool in tools
        ]