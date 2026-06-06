from typing import Any

from transformers import PreTrainedTokenizerBase


class GLiClassPreprocessor:

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        tool_token: str = "[TOOL]",
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.tool_token = tool_token
        self.tool_token_id = tokenizer.convert_tokens_to_ids(
            tool_token
        )

    def __call__(
        self,
        example: dict[str, Any],
    ) -> dict[str, Any]:

        serialized = self.serialize(
            query=example["query"],
            tools=example["tools"],
        )

        tokenized = self.tokenize(
            serialized
        )

        labels = self.build_labels(
            tools=example["tools"],
            answers=example["answer"],
        )

        return {
            **tokenized,
            "labels": labels,
            "tool_names": [
                tool["name"]
                for tool in example["tools"]
            ],
        }

    def serialize(
        self,
        query: str,
        tools: list[dict[str, Any]],
    ) -> str:

        parts = [query]

        for tool in tools:

            parts.append(
                (
                    f"{self.tool_token} "
                    f"{tool['name']}\n"
                    f"{tool['description']}"
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
        tools: list[dict[str, Any]],
        answers: list[str],
    ) -> list[float]:

        answer_set = set(answers)

        return [
            float(tool["name"] in answer_set)
            for tool in tools
        ]