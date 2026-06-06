from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


class ToolSelectionCollator:
    """
    Pads:

    - input_ids
    - attention_mask
    - tool_token_mask

    and tool-level labels.

    Returns a batch of tensors suitable for training.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:

        #
        # Sequence padding
        #

        input_ids = [
            example["input_ids"]
            for example in examples
        ]

        attention_mask = [
            example["attention_mask"]
            for example in examples
        ]

        tool_token_mask = [
            example["tool_token_mask"]
            for example in examples
        ]

        batch_encoding = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        max_seq_len = batch_encoding["input_ids"].shape[1]

        padded_tool_token_mask = []

        for mask in tool_token_mask:
            padded_mask = mask + [0] * (
                max_seq_len - len(mask)
            )
            padded_tool_token_mask.append(padded_mask)

        #
        # Tool-level padding
        #

        labels = [
            example["labels"]
            for example in examples
        ]

        max_tools = max(
            len(label_row)
            for label_row in labels
        )

        padded_labels = []
        label_mask = []

        for label_row in labels:

            pad_size = max_tools - len(label_row)

            padded_labels.append(
                label_row + [0.0] * pad_size
            )

            label_mask.append(
                [1] * len(label_row)
                + [0] * pad_size
            )

        #
        # Optional metadata
        #

        tool_names = [
            example["tool_names"]
            for example in examples
        ]

        return {
            "input_ids": batch_encoding["input_ids"],
            "attention_mask": batch_encoding["attention_mask"],
            "tool_token_mask": torch.tensor(
                padded_tool_token_mask,
                dtype=torch.bool,
            ),
            "labels": torch.tensor(
                padded_labels,
                dtype=torch.float,
            ),
            "label_mask": torch.tensor(
                label_mask,
                dtype=torch.bool,
            ),
            "tool_names": tool_names,
        }