from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoModel


@dataclass
class GLiClassOutput:
    logits: torch.Tensor


class GLiClassModel(nn.Module):

    def __init__(
        self,
        pretrained_model_name: str,
        vocab_size: int,
    ) -> None:

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name
        )

        self.encoder.resize_token_embeddings(
            vocab_size
        )

        self.hidden_size = (
            self.encoder.config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tool_token_mask: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> GLiClassOutput:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        #
        # CLS query embedding
        #
        query_embeddings = hidden_states[:, 0]

        #
        # Extract tool embeddings
        #
        flat_tool_embeddings = hidden_states[
            tool_token_mask
        ]

        batch_size = input_ids.shape[0]
        max_tools = label_mask.shape[1]

        tool_embeddings = torch.zeros(
            batch_size,
            max_tools,
            self.hidden_size,
            device=hidden_states.device,
        )

        start_idx = 0

        for batch_idx in range(batch_size):

            num_tools = int(
                label_mask[batch_idx].sum()
            )

            end_idx = start_idx + num_tools

            tool_embeddings[
                batch_idx,
                :num_tools,
            ] = flat_tool_embeddings[
                start_idx:end_idx
            ]

            start_idx = end_idx

        #
        # Dot product
        #
        logits = torch.sum(
            query_embeddings.unsqueeze(1)
            * tool_embeddings,
            dim=-1,
        )

        return GLiClassOutput(
            logits=logits
        )