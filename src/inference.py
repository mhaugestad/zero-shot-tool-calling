from pathlib import Path
import json

import torch
from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.domain import (
    Message,
    Tool,
)
from src.model import GLiClassModel
from src.preprocessor import GLiClassPreprocessor


class ToolPrediction(BaseModel):
    name: str
    score: float


class ToolSelector:

    def __init__(
        self,
        model: GLiClassModel,
        tokenizer,
        max_length: int = 512,
        device: str = "cpu",
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = GLiClassPreprocessor(
            tokenizer=tokenizer,
            max_length=max_length,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cpu",
    ) -> "ToolSelector":

        model_path = Path(model_path)

        #
        # Tokenizer
        #

        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        #
        # Config
        #

        with open(
            model_path / "config.json"
        ) as f:

            config = json.load(f)

        #
        # Model
        #

        model = GLiClassModel(
            pretrained_model_name=config[
                "pretrained_model_name"
            ],
            vocab_size=len(tokenizer),
        )

        #
        # Weights
        #

        state_dict = load_file(
            model_path
            / "pytorch_model.safetensors"
        )

        model.load_state_dict(
            state_dict
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=config.get(
                "max_length",
                512,
            ),
            device=device,
        )

    def _prepare_inputs(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> dict[str, torch.Tensor]:

        text = self.preprocessor.serialize(
            messages=messages,
            tools=tools
        )

        tokenized = self.preprocessor.tokenize(
            text
        )

        return {
            key: torch.tensor(value)
            .unsqueeze(0)
            .to(self.device)
            for key, value in tokenized.items()
        }

    @torch.no_grad()
    def predict_scores(
        self,
        messages: list[Message],
        tools: list[Tool],
    ) -> list[float]:

        batch = self._prepare_inputs(
            messages=messages,
            tools=tools,
        )

        num_tools = len(tools)

        label_mask = torch.ones(
            (1, num_tools),
            dtype=torch.bool,
            device=self.device,
        )

        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tool_token_mask=batch[
                "tool_token_mask"
            ],
            label_mask=label_mask,
        )

        scores = torch.sigmoid(
            output.logits
        )

        return (
            scores.squeeze(0)
            .cpu()
            .tolist()
        )

    @torch.no_grad()
    def predict(
        self,
        messages: list[Message],
        tools: list[Tool],
        threshold: float = 0.5,
    ) -> list[ToolPrediction]:

        scores = self.predict_scores(
            messages=messages,
            tools=tools,
        )

        predictions = []

        for tool, score in zip(
            tools,
            scores,
        ):

            predictions.append(
                ToolPrediction(
                    name=tool.name,
                    score=float(score),
                )
            )

        predictions.sort(
            key=lambda p: p.score,
            reverse=True,
        )

        return predictions

    @torch.no_grad()
    def predict_names(
        self,
        messages: list[Message],
        tools: list[Tool],
        threshold: float = 0.5,
    ) -> list[str]:

        return [
            prediction.name
            for prediction in self.predict(
                messages=messages,
                tools=tools,
                threshold=threshold,
            )
        ]