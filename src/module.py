import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from src.model import GLiClassModel

from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)


class ToolSelectionModule(pl.LightningModule):

    def __init__(
        self,
        model: GLiClassModel,
        learning_rate: float = 2e-5,
    ) -> None:

        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()

    def forward(
        self,
        **batch,
    ):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tool_token_mask=batch["tool_token_mask"],
            label_mask=batch["label_mask"],
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

        loss = loss * label_mask.float()

        loss = loss.sum() / label_mask.sum()

        return loss

    def training_step(
        self,
        batch,
        batch_idx,
    ):

        output = self.forward(**batch)

        loss = self._compute_loss(
            logits=output.logits,
            labels=batch["labels"],
            label_mask=batch["label_mask"],
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )

        return loss

    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        output = self.forward(**batch)

        loss = self._compute_loss(
            logits=output.logits,
            labels=batch["labels"],
            label_mask=batch["label_mask"],
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )

        predictions = (
            torch.sigmoid(output.logits) >= 0.5
        )

        predictions = predictions[
            batch["label_mask"]
        ]

        labels = batch["labels"][
            batch["label_mask"]
        ]

        self.val_precision.update(
            predictions,
            labels.int(),
        )

        self.val_recall.update(
            predictions,
            labels.int(),
        )

        self.val_f1.update(
            predictions,
            labels.int(),
        )

        return loss

    def test_step(
        self,
        batch,
        batch_idx,
    ):

        output = self.forward(**batch)

        loss = self._compute_loss(
            logits=output.logits,
            labels=batch["labels"],
            label_mask=batch["label_mask"],
        )

        self.log(
            "test_loss",
            loss,
            batch_size=batch["input_ids"].size(0),
        )

        predictions = (
            torch.sigmoid(output.logits) >= 0.5
        )

        predictions = predictions[
            batch["label_mask"]
        ]

        labels = batch["labels"][
            batch["label_mask"]
        ]

        self.test_precision.update(
            predictions,
            labels.int(),
        )

        self.test_recall.update(
            predictions,
            labels.int(),
        )

        self.test_f1.update(
            predictions,
            labels.int(),
        )

        return loss

    def on_validation_epoch_end(self):

        self.log(
            "val_precision",
            self.val_precision.compute(),
        )

        self.log(
            "val_recall",
            self.val_recall.compute(),
        )

        self.log(
            "val_f1",
            self.val_f1.compute(),
        )

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):

        self.log(
            "test_precision",
            self.test_precision.compute(),
        )

        self.log(
            "test_recall",
            self.test_recall.compute(),
        )

        self.log(
            "test_f1",
            self.test_f1.compute(),
        )

        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def configure_optimizers(self):

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )