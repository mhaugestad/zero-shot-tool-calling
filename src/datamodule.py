from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import pytorch_lightning as pl

from src.collators import (
    ToolSelectionCollator,
)
from src.preprocessor import (
    GLiClassPreprocessor,
)
from src.config import DataConfig


class ToolSelectionDataModule(
    pl.LightningDataModule,
):

    def __init__(
        self,
        config: DataConfig,
    ) -> None:

        super().__init__()

        self.config = config

        self.tokenizer = None
        self.collator = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(
        self,
        stage: str | None = None,
    ) -> None:

        #
        # tokenizer
        #

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name
        )

        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.config.tool_token,
                    self.config.system_token,
                    self.config.user_token,
                    self.config.assistant_token,
                ]
            }
        )

        #
        # preprocessing
        #

        preprocessor = GLiClassPreprocessor(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            tool_token=self.config.tool_token,
        )

        #
        # datasets
        #

        dataset = load_from_disk(
            self.config.dataset_path
        )

        dataset = dataset.map(
                preprocessor,
                batched=False,
            )
        
        dataset = dataset.filter(
            lambda ex: (
                len(ex["tool_names"])
                == sum(ex["tool_token_mask"])
            )
        )

        self.train_dataset = dataset["train"].map(
            preprocessor,
            batched=False,
            desc="Tokenizing train",
        )

        self.val_dataset = dataset["dev"].map(
            preprocessor,
            batched=False,
            desc="Tokenizing dev",
        )

        self.test_dataset = dataset["test"].map(
            preprocessor,
            batched=False,
            desc="Tokenizing test",
        )

        #
        # collator
        #

        self.collator = ToolSelectionCollator(
            tokenizer=self.tokenizer,
        )

    def train_dataloader(
        self,
    ) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(
        self,
    ) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(
        self,
    ) -> DataLoader:

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
        )