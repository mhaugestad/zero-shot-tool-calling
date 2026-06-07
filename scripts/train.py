from pathlib import Path

import pytorch_lightning as pl
import yaml

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)

from src.datamodule import ToolSelectionDataModule
from src.module import ToolSelectionModule
from src.model import GLiClassModel
from src.config import DataConfig, ExperimentConfig


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def build_dataconfig(
    params: dict,
) -> DataConfig:

    return DataConfig(
        dataset_path="data/tool-selection/tool_selection_dataset",
        pretrained_model_name=params["model"][
            "pretrained_model_name"
        ],
        max_length=params["model"].get(
            "max_length",
            512,
        ),
        train_batch_size=params["training"][
            "batch_size"
        ],
        eval_batch_size=params["training"][
            "batch_size"
        ],
        num_workers=params["training"].get(
            "num_workers",
            4,
        ),
    )


def main():

    raw = load_params()
    params = ExperimentConfig.model_validate(raw)

    data_config = build_dataconfig(
        params.model_dump()
    )

    #
    # Data
    #

    datamodule = ToolSelectionDataModule(
        data_config
    )

    datamodule.setup()

    #
    # Model
    #

    model = GLiClassModel(
        pretrained_model_name=params.model.pretrained_model_name,
        vocab_size=len(
            datamodule.tokenizer
        ),
    )

    #
    # Lightning module
    #

    module = ToolSelectionModule(
        model=model,
        learning_rate=params.training.learning_rate,
    )

    #
    # Checkpoints
    #

    Path("models").mkdir(
        exist_ok=True
    )

    checkpoint_callback = (
        ModelCheckpoint(
            dirpath="models",
            filename=(
                "best-{epoch:02d}"
                "-{val_f1:.4f}"
            ),
            monitor="val_f1",
            mode="max",
            save_top_k=1,
        )
    )

    #
    # Trainer
    #

    trainer = pl.Trainer(
        max_epochs=params.training.max_epochs,
        callbacks=[
            checkpoint_callback
        ],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    #
    # Train
    #

    trainer.fit(
        module,
        datamodule=datamodule,
    )

    #
    # Test
    #

    trainer.test(
        module,
        datamodule=datamodule,
        ckpt_path="best",
    )


if __name__ == "__main__":
    main()