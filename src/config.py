from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_path: str
    pretrained_model_name: str

    max_length: int = 512

    train_batch_size: int = 16
    eval_batch_size: int = 16

    num_workers: int = 4

    tool_token: str = "[TOOL]"