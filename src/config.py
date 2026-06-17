from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class DataConfig:
    dataset_path: str
    pretrained_model_name: str

    max_length: int = 512

    train_batch_size: int = 16
    eval_batch_size: int = 16

    num_workers: int = 4

    tool_token: str = "[TOOL]"
    system_token: str = "[SYSTEM]"
    user_token: str = "[USER]"
    assistant_token: str = "[ASSISTANT]"


class ModelConfig(BaseModel):
    pretrained_model_name: str
    max_length: int = 512


class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    max_epochs: int
    num_workers: int = 4


class ExperimentConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    data: dict