# Data Change Log

This file tracks changes to datasets, preprocessing, labeling strategy, and train/dev/test splits.

---

## 2026-06-06 — Initial Dataset Baseline

### Dataset
Tool Selection Dataset v1

### Description
Initial dataset for training a dynamic tool-selection model.

Each example contains:

- A user query
- A list of candidate tools
- One or more correct tool selections

Schema:

```python
class Tool(BaseModel):
    name: str
    description: str

class ClassificationExample(BaseModel):
    id: int
    query: str
    answer: List[str]
    tools: List[Tool]
```

## Splits
train
dev
test

Stored as HuggingFace Arrow datasets.

## Notes

The candidate tool set is dynamic per example.

No global label vocabulary is defined.

Tool names and descriptions are treated as input features rather than fixed classes.

## Impact

Enables evaluation of tool-selection architectures that can generalize to previously unseen tools.

## 2026-06-17 - Dataset Generation Framework
Added Synthetic Scenario-Based Dataset Generation

Implemented a synthetic dataset generation pipeline for tool selection experiments. The pipeline transforms XLAM function-calling examples into conversational tool-selection examples using a collection of scenario generators.

Current scenarios include:

- Single Turn
- System Prompt
- Clarification
- Follow Up
- Reference Resolution
- Distractor Conversation
- Missing Tool

Generation follows a one-to-one mapping between source XLAM examples and synthetic examples to simplify provenance tracking and prevent train/test leakage.

Generated examples are written incrementally to JSONL files to support resumable dataset generation and avoid data loss during long-running jobs. Each example records the scenario used for generation, enabling future evaluation and error analysis by scenario type.

A separate dataset publishing step converts the generated JSONL files into a Hugging Face Dataset, creates train/dev/test splits, and uploads the resulting dataset to the Hugging Face Hub.

In total:
- train: 2.28k
- dev: 285
- test: 285