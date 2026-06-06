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