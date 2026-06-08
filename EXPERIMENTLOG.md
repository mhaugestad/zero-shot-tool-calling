
---

## EXPERIMENTLOG.md

```md
# Experiment Log

This file tracks model architectures, training configurations, and experiment outcomes.

---

## EXP-0001 — GLiClass-style Dynamic Tool Selection Baseline: 4236122 [outer-corm]

### Date

2026-06-06

### Goal

Establish a baseline encoder-only architecture for dynamic tool selection.

The model should:

- Accept arbitrary candidate tools at inference time
- Support unseen tools
- Support changing tool inventories
- Predict one or more relevant tools for a user query

### Architecture

Inspired by GLiClass.

Input consists of:

- User query
- Candidate tool names
- Candidate tool descriptions

Tools are serialized into the input sequence using special label tokens.

Example:

```text
[CLS]

What's the weather tomorrow?

[TOOL]
weather
Get weather forecast

[TOOL]
calculator
Perform arithmetic calculations

[TOOL]
search
Search documents
```

## Model

Encoder-only Transformer.

Initial backbone:

DistilBERT (candidate baseline)

## Architecture:

Encode query and candidate tools jointly.
Extract tool representations from special [TOOL] tokens.
Compute similarity between query representation and each tool representation.
Apply sigmoid independently for each tool.
Train using BCE loss.
Why This Approach

Avoids fixed label vocabularies.

## Allows:

Dynamic tool inventories
Zero-shot generalization to new tools
Reuse of tool descriptions as semantic supervision

## Metrics
- Precision
- Recall
- F1
- Exact Match

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     test_exact_match      │    0.9821666479110718     │
│          test_f1          │     0.991936445236206     │
│         test_loss         │   0.024189338088035583    │
│      test_precision       │    0.9899111390113831     │
│        test_recall        │    0.9939700961112976     │
└───────────────────────────┴───────────────────────────┘
## Results: