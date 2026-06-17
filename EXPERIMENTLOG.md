
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

## Results:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     test_exact_match      │    0.9821666479110718     │
│          test_f1          │     0.991936445236206     │
│         test_loss         │   0.024189338088035583    │
│      test_precision       │    0.9899111390113831     │
│        test_recall        │    0.9939700961112976     │
└───────────────────────────┴───────────────────────────┘



# EXP-0002 : prosy-shah

Goal:
Migrate from query-based inputs to conversation-based inputs.

Changes:
- Introduced Message domain object.
- Introduced MessageRole enum.
- Replaced query field with messages field.
- Added [SYSTEM], [USER], [ASSISTANT] special tokens.
- Updated GLiClass preprocessor to serialize conversations.

Expected outcome:
Equivalent performance to EXP-0001 because each example currently contains a single USER message.

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     test_exact_match      │    0.9815000295639038     │
│          test_f1          │    0.9917077422142029     │
│         test_loss         │    0.02932005561888218    │
│      test_precision       │    0.9882634878158569     │
│        test_recall        │    0.9951760768890381     │
└───────────────────────────┴───────────────────────────┘



## EXP-0003: 24a75ab [licht-byte]
Experiment run with distilbert base uncased on a sample of 2.8k examples generated with seeds from XLAM.

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     test_exact_match      │    0.8526315689086914     │
│          test_f1          │    0.9411764740943909     │
│         test_loss         │    0.20633403956890106    │
│      test_precision       │    0.9552238583564758     │
│        test_recall        │    0.9275362491607666     │
└───────────────────────────┴───────────────────────────┘

### Observations

A small set of hand-crafted evaluation scenarios was used to qualitatively assess model behavior beyond aggregate metrics.

The model performed well on simple single-turn requests and was generally able to identify the correct tool when the user's intent was explicit. It also demonstrated reasonable separation between relevant and irrelevant tools.

However, performance degraded when conversations contained topic shifts or distractor context. In multi-turn conversations, the model sometimes assigned elevated scores to tools associated with earlier parts of the conversation rather than focusing exclusively on the most recent user request.

This suggests the model has learned strong lexical and semantic associations between queries and tools, but has not yet fully learned conversational relevance and recency. Additional training data emphasizing topic changes, distractor conversations, and reference resolution may help improve robustness in realistic multi-turn settings.