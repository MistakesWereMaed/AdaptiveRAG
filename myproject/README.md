# Adaptive RAG

This repository is an implementation of the Adaptive-RAG paper by Jeong et al.

Adaptive RAG dynamically routes question answering requests between three strategies:

1. No retrieval
2. Single-step retrieval
3. Multi-step retrieval

The repository is organized around a simple end-to-end flow:

- build dense FAISS retrieval index from a corpus
- generate weak labels by comparing the three strategies against ground-truth answers
- train a router classifier on the labeled questions
- run the router at inference time to choose the best answering strategy

## Paper Summary

Adaptive-RAG is a retrieval-augmented QA framework designed for a realistic query mix: some questions are simple and can be answered without retrieval, some need one retrieval step, and others require multi-step retrieval and reasoning.

The paper’s core idea is to **adapt strategy by query complexity** instead of using one fixed retrieval policy for all inputs.

- Strategy A (No Retrieval): answer directly with the LLM for straightforward questions.
- Strategy B (Single-Step Retrieval): retrieve once, then answer with retrieved context.
- Strategy C (Multi-Step Retrieval): iteratively retrieve and reason for complex multi-hop questions.

To choose among A/B/C, the paper trains a smaller classifier that predicts query complexity. Since explicit complexity labels do not exist, the paper constructs training labels automatically using:

- Model-outcome silver labels (which strategy actually succeeds on each query)
- Inductive bias from dataset structure (single-hop vs multi-hop tendencies)

Across open-domain QA benchmarks (single-hop and multi-hop), the paper reports that this adaptive policy improves the performance/efficiency trade-off compared to fixed simple or fixed complex retrieval strategies.

## What This Repository Implements

This codebase implements the Adaptive-RAG pipeline described in the refactor plan:

1. Three QA strategies in one pipeline:
  - no retrieval
  - single-step retrieval
  - multi-step retrieval
2. Structured dense FAISS retrieval with sentence-transformer embeddings.
3. Title-aware context formatting for retrieval-augmented prompting.
4. Retrieval diagnostics that can be run without generation.
5. Weak-label generation and router training on top of the strategy outputs.
6. Streaming inference with per-query execution traces.

In short, the repository operationalizes the adaptive routing concept end-to-end: build the index, evaluate retrieval, generate predictions, label strategies, train the router, then evaluate adaptively.

## Layout

- `src/stages/` contains all stage-specific code and shared utilities:
  - `src/stages/common/` — Shared data and RAG utilities used by multiple stages (schemas, file I/O, HotpotQA loaders, LLM, retriever, pipeline)
  - `src/stages/<stage>/` — Per-stage entrypoints and workflow helpers (prepare_hotpotqa, build_index, generate_responses, generate_labels, train_router, evaluate)
- `data/` is the default location for prepared datasets and generated artifacts

All code is now consolidatedwithin `src/stages/` for easier readability and maintenance. Each stage contains only the files and functions needed for that stage, with cross-stage imports from `src.stages.common.*` when necessary.

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Run Order

Run the pipeline in this order:

1. Prepare HotpotQA
2. Build retrieval index
3. Run RAG pipelines to generate strategy outputs
4. Generate weak labels
5. Train router classifier
6. Evaluate predictions

## Data Format

Most utilities expect JSON or JSONL records with at least these fields:

```json
{
  "question": "...",
  "answer": "..."
}
```

Generated prediction files are keyed by `id` and contain model outputs separately from the source question:

```json
{
  "id": 1,
  "prediction": "...",
  "gold": "...",
  "strategy": "single",
  "retrieval_count": 1,
  "llm_calls": 1,
  "latency_s": 0.42
}
```

## Streaming Inference

Inference is streamed to JSONL so results are written incrementally instead of accumulating in memory. Each prediction record includes execution traces, retrieval count, LLM call count, latency, and the retrieved context used for the answer.

Retrieval corpora can use `title`, `text`, `content`, `passage`, `document`, or `context` fields. Plain text files with one document per line are still supported by the indexing script.

## Stage Entrypoints

The repository now exposes per-stage entrypoints under `src/stages/`. Each stage has a `main.py` that can be invoked with `python -m`.

Common stages and their purposes:

1. `src/stages/prepare_hotpotqa/main.py`
- Purpose: Downloads and preprocesses HotpotQA into local JSONL files.
- Main outputs: `data/hotpotqa/train.jsonl`, `data/hotpotqa/validation.jsonl`, and optional `data/hotpotqa/corpus.jsonl`.

2. `src/stages/build_index/main.py`
- Purpose: Builds the dense FAISS retrieval index and stores structured document metadata.
- Main outputs: `data/index/index.faiss` and `data/index/documents.json`.

3. `src/stages/evaluate/main.py`
- Purpose: Evaluates retrieval quality and other diagnostics (retrieval-only modes available).
- Main outputs: retrieval metrics JSON plus per-example JSONL diagnostics.

4. `src/stages/generate_responses/main.py`
- Purpose: Runs the QA pipeline with `no-rag`, `single`, or `multi` strategies.
- Main outputs: streaming prediction JSONL and per-strategy stats files.

5. `src/stages/generate_labels/main.py`
- Purpose: Creates weak supervision labels for router training by scoring the output of each strategy.

6. `src/stages/train_router/main.py`
- Purpose: Trains the query-complexity router/classifier from config-controlled training settings.

7. `src/stages/evaluate/main.py`
- Purpose: Computes answer metrics such as EM/F1 from predictions and reference answers.

## Ordered Commands

### 1) Prepare HotpotQA data

```bash
python -m src.stages.prepare_hotpotqa.main
```

### 2) Build dense retrieval index (one-time, reusable)

```bash
python -m src.stages.build_index.main
```

### 3) Evaluate retrieval quality

```bash
python -m src.stages.evaluate.main --mode retrieval --retriever dense --output experiments/retrieval_eval/dense_metrics.json
```

### 4) Run RAG pipelines to produce predictions

```bash
python -m src.stages.generate_responses.main
```

Index reuse behavior:

-- `index_dir` in the pipeline config is required and must contain `documents.json`.
-- If the index is missing, the stage exits with a clear error.
-- Build (or rebuild) the index first with `python -m src.stages.build_index.main`.

### 6) Generate weak labels (required before classifier training)

Label generation uses all three RAG strategies and scores each result against the ground truth answer.

```bash
python -m src.stages.generate_labels.main
```

### 7) Train router classifier


```bash
python -m src.stages.train_router.main
```

### 8) Evaluate predictions

```bash
python -m src.stages.evaluate.main
```