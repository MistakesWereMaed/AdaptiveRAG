# Adaptive RAG

This repository is an implementation of the Adaptive-RAG paper by Jeong et al.

Adaptive RAG dynamically routes question answering requests between three strategies:

1. No retrieval
2. Single-step retrieval
3. Multi-step retrieval

The repository is organized around a simple end-to-end flow:

- build a Qdrant-backed retrieval index from a corpus
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

This codebase implements the main Adaptive-RAG pipeline described above:

1. Three QA strategies in one pipeline:
  - no retrieval
  - single-step retrieval
  - multi-step retrieval
2. Weak-label generation by running all strategies and scoring outputs against references.
3. A router/classifier trained on those labels to predict strategy choice.
4. Retrieval with sentence-transformers + Qdrant.
5. LLM inference powered by vLLM (continuous batching and KV-cache management handled by backend).
6. Single-process and python-launched multi-process dataset sharding for RAG pipeline runs (`--num-procs`, no `torchrun` required).

In short, the repository operationalizes the paper’s adaptive routing concept end-to-end: generate labels -> train complexity router -> route each query to the most suitable RAG strategy.

## Layout

- `configs/` holds YAML configuration stubs for model, retriever, and training settings
- `src/` contains the reusable library code
- `scripts/` contains command-line entry points for indexing, labeling, training, and evaluation
- `outputs/` is the default location for generated artifacts

`LocalLLM` defaults are defined in `configs/llm.yaml` and are vLLM-specific.

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

Retrieval corpora can use `text`, `content`, `passage`, or `document` fields. Plain text files with one document per line are also supported in the indexing script.

## Script Reference

This section describes what each script does and where it fits in the workflow.

1. `scripts/prepare_hotpotqa.py`
- Purpose: Downloads and preprocesses HotpotQA into local JSONL files.
- Main outputs: `data/hotpotqa/train.jsonl`, `data/hotpotqa/validation.jsonl`, and optional `data/hotpotqa/corpus.jsonl`.
- Run when: First step, or whenever you want to refresh local dataset files.

2. `scripts/build_index.py`
- Purpose: Builds a Qdrant retrieval collection from your corpus and saves index metadata.
- Main outputs: `documents.json` in the configured output directory and local Qdrant storage under `.qdrant_db`.
- Run when: After preparing corpus data; rerun only if corpus or embedding model changes.

3. `scripts/run_rag_pipelines.py`
- Purpose: Runs the QA pipeline with strategy `no`, `single`, `multi`, or `all`.
- Main outputs: Prediction JSON for selected strategy/strategies.
- Special behavior: Requires a prebuilt index configured in `configs/pipeline.yaml` (the directory must contain `documents.json`); exits with an error if missing.
- Parallel behavior: Supports python-based local sharding driven by `num_procs` in `configs/pipeline.yaml` (no `torchrun` needed).
- Inference backend: Uses vLLM via `src/llm.py`; internal batching/caching is handled by vLLM.
- Run when: To generate answers/predictions for train/dev/test questions.

4. `scripts/generate_labels.py`
- Purpose: Creates weak supervision labels for router training by scoring the output of each strategy.
- Main outputs: Labeled training JSON (e.g., `outputs/labeled_train.json`).
- Run when: Before training the classifier/router.

5. `scripts/train_classifier.py`
- Purpose: Trains the query-complexity router/classifier from config-controlled training settings.
- Main outputs: Trained checkpoint(s) and training logs.
- Run when: After weak labels are generated.

6. `scripts/evaluate.py`
- Purpose: Computes evaluation metrics (EM/F1) from predictions and reference answers.
- Main outputs: Printed metrics summary.
- Run when: After inference to compare model variants and settings.

## Ordered Commands

### 1) Prepare HotpotQA data

```bash
python -m scripts.prepare_hotpotqa --config configs/hotpotqa.yaml
```

### 2) Build retrieval index (one-time, reusable)

```bash
python -m scripts.build_index --config configs/retriever.yaml
```

### 3) Run RAG pipelines to produce predictions

```bash
python -m scripts.generate_responses --config configs/pipeline.yaml
```

Index reuse behavior:

- `index_dir` in the pipeline config is required and must contain `documents.json`.
- If the index is missing, the script exits with a clear error.
- Build (or rebuild) the index first with `python -m scripts.build_index --config configs/retriever.yaml`.

### 4) Generate weak labels (required before classifier training)

Label generation uses all three RAG strategies and scores each result against the ground truth answer.

```bash
python -m scripts.generate_labels --config configs/labels.yaml
```

### 5) Train router classifier


```bash
python -m scripts.train_classifier --config configs/train.yaml
```

### 6) Evaluate predictions

```bash
python -m scripts.evaluate --config configs/evaluate.yaml
```