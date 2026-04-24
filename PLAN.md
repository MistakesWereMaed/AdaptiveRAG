# Adaptive RAG Project Plan

## 1. Objective

Reproduce the core idea of *Adaptive-RAG*: dynamically selecting the most effective question answering (QA) strategy based on query complexity.

The system will:

* Route each question to one of three strategies:

  1. No retrieval
  2. Single-step retrieval
  3. Multi-step retrieval
* Learn routing via weak supervision derived from empirical performance
* Compare adaptive routing against static baselines

---

## 2. High-Level Architecture

### Components

1. **LLM (Inference Engine)**

   * Local deployment (Mistral 7B)
   * Used across all strategies

2. **Retriever**

   * Dense embedding-based retrieval (FAISS + sentence-transformers)

3. **RAG Pipelines**

   * No Retrieval
   * Single-step RAG
   * Multi-step RAG

4. **Label Generator**

   * Runs all strategies on training data
   * Assigns label based on best-performing strategy

5. **Classifier (Router)**

   * Predicts optimal strategy given a question
   * Implemented with PyTorch Lightning

6. **Evaluation Pipeline**

   * Compares adaptive routing vs static strategies

---

## 3. System Flow

### Training Phase

1. Input: QA dataset (question, answer)
2. For each question:

   * Run all three strategies
   * Score outputs against ground truth
   * Assign best-performing strategy as label
3. Train classifier on:

   * Input: question
   * Output: strategy label

---

### Inference Phase

1. Input: question
2. Classifier predicts strategy
3. Selected RAG pipeline executes
4. Output: final answer

---

## 4. Repository Structure

```
adaptive_rag/
│
├── configs/
│   ├── model.yaml
│   ├── retriever.yaml
│   └── train.yaml
│
├── src/
│   ├── data/
│   ├── retrieval/
│   ├── rag/
│   ├── labeling/
│   ├── classifier/
│   ├── routing/
│   └── utils/
│
├── scripts/
│   ├── build_index.py
│   ├── run_rag_pipelines.py
│   ├── train_classifier.py
│   └── evaluate.py
│
├── outputs/
└── plan.md
```

---

## 5. Dataset

### Primary Choice

* HotpotQA (multi-hop QA)

### Optional Additions

* Natural Questions
* TriviaQA

### Preprocessing

* Normalize answers
* Remove ambiguous samples (optional)
* Subsample for early experiments (e.g., 1k examples)

---

## 6. Retrieval System

### Embeddings

* sentence-transformers (e.g., MiniLM)

### Index

* FAISS (Flat or IVF)

### Retrieval Parameters

* Top-k:

  * Single-step: k=5
  * Multi-step: k=3 per step

---

## 7. RAG Pipelines

### 7.1 No Retrieval

* Direct LLM prompt
* Baseline for simple queries

---

### 7.2 Single-Step RAG

* Retrieve once
* Concatenate context
* Generate answer

---

### 7.3 Multi-Step RAG

#### Initial Version (Required)

* Fixed number of steps (2–3)
* Query refinement via concatenation

#### Optional Improvements

* Query rewriting
* Chain-of-thought prompting
* Iterative reasoning

---

## 8. Label Generation

### Process

For each question:

1. Run all strategies
2. Evaluate outputs using:

   * Exact Match (EM)
   * F1 score
3. Assign label:

   * 0 → No Retrieval
   * 1 → Single-step
   * 2 → Multi-step

### Output Format

```
{
  "question": "...",
  "label": 1
}
```

### Notes

* Drop samples where all strategies fail (optional)
* Handle ties carefully (choose simplest strategy or discard)

---

## 9. Classifier (Router)

### Model Options

#### Baseline

* BERT-style encoder

#### Input

* Question text

#### Output

* 3-class classification

---

### Training Setup

* Framework: PyTorch Lightning
* Loss: CrossEntropyLoss
* Optimizer: AdamW
* Learning rate: 2e-5

---

### Logging

* Weights & Biases

  * Train loss
  * Validation accuracy
  * Class distribution

---

## 10. Evaluation

### Metrics

* Exact Match (EM)
* F1 Score
* Latency (optional but valuable)

---

### Baselines

1. Always No Retrieval
2. Always Single-step
3. Always Multi-step

---

### Adaptive System

* Classifier-based routing

---

### Expected Outcomes

* Adaptive system ≥ best baseline in accuracy
* Lower average compute than always multi-step

---

## 11. HPC Execution Strategy

### Job Types

1. Index Building (CPU/GPU)
2. RAG Pipeline Execution (GPU-heavy)
3. Classifier Training (GPU)
4. Evaluation (CPU/GPU)

---

### Recommendations

* Batch LLM inference to maximize throughput
* Cache retrieval results
* Store intermediate outputs to avoid recomputation

---

## 12. Milestones

### Phase 1 — Setup (Day 1–2)

* Environment ready
* LLM inference working
* Retriever implemented

---

### Phase 2 — Baselines (Day 3–4)

* No retrieval pipeline
* Single-step RAG
* Basic evaluation

---

### Phase 3 — Multi-Step RAG (Day 5–6)

* Implement iterative retrieval
* Validate outputs

---

### Phase 4 — Label Generation (Day 7–8)

* Run all pipelines
* Generate labeled dataset

---

### Phase 5 — Classifier (Day 9–10)

* Train routing model
* Validate accuracy

---

### Phase 6 — Integration (Day 11–12)

* End-to-end system
* Routing + inference

---

### Phase 7 — Evaluation (Day 13–14)

* Compare against baselines
* Analyze performance

---

## 13. Risks and Mitigations

### 1. Slow Inference

* Mitigation: batching, caching, smaller dataset

---

### 2. Noisy Labels

* Mitigation:

  * Filter low-confidence samples
  * Use thresholded scoring

---

### 3. Class Imbalance

* Mitigation:

  * Weighted loss
  * Oversampling minority classes

---

### 4. Multi-Step Drift

* Mitigation:

  * Limit steps (2–3)
  * Keep retrieval grounded

---

## 14. Stretch Goals

* Confidence-based routing (instead of hard classification)
* Reinforcement learning policy
* Dynamic number of retrieval steps
* Retrieval uncertainty features

---

## 15. Deliverables

* Codebase
* Trained classifier
* Evaluation report
* Plots (accuracy vs baseline, latency vs accuracy)

---

## 16. Success Criteria

Minimum:

* Working adaptive routing system
* Clear comparison vs baselines

Strong:

* Demonstrated improvement in efficiency or accuracy
* Insightful analysis of routing behavior

---

## 17. Immediate Next Steps

1. Verify Mistral 7B inference stability on HPC
2. Implement retriever + index
3. Run small-scale pipeline (≤100 samples)
4. Validate outputs before scaling

---
