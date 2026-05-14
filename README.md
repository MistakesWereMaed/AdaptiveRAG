# Adaptive-RAG Reproduction and Extension

Reproduction and extension of the paper:

> **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**
> Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park
> KAIST, 2024

This repository reproduces the Adaptive-RAG pipeline using the official IRCoT codebase as the foundation, while re-implementing the adaptive routing framework, preprocessing pipeline, classifier training, router inference, evaluation pipeline, and table reproduction scripts.

In addition to reproducing the original paper, this project also evaluates an extension to the routing classifier by replacing the original generative T5-Large router with discriminative encoder-based classifiers such as DeBERTa-v3.

---

# Repository Overview

The project contains:

* Full Adaptive-RAG preprocessing pipeline
* Dataset conversion and subsampling
* BM25 retrieval indexing
* Retrieval-augmented generation
* Automatic query-complexity labeling
* Router classifier training
* Adaptive routing evaluation
* Paper table reproduction scripts
* Encoder-based router improvements

The implementation closely follows the official repository behavior and interfaces wherever possible.

---

# Project Structure

```text
AdaptiveRAG/
│
├── classifier/
├── data/
├── predictions/
├── processed_data/
├── retriever_server/
├── router/
├── shell_scripts/
├── scripts/
├── results/
└── README.md
```

---

# Supported Datasets

## Single-Hop QA

* SQuAD
* Natural Questions
* TriviaQA

## Multi-Hop QA

* MuSiQue
* HotpotQA
* 2WikiMultiHopQA

---

# Supported Retrieval Strategies

| Strategy   | Description                |
| ---------- | -------------------------- |
| `nor_qa`   | No retrieval               |
| `oner_qa`  | Single-step retrieval      |
| `ircot_qa` | Multi-step IRCoT retrieval |

---

# Router Models

## Original Paper Router

* T5-Small
* T5-Base
* T5-Large

## Additional Improvement Routers

* DeBERTa-v3-Large

---

# Hardware Requirements

## Recommended

* NVIDIA V100 / A100 / RTX 4090
* 32GB+ GPU VRAM recommended
* Linux environment
* CUDA 11.8+

## Minimum Practical Setup

* 24GB VRAM GPU
* 32GB RAM

---

# Estimated Runtime

The full original pipeline is extremely expensive.

## Recommended Path (Using Official Prediction Artifacts)

This repository defaults to using downloaded official prediction artifacts where possible.

| Stage                            | Estimated Time |
| -------------------------------- | -------------- |
| Environment setup                | 10–20 min      |
| Dataset download + preprocessing | 20–40 min      |
| Router training                  | 1–2 hours      |
| Router evaluation                | 10–30 min      |
| Table reproduction               | <5 min         |

## Full End-to-End Regeneration (Optional)

| Stage                  | Estimated Time |
| ---------------------- | -------------- |
| Elasticsearch indexing | 2-3 hours     |
| Retrieval generation   | 12–48+ hours   |
| Total full pipeline    | 1–3 days       |

The retrieval generation stage is the primary bottleneck.

---

# Quickstart

This is the recommended workflow.

## Clone Repository

```bash
git clone <repo_url>
cd AdaptiveRAG
```

---


## Step 1 — Environment Setup

```bash
bash shell_scripts/create_env.bash
```

---

## Step 2 — Download Datasets

```bash
bash shell_scripts/download_raw_data.bash
```

---

## Step 3 — Download Official Prediction Data

```bash
bash shell_scripts/download_official_data.bash
```

---

## Step 4 — Process Datasets

```bash
bash shell_scripts/run_all_processing.bash
```

---

## Step 5 — Build Subsampled Splits

```bash
bash shell_scripts/run_all_subsampling.bash
```

---

## Step 6 — Label Training Data

```bash
bash shell_scripts/label_data.bash
```

This generates:

* Silver labels
* Binary labels
* Train/validation router datasets

---

## Step 7 — Train Router

```bash
bash shell_scripts/run_train_router.bash
```

---

## Step 8 — Run Adaptive Router

```bash
bash shell_scripts/run_router.bash
```

This performs:

* Query routing
* Routed prediction assembly
* Adaptive evaluation

---

## Step 9 — Reproduce Paper Tables

```bash
bash shell_scripts/reproduce_tables.bash
```

Generated outputs:

```text
results/
├── table_1.md
├── table_2.md
├── table_3.md
├── table_4.md
├── table_5.md
├── table_6.md
└── results.md
```

---

# Optional Full End-to-End Generation Pipeline

The following stages are optional because the runtime is prohibitively expensive for practical reproduction.

These are only necessary if regenerating all retrieval outputs from scratch.

---

## Optional Step 4 — Download Elasticsearch

```bash
bash shell_scripts/download_elasticsearch.bash
```

---

## Optional Step 5 — Start Retrieval Servers

```bash
bash shell_scripts/start_servers.bash
```

This launches:

* Elasticsearch
* Retriever server
* LLM server

---

## Optional Step 6 — Build All Indices

```bash
bash shell_scripts/build_all_indices.bash
```

---

## Optional Step 7 — Generate All Retrieval Outputs

```bash
bash shell_scripts/run_all_generation.bash
```

This stage may take multiple days depending on hardware.

---

## Optional Step 8 — Stop Servers

```bash
bash shell_scripts/stop_servers.bash
```

---

# Full Script Order

The complete pipeline order is:

```text
1  create_env
2  download_raw_data
3  download_official_data
4  download_elasticsearch
5  run_all_processing
6  run_all_subsampling
7  start_servers
8  build_all_indices
9  run_all_generation
10 label_data
11 stop_servers
12 run_train_router
13 run_router
14 reproduce_tables
```

---

# Improvement Over Original Paper

The original paper uses a generative T5-Large model as the query-complexity router.

This repository additionally evaluates discriminative encoder-based routers such as:

* DeBERTa-v3-Large

## Hypothesis

Query complexity routing is fundamentally a classification task rather than a text-generation task.

Encoder-only architectures should therefore:

* improve routing accuracy,
* reduce inference overhead,
* reduce training instability,
* and improve overall Adaptive-RAG efficiency.

The resulting router metrics and reproduced paper tables include these comparisons directly alongside the original T5-Large implementation.

---

# Reproduced Tables

This repository reproduces:

| Table   | Description                |
| ------- | -------------------------- |
| Table 1 | Average QA results         |
| Table 2 | Per-dataset QA results     |
| Table 3 | Query routing distribution |
| Table 4 | Training-data ablations    |
| Table 5 | Case studies               |
| Table 6 | Classifier-size ablations  |

Some original paper rows remain incomplete because only FLAN-T5-XL retrieval artifacts were generated locally.

---

# Notes

* The repository prioritizes reproducibility over optimization.
* All random seeds are fixed where possible.
* The implementation follows the official IRCoT interfaces closely.
* The retrieval generation stages are extremely computationally expensive and were therefore replaced with official artifacts for practical reproduction.

---

# Citation

```bibtex
@article{jeong2024adaptiverag,
  title={Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity},
  author={Jeong, Soyeong and Baek, Jinheon and Cho, Sukmin and Hwang, Sung Ju and Park, Jong C.},
  journal={arXiv preprint arXiv:2403.14403},
  year={2024}
}
```
