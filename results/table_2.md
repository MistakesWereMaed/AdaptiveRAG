# Table 2 — Per-Dataset FLAN-T5-XL Results

| Dataset | Type | Method | EM | F1 | Acc | Step | Time | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SQuAD | Single-step | No Retrieval | 3.60 | 10.50 | 5.00 | 0.00 | 0.11 | Paper |
| SQuAD | Single-step | Single-step Approach | 27.80 | 39.30 | 34.00 | 1.00 | 1.00 | Paper |
| SQuAD | Single-step | Adaptive Retrieval | 13.40 | 23.10 | 17.60 | 0.50 | 0.55 | Paper |
| SQuAD | Single-step | Self-RAG* | 2.20 | 11.20 | 18.40 | 0.63 | 0.50 | Paper |
| SQuAD | Single-step | Adaptive-RAG (Ours) | 26.80 | 38.30 | 33.00 | 1.37 | 2.02 | Paper |
| SQuAD | Single-step | Multi-step Approach | 24.40 | 35.60 | 29.60 | 4.52 | 9.03 | Paper |
| SQuAD | Single-step | Adaptive-RAG w/ Oracle | 32.00 | 45.60 | 38.20 | 1.24 | 1.60 | Paper |
| Natural Questions | Single-step | No Retrieval | 14.20 | 19.00 | 15.60 | 0.00 | 0.13 | Paper |
| Natural Questions | Single-step | Single-step Approach | 37.80 | 47.30 | 44.60 | 1.00 | 1.00 | Paper |
| Natural Questions | Single-step | Adaptive Retrieval | 28.20 | 36.00 | 33.00 | 0.50 | 0.56 | Paper |
| Natural Questions | Single-step | Self-RAG* | 31.40 | 39.00 | 33.60 | 0.63 | 0.17 | Paper |
| Natural Questions | Single-step | Adaptive-RAG (Ours) | 37.80 | 47.30 | 44.60 | 1.00 | 1.00 | Paper |
| Natural Questions | Single-step | Multi-step Approach | 38.60 | 47.80 | 44.20 | 5.04 | 10.18 | Paper |
| Natural Questions | Single-step | Adaptive-RAG w/ Oracle | 47.40 | 57.10 | 53.60 | 1.10 | 1.55 | Paper |
| TriviaQA | Single-step | No Retrieval | 25.00 | 31.80 | 27.00 | 0.00 | 0.13 | Paper |
| TriviaQA | Single-step | Single-step Approach | 53.60 | 62.40 | 60.20 | 1.00 | 1.00 | Paper |
| TriviaQA | Single-step | Adaptive Retrieval | 38.40 | 46.90 | 42.60 | 0.50 | 0.56 | Paper |
| TriviaQA | Single-step | Self-RAG* | 12.80 | 29.30 | 57.00 | 0.68 | 0.45 | Paper |
| TriviaQA | Single-step | Adaptive-RAG (Ours) | 52.20 | 60.70 | 58.20 | 1.23 | 1.54 | Paper |
| TriviaQA | Single-step | Multi-step Approach | 53.80 | 62.40 | 60.20 | 5.28 | 9.22 | Paper |
| TriviaQA | Single-step | Adaptive-RAG w/ Oracle | 61.60 | 70.20 | 66.40 | 0.79 | 1.10 | Paper |
| MuSiQue | Multi-step | No Retrieval | 2.40 | 10.70 | 3.20 | 0.00 | 0.11 | Paper |
| MuSiQue | Multi-step | Single-step Approach | 13.80 | 22.80 | 15.20 | 1.00 | 1.00 | Paper |
| MuSiQue | Multi-step | Adaptive Retrieval | 6.40 | 15.80 | 8.00 | 0.50 | 0.55 | Paper |
| MuSiQue | Multi-step | Self-RAG* | 1.60 | 8.10 | 12.00 | 0.73 | 0.51 | Paper |
| MuSiQue | Multi-step | Adaptive-RAG (Ours) | 23.60 | 31.80 | 26.00 | 3.22 | 6.61 | Paper |
| MuSiQue | Multi-step | Multi-step Approach | 23.00 | 31.90 | 25.80 | 3.60 | 7.58 | Paper |
| MuSiQue | Multi-step | Adaptive-RAG w/ Oracle | 24.80 | 38.50 | 27.00 | 1.98 | 3.99 | Paper |
| HotpotQA | Multi-step | No Retrieval | 16.60 | 22.71 | 17.20 | 0.00 | 0.11 | Paper |
| HotpotQA | Multi-step | Single-step Approach | 34.40 | 46.15 | 36.40 | 1.00 | 1.00 | Paper |
| HotpotQA | Multi-step | Adaptive Retrieval | 23.60 | 32.22 | 25.00 | 0.50 | 0.55 | Paper |
| HotpotQA | Multi-step | Self-RAG* | 6.80 | 17.53 | 29.60 | 0.73 | 0.45 | Paper |
| HotpotQA | Multi-step | Adaptive-RAG (Ours) | 42.00 | 53.82 | 44.40 | 3.55 | 5.99 | Paper |
| HotpotQA | Multi-step | Multi-step Approach | 44.60 | 56.54 | 47.00 | 5.53 | 9.38 | Paper |
| HotpotQA | Multi-step | Adaptive-RAG w/ Oracle | 51.20 | 64.00 | 54.80 | 1.59 | 2.77 | Paper |
| 2WikiMultiHopQA | Multi-step | No Retrieval | 27.40 | 32.04 | 27.80 | 0.00 | 0.10 | Paper |
| 2WikiMultiHopQA | Multi-step | Single-step Approach | 41.60 | 47.90 | 42.80 | 1.00 | 1.00 | Paper |
| 2WikiMultiHopQA | Multi-step | Adaptive Retrieval | 33.20 | 39.44 | 34.20 | 0.50 | 0.55 | Paper |
| 2WikiMultiHopQA | Multi-step | Self-RAG* | 4.60 | 19.59 | 38.80 | 0.93 | 0.49 | Paper |
| 2WikiMultiHopQA | Multi-step | Adaptive-RAG (Ours) | 40.60 | 49.75 | 46.40 | 2.63 | 4.68 | Paper |
| 2WikiMultiHopQA | Multi-step | Multi-step Approach | 49.60 | 58.85 | 55.40 | 4.17 | 7.37 | Paper |
| 2WikiMultiHopQA | Multi-step | Adaptive-RAG w/ Oracle | 53.00 | 62.30 | 59.40 | 1.01 | 1.69 | Paper |
| SQuAD | Local | Adaptive-RAG (flan-t5-base) | 26.80 | 38.70 | — | — | — | Local |
| Natural Questions | Local | Adaptive-RAG (flan-t5-base) | 37.20 | 46.70 | — | — | — | Local |
| TriviaQA | Local | Adaptive-RAG (flan-t5-base) | 51.00 | 59.50 | — | — | — | Local |
| MuSiQue | Local | Adaptive-RAG (flan-t5-base) | 20.20 | 29.80 | — | — | — | Local |
| HotpotQA | Local | Adaptive-RAG (flan-t5-base) | 42.80 | 53.70 | — | — | — | Local |
| 2WikiMultiHopQA | Local | Adaptive-RAG (flan-t5-base) | 39.40 | 48.30 | — | — | — | Local |
| SQuAD | Local | Adaptive-RAG (flan-t5-large) | 27.20 | 38.80 | — | — | — | Local |
| Natural Questions | Local | Adaptive-RAG (flan-t5-large) | 37.80 | 47.30 | — | — | — | Local |
| TriviaQA | Local | Adaptive-RAG (flan-t5-large) | 53.00 | 61.60 | — | — | — | Local |
| MuSiQue | Local | Adaptive-RAG (flan-t5-large) | 20.80 | 30.70 | — | — | — | Local |
| HotpotQA | Local | Adaptive-RAG (flan-t5-large) | 42.40 | 53.00 | — | — | — | Local |
| 2WikiMultiHopQA | Local | Adaptive-RAG (flan-t5-large) | 39.00 | 48.00 | — | — | — | Local |
| SQuAD | Local | Adaptive-RAG (flan-t5-small) | 27.60 | 39.30 | — | — | — | Local |
| Natural Questions | Local | Adaptive-RAG (flan-t5-small) | 37.40 | 46.90 | — | — | — | Local |
| TriviaQA | Local | Adaptive-RAG (flan-t5-small) | 53.60 | 62.30 | — | — | — | Local |
| MuSiQue | Local | Adaptive-RAG (flan-t5-small) | 20.20 | 29.80 | — | — | — | Local |
| HotpotQA | Local | Adaptive-RAG (flan-t5-small) | 41.80 | 52.60 | — | — | — | Local |
| 2WikiMultiHopQA | Local | Adaptive-RAG (flan-t5-small) | 39.80 | 48.60 | — | — | — | Local |
