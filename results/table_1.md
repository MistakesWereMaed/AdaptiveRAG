# Table 1 — Average QA Results

| LLM | Type | Method | EM | F1 | Acc | Step | Time | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FLAN-T5-XL (3B) | Simple | No Retrieval | 14.87 | 21.12 | 15.97 | 0.00 | 0.11 | Paper |
| FLAN-T5-XL (3B) | Simple | Single-step Approach | 34.83 | 44.31 | 38.87 | 1.00 | 1.00 | Paper |
| FLAN-T5-XL (3B) | Adaptive | Adaptive Retrieval | 23.87 | 32.24 | 26.73 | 0.50 | 0.56 | Paper |
| FLAN-T5-XL (3B) | Adaptive | Self-RAG* | 9.90 | 20.79 | 31.57 | 0.72 | 0.43 | Paper |
| FLAN-T5-XL (3B) | Adaptive | Adaptive-RAG (Ours) | 37.17 | 46.94 | 42.10 | 2.17 | 3.60 | Paper |
| FLAN-T5-XL (3B) | Complex | Multi-step Approach | 39.00 | 48.85 | 43.70 | 4.69 | 8.81 | Paper |
| FLAN-T5-XL (3B) | Oracle | Adaptive-RAG w/ Oracle | 45.00 | 56.28 | 49.90 | 1.28 | 2.11 | Paper |
| FLAN-T5-XXL (11B) | Simple | No Retrieval | 17.83 | 25.14 | 19.33 | 0.00 | 0.08 | Paper |
| FLAN-T5-XXL (11B) | Simple | Single-step Approach | 37.87 | 47.63 | 41.90 | 1.00 | 1.00 | Paper |
| FLAN-T5-XXL (11B) | Adaptive | Adaptive Retrieval | 26.93 | 35.67 | 29.73 | 0.50 | 0.54 | Paper |
| FLAN-T5-XXL (11B) | Adaptive | Self-RAG* | 10.87 | 22.98 | 34.13 | 0.74 | 0.23 | Paper |
| FLAN-T5-XXL (11B) | Adaptive | Adaptive-RAG (Ours) | 38.90 | 48.62 | 43.77 | 1.35 | 2.00 | Paper |
| FLAN-T5-XXL (11B) | Complex | Multi-step Approach | 40.13 | 50.09 | 45.20 | 2.13 | 3.80 | Paper |
| FLAN-T5-XXL (11B) | Oracle | Adaptive-RAG w/ Oracle | 47.17 | 58.60 | 52.20 | 0.84 | 1.10 | Paper |
| GPT-3.5 (Turbo) | Simple | No Retrieval | 35.77 | 48.56 | 44.27 | 0.00 | 0.71 | Paper |
| GPT-3.5 (Turbo) | Simple | Single-step Approach | 34.73 | 46.99 | 45.27 | 1.00 | 1.00 | Paper |
| GPT-3.5 (Turbo) | Adaptive | Adaptive Retrieval | 35.90 | 48.20 | 45.30 | 0.50 | 0.86 | Paper |
| GPT-3.5 (Turbo) | Adaptive | Self-RAG* | 10.87 | 22.98 | 34.13 | 0.74 | 1.50 | Paper |
| GPT-3.5 (Turbo) | Adaptive | Adaptive-RAG (Ours) | 37.97 | 50.91 | 48.97 | 1.03 | 1.46 | Paper |
| GPT-3.5 (Turbo) | Complex | Multi-step Approach | 38.13 | 50.87 | 49.70 | 2.81 | 3.33 | Paper |
| GPT-3.5 (Turbo) | Oracle | Adaptive-RAG w/ Oracle | 47.70 | 62.80 | 58.57 | 0.50 | 1.03 | Paper |
| flan_t5_xl | Local baseline | No Retrieval | 14.67 | 20.50 | — | 0.00 | — | Local |
| flan_t5_xl | Local baseline | Single-step Approach | 34.23 | 43.58 | — | 1.00 | — | Local |
| flan_t5_xl | Local baseline | Multi-step Approach | 38.30 | 48.08 | — | 4.69 | — | Local |
| flan_t5_xl | Local router | Adaptive-RAG (flan-t5-base) | 36.23 | 46.12 | — | 2.30 | — | Local |
| flan_t5_xl | Local router | Adaptive-RAG (flan-t5-large) | 36.70 | 46.57 | — | 2.26 | — | Local |
| flan_t5_xl | Local router | Adaptive-RAG (flan-t5-small) | 36.73 | 46.58 | — | 2.33 | — | Local |
