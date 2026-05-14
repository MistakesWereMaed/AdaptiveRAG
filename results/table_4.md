# Table 4 — Training-Data Ablation

| Training Strategy | QA F1 | Step | Cls. All | Cls. No | Cls. One | Cls. Multi | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Adaptive-RAG (Ours) | 46.94 | 1084 | 54.52 | 30.52 | 66.28 | 65.45 | Paper |
| w/o Binary | 43.43 | 640 | 60.30 | 62.19 | 65.70 | 39.55 | Paper |
| w/o Silver | 48.79 | 1464 | 40.00 | 0.00 | 53.98 | 75.91 | Paper |
| Adaptive-RAG (flan-t5-base) | 46.12 | 2 | — | — | — | — | Local |
| Adaptive-RAG (flan-t5-large) | 46.57 | 2 | — | — | — | — | Local |
| Adaptive-RAG (flan-t5-small) | 46.58 | 2 | — | — | — | — | Local |
