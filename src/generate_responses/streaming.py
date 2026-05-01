from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class MetricsAccumulator:
    def __init__(self):
        self.count = 0
        self.total_latency_s = 0.0
        self.total_retrievals = 0
        self.total_llm_calls = 0

    def record(self, latency_s: float, retrieval_count: int, llm_calls: int) -> None:
        self.count += 1
        self.total_latency_s += float(latency_s)
        self.total_retrievals += int(retrieval_count)
        self.total_llm_calls += int(llm_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_examples": self.count,
            "avg_latency_s": self.total_latency_s / self.count if self.count else 0.0,
            "total_latency_s": self.total_latency_s,
            "total_retrievals": self.total_retrievals,
            "total_llm_calls": self.total_llm_calls,
        }


class StreamingJSONLWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = None
        self.count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write(self, record: Dict[str, Any]) -> None:
        if self.file is None:
            raise RuntimeError("Writer must be used as a context manager")

        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
