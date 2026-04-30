"""Streaming JSONL writer for per-query result output."""

import json
from pathlib import Path
from typing import Dict, Any


class MetricsAccumulator:
    def __init__(self):
        self.latencies = []
        self.retrieval_counts = []
        self.llm_call_counts = []
        self.count = 0

    def record(self, latency_s: float, retrieval_count: int, llm_call_count: int):
        self.latencies.append(latency_s)
        self.retrieval_counts.append(retrieval_count)
        self.llm_call_counts.append(llm_call_count)
        self.count += 1

    def to_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "num_examples": 0,
                "avg_latency_s": 0.0,
                "total_latency_s": 0.0,
                "total_retrievals": 0,
                "total_llm_calls": 0,
            }

        return {
            "num_examples": self.count,
            "avg_latency_s": sum(self.latencies) / self.count,
            "total_latency_s": sum(self.latencies),
            "total_retrievals": sum(self.retrieval_counts),
            "total_llm_calls": sum(self.llm_call_counts),
        }


class StreamingPrettyWriter:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.file = None
        self.first = True

    def __enter__(self):
        pretty_path = self.output_path.with_suffix(".json")
        pretty_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = pretty_path.open("w", encoding="utf-8")
        self.file.write("[\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.write("\n]\n")
            self.file.close()

    def write(self, record: Dict[str, Any]):
        if self.file is None:
            raise RuntimeError("Must use context manager: with StreamingPrettyWriter(...) as w:")

        if not self.first:
            self.file.write(",\n")
        self.first = False

        pretty = json.dumps(record, ensure_ascii=False, indent=2)
        indented = "\n".join("  " + line for line in pretty.splitlines())
        self.file.write(indented)
        self.file.flush()

    def get_count(self) -> int:
        # best-effort: count lines in pretty file (not strictly necessary)
        return self.first == False and 0 or 0
