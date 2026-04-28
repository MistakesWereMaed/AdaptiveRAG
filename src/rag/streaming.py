"""Streaming JSONL writer for per-query result output."""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class StreamingJSONLWriter:
    """Write results to JSONL format incrementally, one per line.
    
    No in-memory aggregation: each result is written and flushed immediately.
    Safe for interruption - partial results remain on disk.
    """
    
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.count = 0
    
    def __enter__(self):
        self.file = self.output_path.open("w", encoding="utf-8")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def write(self, record: Dict[str, Any]):
        """Write a single record as a JSONL line.
        
        Each record is a complete JSON object on its own line.
        Automatically flushes to disk after each write.
        """
        if self.file is None:
            raise RuntimeError("Must use context manager: with StreamingJSONLWriter(...) as w:")
        
        # Serialize to JSON and write as single line
        line = json.dumps(record, ensure_ascii=False)
        self.file.write(line + "\n")
        self.file.flush()  # Ensure immediate disk write
        self.count += 1
    
    def get_count(self) -> int:
        """Return number of records written so far."""
        return self.count


class MetricsAccumulator:
    """Accumulate per-query metrics for post-hoc aggregation.
    
    Computes statistics from individual trace objects without
    requiring in-memory storage of full results.
    """
    
    def __init__(self):
        self.latencies = []
        self.retrieval_counts = []
        self.llm_call_counts = []
        self.count = 0
    
    def record(self, latency_s: float, retrieval_count: int, llm_call_count: int):
        """Record a single query's metrics."""
        self.latencies.append(latency_s)
        self.retrieval_counts.append(retrieval_count)
        self.llm_call_counts.append(llm_call_count)
        self.count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Compute aggregated stats."""
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
