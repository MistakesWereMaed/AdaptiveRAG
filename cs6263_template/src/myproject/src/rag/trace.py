"""Execution trace collection for per-query instrumentation."""

import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ExecutionTrace:
    """Lightweight trace object to track per-query execution metrics.
    
    Fields track cumulative counts and timing across a single query's
    full pipeline execution (including all retrieval and generation steps).
    """
    
    # Identifiers
    query_id: int
    question: str
    
    # Execution counts (cumulative for multi-step)
    retrieval_count: int = 0
    llm_call_count: int = 0
    
    # Timing (seconds)
    start_time: float = 0.0
    end_time: Optional[float] = None
    
    def __post_init__(self):
        """Initialize start_time if not set."""
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    def record_retrieval(self, num_calls: int = 1):
        """Record one or more retrieval calls."""
        self.retrieval_count += num_calls
    
    def record_llm_call(self, num_calls: int = 1):
        """Record one or more LLM generation calls."""
        self.llm_call_count += num_calls
    
    def finalize(self) -> float:
        """Mark trace as complete and return latency in seconds."""
        if self.end_time is None:
            self.end_time = time.time()
        return self.latency_s
    
    @property
    def latency_s(self) -> float:
        """Get total latency in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "query_id": self.query_id,
            "question": self.question,
            "retrieval_count": self.retrieval_count,
            "llm_call_count": self.llm_call_count,
            "latency_s": self.latency_s,
        }
