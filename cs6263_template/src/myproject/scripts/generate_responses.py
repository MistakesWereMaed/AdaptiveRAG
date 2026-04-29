#!/usr/bin/env python
"""Streaming inference for adaptive RAG with execution tracing.

Outputs:
    - data/predictions/predictions-<strategy>.jsonl - streaming predictions with trace info
    - data/predictions/predictions-<strategy>_stats.json - aggregated execution metrics per strategy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Any

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs6263_template.src.myproject.src.data.file_loader import load_records, load_yaml_config
from cs6263_template.src.myproject.src.rag.llm import LocalLLM
from cs6263_template.src.myproject.src.rag.pipeline import AdaptiveRAGPipeline
from cs6263_template.src.myproject.src.rag.retriever import FaissIVFRetriever
from cs6263_template.src.myproject.src.rag.streaming import StreamingJSONLWriter, MetricsAccumulator, StreamingPrettyWriter


# ============================================================
# Helpers
# ============================================================
def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _require_index_dir(index_dir: str) -> Path:
    p = Path(index_dir)
    if not (p / "documents.json").exists():
        raise FileNotFoundError(f"Missing index at {p}")
    return p


def _output_path_for_strategy(base: Path, strategy: str) -> Path:
    """Generate output path: predictions-<strategy>.jsonl in the shared predictions directory."""
    return base.with_name(f"{base.stem}-{strategy}.jsonl")


def _stats_path_for_strategy(base: Path, strategy: str) -> Path:
    """Generate stats path: predictions-<strategy>_stats.json in the shared predictions directory."""
    return base.with_name(f"{base.stem}-{strategy}_stats.json")


# ============================================================
# Streaming execution
# ============================================================
def run_strategy_streaming(
    pipeline: AdaptiveRAGPipeline,
    strategy: str,
    records: List[Any],
    output_path: Path,
    stats_path: Path,
    single_k: int,
    multi_k: int,
    batch_size: int = 8,
):
    """Execute strategy on all records, streaming results to JSONL.
    
    For each query:
    1. Call appropriate pipeline method (with tracing)
    2. Write result + trace to JSONL immediately
    3. Accumulate metrics
    
    This maintains batch inference semantics (LLM/retriever still batch)
    while streaming results incrementally to disk.
    """
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    metrics = MetricsAccumulator()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write both compact JSONL (tooling) and a pretty JSON array for humans
    with StreamingJSONLWriter(output_path) as writer, StreamingPrettyWriter(output_path) as pretty_writer:
        pbar = tqdm(total=len(records), desc=strategy, unit="query")

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            questions = [r.question for r in batch]

            # Execute batched strategy with tracing
            if strategy == "no-rag":
                paired = pipeline.no_retrieval(questions, return_traces=True, start_query_id=start)
            elif strategy == "single":
                paired = pipeline.single_step(
                    questions,
                    k=single_k,
                    return_traces=True,
                    start_query_id=start,
                )
            elif strategy == "multi":
                paired = pipeline.multi_step(
                    questions,
                    k=multi_k,
                    return_traces=True,
                    start_query_id=start,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # paired: list of (prediction, trace) corresponding to batch order
            for i, item in enumerate(paired):
                # pipeline returns (prediction, trace, context)
                if len(item) == 3:
                    prediction, trace, context = item
                else:
                    prediction, trace = item
                    context = ""

                record = batch[i]

                result = {
                    "id": record.id,
                    "question": record.question,
                    "gold": record.gold,
                    "prediction": prediction.strip(),
                    "context": context,
                    "strategy": strategy,
                    "retrieval_count": trace.retrieval_count,
                    "llm_calls": trace.llm_call_count,
                    "latency_s": trace.latency_s,
                }

                # Stream compact line for tooling
                writer.write(result)
                # Stream pretty human-readable JSON array
                pretty_writer.write(result)

                # Accumulate metrics
                metrics.record(
                    latency_s=trace.latency_s,
                    retrieval_count=trace.retrieval_count,
                    llm_call_count=trace.llm_call_count,
                )

                # Advance progress bar and show running metrics
                pbar.update(1)
                if len(metrics.latencies) >= 10:
                    avg_latency = sum(metrics.latencies[-10:]) / 10
                    pbar.set_postfix({
                        "latency": f"{avg_latency:.2f}s",
                        "retrievals": metrics.retrieval_counts[-1],
                        "llm_calls": metrics.llm_call_counts[-1],
                    })

    # Write aggregated stats
    stats = metrics.to_dict()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved: {stats_path}")
    print(json.dumps(stats, indent=2))

    # Verify output count
    written_count = writer.get_count()
    assert written_count == len(records), (
        f"Output count mismatch: {written_count} written vs {len(records)} records"
    )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stream adaptive RAG predictions with execution traces"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Load config
    paths = load_yaml_config(args.config, section="paths")
    cfg = load_yaml_config(args.config, section="pipeline")
    if "llm_config" in cfg and "retriever_config" in cfg:
        llm_cfg = load_yaml_config(cfg["llm_config"], section="llm")
        retr_cfg = load_yaml_config(cfg["retriever_config"], section="retriever")
    else:
        llm_cfg = load_yaml_config(args.config, section="llm")
        retr_cfg = load_yaml_config(args.config, section="retriever")

    # Load data
    output_base = _as_path(paths["predictions_base"])
    index_dir = _require_index_dir(paths["index_dir"])
    records = load_records(paths["train_data"])
    single_k = int(retr_cfg.get("top_k_single"))
    multi_k = int(retr_cfg.get("top_k_multi"))

    print(f"[inference] Loaded {len(records)} records")
    print(f"[inference] Output base: {output_base}")

    # Initialize components
    print("[inference] Initializing LLM...")
    llm = LocalLLM(llm_cfg)

    print("[inference] Loading retriever index...")
    retriever = FaissIVFRetriever(encoder_name=retr_cfg["encoder_name"])
    retriever.load(index_dir)

    pipeline = AdaptiveRAGPipeline(llm, retriever)

    # Determine strategies to run
    strategy_cfg = cfg.get("strategy", "multi")
    if strategy_cfg == "all":
        strategies = ["no-rag", "single", "multi"]
    elif isinstance(strategy_cfg, list):
        strategies = strategy_cfg
    else:
        strategies = [strategy_cfg]

    # Execute each strategy with streaming output
    for strategy in strategies:
        output_path = _output_path_for_strategy(output_base, strategy)
        stats_path = _stats_path_for_strategy(output_base, strategy)

        batch_size = int(cfg.get("pipeline_batch_size"))

        run_strategy_streaming(
            pipeline,
            strategy,
            records,
            output_path,
            stats_path,
            single_k=single_k,
            multi_k=multi_k,
            batch_size=batch_size,
        )

    print("\n" + "="*60)
    print("Streaming inference complete")
    print("="*60)


if __name__ == "__main__":
    main()