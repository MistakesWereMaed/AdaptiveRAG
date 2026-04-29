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

from src.data.file_loader import load_records, load_yaml_config, normalize_text
from src.rag.llm import LocalLLM
from src.rag.pipeline import AdaptiveRAGPipeline
from src.rag.retriever import FaissIVFRetriever
from src.rag.streaming import StreamingJSONLWriter, MetricsAccumulator, StreamingPrettyWriter


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


def _extract_support_titles(record: Any) -> List[str]:
    metadata = getattr(record, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return []
    maybe_titles = metadata.get("supporting_titles") or metadata.get("support_titles") or []
    if not isinstance(maybe_titles, list):
        return []
    return [str(title).strip() for title in maybe_titles if str(title).strip()]


def _is_unknown_answer(prediction: str) -> bool:
    value = str(prediction or "").strip().lower()
    return value in {"", "unknown"}


def _compute_retrieval_debug(gold: str, support_titles: List[str], retrieved_docs: List[Any]) -> dict:
    normalized_gold = normalize_text(gold)
    top_text = normalize_text(" ".join(str(doc.text) for doc in retrieved_docs))
    retrieved_titles = {str(doc.title).strip().lower() for doc in retrieved_docs if str(doc.title).strip()}
    normalized_support_titles = {title.strip().lower() for title in support_titles if title.strip()}

    retrieval_answer_hit = bool(normalized_gold and normalized_gold in top_text)
    support_hits = sum(1 for title in normalized_support_titles if title in retrieved_titles)
    retrieval_support_hit = support_hits > 0
    retrieval_full_support_hit = bool(normalized_support_titles) and support_hits == len(normalized_support_titles)
    oracle_context_answerable = retrieval_answer_hit or retrieval_support_hit

    return {
        "retrieval_answer_hit": retrieval_answer_hit,
        "retrieval_support_hit": retrieval_support_hit,
        "retrieval_full_support_hit": retrieval_full_support_hit,
        "oracle_context_answerable": oracle_context_answerable,
    }


def _classify_failure(prediction: str, gold: str, retrieval_debug: dict) -> str:
    prediction_is_unknown = _is_unknown_answer(prediction)
    answer_hit = bool(retrieval_debug.get("retrieval_answer_hit"))
    support_hit = bool(retrieval_debug.get("retrieval_support_hit"))
    oracle_answerable = bool(retrieval_debug.get("oracle_context_answerable"))
    prediction_correct = normalize_text(prediction) == normalize_text(gold)

    if prediction_is_unknown and oracle_answerable:
        return "generation_false_unknown"
    if prediction_is_unknown and not oracle_answerable:
        return "retrieval_missing_evidence"
    if (not prediction_correct) and answer_hit:
        return "wrong_answer_with_answer_present"
    if (not prediction_correct) and support_hit:
        return "wrong_answer_with_support_present"
    if prediction_correct:
        return "correct"
    return "other"


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
    log_prompts: bool = False,
    max_logged_prompts: int = 100,
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
    unknown_count = 0
    oracle_answerable_count = 0
    unknown_when_oracle_answerable_count = 0
    support_hit_count = 0
    unknown_when_support_hit_count = 0
    answer_hit_count = 0
    unknown_when_answer_hit_count = 0
    failure_type_counts = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with StreamingJSONLWriter(output_path) as jsonl_writer, StreamingPrettyWriter(output_path) as pretty_writer:
        pbar = tqdm(total=len(records), desc=strategy, unit="query")

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            questions = [r.question for r in batch]

            # Execute batched strategy with tracing
            if strategy == "no-rag":
                paired = pipeline.no_retrieval(
                    questions,
                    return_traces=True,
                    return_debug=True,
                    start_query_id=start,
                )
            elif strategy == "single":
                paired = pipeline.single_step(
                    questions,
                    k=single_k,
                    return_traces=True,
                    return_debug=True,
                    start_query_id=start,
                )
            elif strategy == "multi":
                paired = pipeline.multi_step(
                    questions,
                    k=multi_k,
                    return_traces=True,
                    return_debug=True,
                    start_query_id=start,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # paired: list of (prediction, trace) corresponding to batch order
            for i, item in enumerate(paired):
                # pipeline returns (prediction, trace, context, retrieved_docs, debug)
                if len(item) >= 5:
                    prediction, trace, context, retrieved_docs, debug_row = item[:5]
                elif len(item) == 3:
                    prediction, trace, context = item
                    retrieved_docs = []
                    debug_row = {}
                else:
                    prediction, trace = item
                    context = ""
                    retrieved_docs = []
                    debug_row = {}

                record = batch[i]
                support_titles = _extract_support_titles(record)
                retrieval_debug = _compute_retrieval_debug(record.gold, support_titles, retrieved_docs)
                prediction_is_unknown = _is_unknown_answer(prediction)
                failure_type = _classify_failure(prediction, record.gold, retrieval_debug)

                if prediction_is_unknown:
                    unknown_count += 1
                if retrieval_debug["oracle_context_answerable"]:
                    oracle_answerable_count += 1
                    if prediction_is_unknown:
                        unknown_when_oracle_answerable_count += 1
                if retrieval_debug["retrieval_support_hit"]:
                    support_hit_count += 1
                    if prediction_is_unknown:
                        unknown_when_support_hit_count += 1
                if retrieval_debug["retrieval_answer_hit"]:
                    answer_hit_count += 1
                    if prediction_is_unknown:
                        unknown_when_answer_hit_count += 1
                failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1

                result = {
                    "id": record.id,
                    "question": record.question,
                    "gold": record.gold,
                    "prediction": prediction.strip(),
                    "raw_generation": debug_row.get("raw_generation", ""),
                    "cleaned_generation": debug_row.get("cleaned_generation", prediction.strip()),
                    "final_generation": debug_row.get("final_generation", prediction.strip()),
                    "context": context,
                    "strategy": strategy,
                    "retrieval_count": trace.retrieval_count,
                    "llm_calls": trace.llm_call_count,
                    "latency_s": trace.latency_s,
                    "support_titles": support_titles,
                    "prediction_is_unknown": prediction_is_unknown,
                    "failure_type": failure_type,
                    **retrieval_debug,
                }

                if log_prompts and metrics.count < max_logged_prompts:
                    result["prompt"] = debug_row.get("prompt", "")

                jsonl_writer.write(result)
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
    num_examples = max(1, metrics.count)
    stats.update(
        {
            "unknown_rate": unknown_count / num_examples,
            "oracle_context_answerable_count": oracle_answerable_count,
            "unknown_when_oracle_context_answerable": (
                unknown_when_oracle_answerable_count / max(1, oracle_answerable_count)
            ),
            "support_hit_count": support_hit_count,
            "unknown_when_support_hit": unknown_when_support_hit_count / max(1, support_hit_count),
            "answer_hit_count": answer_hit_count,
            "unknown_when_answer_hit": unknown_when_answer_hit_count / max(1, answer_hit_count),
            "failure_type_counts": failure_type_counts,
        }
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved: {stats_path}")
    print(json.dumps(stats, indent=2))

    # Verify output count
    written_count = jsonl_writer.get_count()
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
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--strategy", default=None, choices=["no-rag", "single", "multi", "all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument("--max-logged-prompts", type=int, default=None)
    args = parser.parse_args()

    # Load config
    root_cfg = load_yaml_config(args.config)
    paths = load_yaml_config(args.config, section="paths")
    cfg = load_yaml_config(args.config, section="pipeline")
    if "llm_config" in cfg and "retriever_config" in cfg:
        llm_cfg = load_yaml_config(cfg["llm_config"], section="llm")
        retr_cfg = load_yaml_config(cfg["retriever_config"], section="retriever")
    else:
        llm_cfg = load_yaml_config(args.config, section="llm")
        retr_cfg = load_yaml_config(args.config, section="retriever")
    debug_cfg = root_cfg.get("debug", {}) if isinstance(root_cfg, dict) else {}

    # Load data
    output_base = _as_path(paths["predictions_base"])
    index_dir = _require_index_dir(paths["index_dir"])
    data_key = "train_data" if args.split == "train" else "validation_data"
    records = load_records(paths[data_key])
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]
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
    strategy_cfg = args.strategy if args.strategy is not None else cfg.get("strategy", "multi")
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
        log_prompts = bool(args.debug_prompts or debug_cfg.get("log_prompts", False))
        max_logged_prompts = int(
            args.max_logged_prompts if args.max_logged_prompts is not None else debug_cfg.get("max_logged_prompts", 100)
        )

        run_strategy_streaming(
            pipeline,
            strategy,
            records,
            output_path,
            stats_path,
            single_k=single_k,
            multi_k=multi_k,
            batch_size=batch_size,
            log_prompts=log_prompts,
            max_logged_prompts=max_logged_prompts,
        )

    print("\n" + "="*60)
    print("Streaming inference complete")
    print("="*60)


if __name__ == "__main__":
    main()