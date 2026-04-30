import argparse

import json
from pathlib import Path
from typing import Any, List

from tqdm.auto import tqdm

from myproject.src.file_loader import load_records, load_yaml_config
from myproject.src.generate_responses.llm import LocalLLM
from myproject.src.generate_responses.pipeline import AdaptiveRAGPipeline
from myproject.src.build_index.retriever import FaissIVFRetriever
from myproject.src.generate_responses.streaming import MetricsAccumulator, StreamingPrettyWriter


def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _require_index_dir(index_dir: str) -> Path:
    p = Path(index_dir)
    if not (p / "documents.json").exists():
        raise FileNotFoundError(f"Missing index at {p}")
    return p


def _output_path_for_strategy(base: Path, strategy: str) -> Path:
    return base.with_name(f"{base.stem}-{strategy}.jsonl")


def _stats_path_for_strategy(base: Path, strategy: str) -> Path:
    return base.with_name(f"{base.stem}-{strategy}_stats.json")


def run_strategy_streaming(
    pipeline: AdaptiveRAGPipeline,
    strategy: str,
    records: List[Any],
    output_path: Path,
    stats_path: Path,
    single_k: int,
    multi_k: int,
    multi_steps: int,
    final_k_multi: int,
    batch_size: int = 8,
):
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    metrics = MetricsAccumulator()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with StreamingPrettyWriter(output_path) as pretty_writer:
        pbar = tqdm(total=len(records), desc=strategy, unit="query")

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            questions = [r.question for r in batch]

            if strategy == "no-rag":
                outputs = pipeline.no_retrieval(questions, start_query_id=start)
            elif strategy == "single":
                outputs = pipeline.single_step(questions, k=single_k, start_query_id=start)
            elif strategy == "multi":
                outputs = pipeline.multi_step(
                    questions,
                    steps=multi_steps,
                    k=multi_k,
                    final_k=final_k_multi,
                    start_query_id=start,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            for i, output in enumerate(outputs):
                record = batch[i]
                result = {
                    "id": record.id,
                    "question": record.question,
                    "gold": record.gold,
                    "prediction": output.prediction.strip(),
                    "context": output.context,
                    "strategy": strategy,
                    "retrieval_count": output.retrieval_count,
                    "llm_calls": output.llm_calls,
                    "latency_s": output.latency_s,
                }

                pretty_writer.write(result)

                metrics.record(
                    latency_s=output.latency_s,
                    retrieval_count=output.retrieval_count,
                    llm_call_count=output.llm_calls,
                )

                pbar.update(1)
                if len(metrics.latencies) >= 10:
                    avg_latency = sum(metrics.latencies[-10:]) / 10
                    pbar.set_postfix(
                        {
                            "latency": f"{avg_latency:.2f}s",
                            "retrievals": metrics.retrieval_counts[-1],
                            "llm_calls": metrics.llm_call_counts[-1],
                        }
                    )

    stats = metrics.to_dict()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved: {stats_path}")
    print(json.dumps(stats, indent=2))

    written_count = pretty_writer.get_count()
    assert written_count == len(records), (
        f"Output count mismatch: {written_count} written vs {len(records)} records"
    )


def run_generate_responses(
    config_path: str = "config.yaml",
    split: str = "train",
    strategy_override: str | None = None,
    limit: int | None = None,
) -> None:
    paths = load_yaml_config(config_path, section="paths")
    cfg = load_yaml_config(config_path, section="pipeline")

    if "llm_config" in cfg and "retriever_config" in cfg:
        llm_cfg = load_yaml_config(cfg["llm_config"], section="llm")
        retr_cfg = load_yaml_config(cfg["retriever_config"], section="retriever")
    else:
        llm_cfg = load_yaml_config(config_path, section="llm")
        retr_cfg = load_yaml_config(config_path, section="retriever")

    output_base = _as_path(paths["predictions_base"])
    index_dir = _require_index_dir(paths["index_dir"])
    data_key = "train_data" if split == "train" else "validation_data"
    records = load_records(paths[data_key])

    if limit is not None:
        records = records[: max(0, int(limit))]

    single_k = int(retr_cfg.get("top_k_single"))
    multi_k = int(retr_cfg.get("top_k_multi"))
    final_k_multi = int(retr_cfg.get("final_k_multi", multi_k))
    multi_steps = int((cfg.get("multi", {}) or {}).get("steps", 2))

    print(f"[inference] Loaded {len(records)} records")
    print(f"[inference] Output base: {output_base}")

    print("[inference] Initializing LLM...", flush=True)
    llm = LocalLLM(llm_cfg)

    print("[inference] Loading retriever index...", flush=True)
    retriever = FaissIVFRetriever(encoder_name=retr_cfg["encoder_name"])
    retriever.load(index_dir)

    pipeline = AdaptiveRAGPipeline(llm, retriever)

    strategy_cfg = strategy_override if strategy_override is not None else cfg.get("strategy", "multi")
    if strategy_cfg == "all":
        strategies = ["no-rag", "single", "multi"]
    elif isinstance(strategy_cfg, list):
        strategies = strategy_cfg
    else:
        strategies = [strategy_cfg]

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
            multi_steps=multi_steps,
            final_k_multi=final_k_multi,
            batch_size=batch_size,
        )

    print("\n" + "=" * 60)
    print("Streaming inference complete")
    print("=" * 60)



def main() -> None:
    parser = argparse.ArgumentParser(description="Stream adaptive RAG predictions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--strategy", default=None, choices=["no-rag", "single", "multi", "all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_generate_responses(
        config_path=args.config,
        split=args.split,
        strategy_override=args.strategy,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
