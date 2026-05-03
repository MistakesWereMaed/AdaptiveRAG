from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional

from tqdm.auto import tqdm

from src.file_loader import load_records, load_yaml_config
from src.generate_responses.llm import LocalLLM
from src.generate_responses.pipeline import AdaptiveRAGPipeline
from src.generate_responses.streaming import MetricsAccumulator, StreamingJSONLWriter
from src.index.seach_pipeline import BM25ElasticsearchPipelineRetriever
from src.index.retriever import ElasticsearchRetriever


VALID_SPLITS = {"train", "validation", "eval", "test"}


def _data_path(paths: dict, split: str) -> str:
    candidate_keys = [f"{split}_data"]
    if split == "eval":
        candidate_keys.append("validation_data")
    if split == "validation":
        candidate_keys.append("eval_data")

    for key in candidate_keys:
        if key in paths:
            return paths[key]

    raise KeyError(f"Missing one of these path keys in config: {candidate_keys}")


def _prediction_base(paths: dict, split: str) -> Path:
    split_key = f"predictions_{split}_base"
    if split_key in paths:
        return Path(paths[split_key])

    base = Path(paths["predictions_base"])
    return base.with_name(f"{base.stem}-{split}{base.suffix}")


def _output_path(base: Path, strategy: str) -> Path:
    return base.with_name(f"{base.stem}-{strategy}.jsonl")


def _stats_path(base: Path, strategy: str) -> Path:
    return base.with_name(f"{base.stem}-{strategy}_stats.json")


def _load_configs(config_path: str):
    paths = load_yaml_config(config_path, section="paths")
    pipeline_cfg = load_yaml_config(config_path, section="pipeline")

    if "llm_config" in pipeline_cfg and "retriever_config" in pipeline_cfg:
        llm_cfg = load_yaml_config(pipeline_cfg["llm_config"], section="llm")
        retriever_cfg = load_yaml_config(pipeline_cfg["retriever_config"], section="retriever")
    else:
        llm_cfg = load_yaml_config(config_path, section="llm")
        retriever_cfg = load_yaml_config(config_path, section="retriever")

    return paths, pipeline_cfg, llm_cfg, retriever_cfg


def _selected_strategies(strategy_cfg) -> List[str]:
    if strategy_cfg == "all":
        return ["no-rag", "single", "multi"]
    if isinstance(strategy_cfg, list):
        return strategy_cfg
    return [strategy_cfg]


def _record_dataset(record: Any) -> Optional[str]:
    if hasattr(record, "dataset"):
        return getattr(record, "dataset")
    if isinstance(record, dict):
        return record.get("dataset")
    return None


def _record_id(record: Any) -> Any:
    if hasattr(record, "id"):
        return getattr(record, "id")
    if isinstance(record, dict):
        return record.get("id")
    return None


def _record_question(record: Any) -> str:
    if hasattr(record, "question"):
        return getattr(record, "question")
    return record["question"]


def _record_gold(record: Any) -> str:
    if hasattr(record, "gold"):
        return getattr(record, "gold")
    return record.get("gold") or record.get("answer") or ""


def _run_batch(
    pipeline: AdaptiveRAGPipeline,
    strategy: str,
    questions: List[str],
    datasets: List[Optional[str]],
    single_k: int,
    multi_k: int,
    multi_steps: int,
    final_k_multi: int,
):
    if strategy == "no-rag":
        return pipeline.no_retrieval(questions, datasets=datasets)

    if strategy == "single":
        return pipeline.single_step(questions, datasets=datasets, k=single_k)

    if strategy == "multi":
        return pipeline.multi_step(
            questions,
            datasets=datasets,
            steps=multi_steps,
            k=multi_k,
            final_k=final_k_multi,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def run_strategy(
    pipeline: AdaptiveRAGPipeline,
    strategy: str,
    records: List[Any],
    output_path: Path,
    stats_path: Path,
    batch_size: int,
    single_k: int,
    multi_k: int,
    multi_steps: int,
    final_k_multi: int,
) -> None:
    metrics = MetricsAccumulator()

    with StreamingJSONLWriter(output_path) as writer:
        for start in tqdm(range(0, len(records), batch_size), desc=strategy, unit="batch"):
            batch = records[start : start + batch_size]
            questions = [_record_question(record) for record in batch]
            datasets = [_record_dataset(record) for record in batch]

            outputs = _run_batch(
                pipeline,
                strategy,
                questions,
                datasets=datasets,
                single_k=single_k,
                multi_k=multi_k,
                multi_steps=multi_steps,
                final_k_multi=final_k_multi,
            )

            for record, dataset, output in zip(batch, datasets, outputs):
                writer.write(
                    {
                        "id": _record_id(record),
                        "dataset": dataset,
                        "question": _record_question(record),
                        "gold": _record_gold(record),
                        "prediction": output.prediction.strip(),
                        "context": output.context,
                        "retrieved_docs": pipeline.serialize_docs(output.retrieved_docs),
                        "strategy": strategy,
                        "retrieval_count": output.retrieval_count,
                        "llm_calls": output.llm_calls,
                        "latency_s": output.latency_s,
                    }
                )

                metrics.record(
                    latency_s=output.latency_s,
                    retrieval_count=output.retrieval_count,
                    llm_calls=output.llm_calls,
                )

    if writer.count != len(records):
        raise RuntimeError(f"Expected {len(records)} outputs, wrote {writer.count}")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print(f"[generate_responses] {strategy}: wrote {writer.count} records to {output_path}", flush=True)


def _build_retriever(retriever_cfg: dict) -> BM25ElasticsearchPipelineRetriever:
    es = ElasticsearchRetriever(
        host=str(retriever_cfg.get("host", "localhost")),
        port=int(retriever_cfg.get("port", 9200)),
    )

    return BM25ElasticsearchPipelineRetriever(
        es_retriever=es,
        dataset_to_index=retriever_cfg.get("dataset_to_index"),
        default_index=str(retriever_cfg.get("default_index", "wiki")),
        query_title_field_too=bool(retriever_cfg.get("query_title_field_too", True)),
        max_buffer_count=int(retriever_cfg.get("max_buffer_count", 100)),
    )


def run_generate_responses(
    config_path: str = "config.yaml",
    split: str = "train",
    strategy_override: str | None = None,
    limit: int | None = None,
) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")

    paths, pipeline_cfg, llm_cfg, retriever_cfg = _load_configs(config_path)

    records = load_records(_data_path(paths, split))
    if limit is not None:
        records = records[: max(0, int(limit))]

    output_base = _prediction_base(paths, split)

    single_k = int(retriever_cfg.get("top_k_single", 6))
    multi_k = int(retriever_cfg.get("top_k_multi", 6))
    final_k_multi = int(retriever_cfg.get("final_k_multi", multi_k))
    multi_steps = int((pipeline_cfg.get("multi", {}) or {}).get("steps", 2))
    batch_size = int(pipeline_cfg.get("pipeline_batch_size", 8))
    max_chars_per_doc = int(pipeline_cfg.get("max_chars_per_doc", 900))

    strategy_cfg = strategy_override or pipeline_cfg.get("strategy", "all")
    strategies = _selected_strategies(strategy_cfg)

    print(
        f"[generate_responses] split={split} records={len(records)} "
        f"strategies={strategies} output_base={output_base}",
        flush=True,
    )

    llm = LocalLLM(llm_cfg)
    retriever = _build_retriever(retriever_cfg)
    pipeline = AdaptiveRAGPipeline(llm, retriever, max_chars_per_doc=max_chars_per_doc)

    for strategy in strategies:
        run_strategy(
            pipeline=pipeline,
            strategy=strategy,
            records=records,
            output_path=_output_path(output_base, strategy),
            stats_path=_stats_path(output_base, strategy),
            batch_size=batch_size,
            single_k=single_k,
            multi_k=multi_k,
            multi_steps=multi_steps,
            final_k_multi=final_k_multi,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate no-rag, single, and multi RAG responses")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="train", choices=sorted(VALID_SPLITS))
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
