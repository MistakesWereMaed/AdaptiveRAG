from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from tqdm.auto import tqdm

from src.build_index.retriever import FaissIVFRetriever
from src.file_loader import load_records, load_yaml_config
from src.generate_responses.llm import LocalLLM
from src.generate_responses.pipeline import AdaptiveRAGPipeline
from src.generate_responses.streaming import MetricsAccumulator, StreamingJSONLWriter


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


def _run_batch(
    pipeline: AdaptiveRAGPipeline,
    strategy: str,
    questions: List[str],
    single_k: int,
    multi_k: int,
    multi_steps: int,
    final_k_multi: int,
):
    if strategy == "no-rag":
        return pipeline.no_retrieval(questions)

    if strategy == "single":
        return pipeline.single_step(questions, k=single_k)

    if strategy == "multi":
        return pipeline.multi_step(
            questions,
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
            questions = [record.question for record in batch]

            outputs = _run_batch(
                pipeline,
                strategy,
                questions,
                single_k=single_k,
                multi_k=multi_k,
                multi_steps=multi_steps,
                final_k_multi=final_k_multi,
            )

            for record, output in zip(batch, outputs):
                writer.write(
                    {
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


def run_generate_responses(config_path: str = "config.yaml") -> None:
    paths, pipeline_cfg, llm_cfg, retriever_cfg = _load_configs(config_path)

    split = str(pipeline_cfg.get("split", "train"))
    limit = pipeline_cfg.get("limit")
    data_path = paths["train_data"] if split == "train" else paths["validation_data"]
    records = load_records(data_path)

    if limit is not None:
        records = records[: max(0, int(limit))]

    output_base = Path(paths["predictions_base"])
    index_dir = Path(paths["index_dir"])
    if not (index_dir / "index.faiss").exists() or not (index_dir / "documents.json").exists():
        raise FileNotFoundError(f"Missing FAISS index files in {index_dir}")

    single_k = int(retriever_cfg.get("top_k_single", 6))
    multi_k = int(retriever_cfg.get("top_k_multi", 6))
    final_k_multi = int(retriever_cfg.get("final_k_multi", multi_k))
    multi_steps = int((pipeline_cfg.get("multi", {}) or {}).get("steps", 2))
    batch_size = int(pipeline_cfg.get("pipeline_batch_size", 8))

    strategy_cfg = pipeline_cfg.get("strategy", "all")
    strategies = _selected_strategies(strategy_cfg)

    print(f"[generate_responses] records={len(records)} strategies={strategies}", flush=True)

    llm = LocalLLM(llm_cfg)
    retriever = FaissIVFRetriever(
        encoder_name=retriever_cfg["encoder_name"],
        nprobe=int(retriever_cfg.get("nprobe", 8)),
    ).load(index_dir)

    pipeline = AdaptiveRAGPipeline(llm, retriever)

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
    run_generate_responses("config.yaml")


if __name__ == "__main__":
    main()
