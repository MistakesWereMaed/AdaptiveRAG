from __future__ import annotations

import argparse
import json

from collections import Counter, defaultdict
from typing import Any, List
from tqdm.auto import tqdm

from src.build_index.retriever import FaissIVFRetriever
from src.evaluate_router.router import RouterPredictor
from src.file_loader import load_records
from src.generate_responses.llm import LocalLLM
from src.generate_responses.pipeline import AdaptiveRAGPipeline, PipelineOutput
from src.evaluate_router.eval import (
    _data_path,
    _adaptive_output_path,
    _adaptive_stats_path,
    _load_configs,
    _find_checkpoint,
    _score_predictions,
    _load_full_strategy_scores,
    _oracle_score_from_full_predictions,
    _load_full_strategy_efficiency,
)


VALID_SPLITS = {"train", "validation"}
STRATEGIES = ["no-rag", "single", "multi"]


def _run_group(
    pipeline: AdaptiveRAGPipeline,
    strategy: str,
    records: List[Any],
    batch_size: int,
    single_k: int,
    multi_k: int,
    multi_steps: int,
    final_k_multi: int,
) -> List[PipelineOutput]:
    outputs: List[PipelineOutput] = []

    for start in tqdm(range(0, len(records), batch_size), desc=f"adaptive {strategy}", unit="batch"):
        batch = records[start : start + batch_size]
        questions = [record.question for record in batch]

        if strategy == "no-rag":
            outputs.extend(pipeline.no_retrieval(questions))
        elif strategy == "single":
            outputs.extend(pipeline.single_step(questions, k=single_k))
        elif strategy == "multi":
            outputs.extend(pipeline.multi_step(questions, steps=multi_steps, k=multi_k, final_k=final_k_multi))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return outputs


def run_router_evaluation(
    config_path: str = "config.yaml",
    split: str = "validation",
    checkpoint_path: str | None = None,
    limit: int | None = None,
) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")

    paths, pipeline_cfg, llm_cfg, retriever_cfg, train_cfg, model_cfg = _load_configs(config_path)

    records = load_records(_data_path(paths, split))
    if limit is not None:
        records = records[: max(0, int(limit))]

    checkpoint_path = _find_checkpoint(paths, train_cfg, explicit=checkpoint_path)
    model_name = str(model_cfg.get("model_name", train_cfg.get("model_name", "microsoft/deberta-v3-base")))
    max_length = int(model_cfg.get("max_length", train_cfg.get("max_length", 128)))
    router_batch_size = int(model_cfg.get("eval_batch_size", model_cfg.get("batch_size", 64)))

    print(f"[router_eval] split={split} records={len(records)}", flush=True)
    print(f"[router_eval] checkpoint={checkpoint_path}", flush=True)

    router = RouterPredictor(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        max_length=max_length,
        num_classes=int(model_cfg.get("num_classes", 3)),
    )

    router_rows = router.predict([record.question for record in records], batch_size=router_batch_size)

    grouped_indices = defaultdict(list)
    for i, row in enumerate(router_rows):
        grouped_indices[row["strategy"]].append(i)

    print("[router_eval] predicted label distribution:")
    for strategy in STRATEGIES:
        print(f"  {strategy:8s}: {len(grouped_indices[strategy])}")

    single_k = int(retriever_cfg.get("top_k_single", 6))
    multi_k = int(retriever_cfg.get("top_k_multi", 6))
    final_k_multi = int(retriever_cfg.get("final_k_multi", multi_k))
    multi_steps = int((pipeline_cfg.get("multi", {}) or {}).get("steps", 2))
    batch_size = int(pipeline_cfg.get("pipeline_batch_size", 8))

    llm = LocalLLM(llm_cfg)
    retriever = FaissIVFRetriever(
        encoder_name=retriever_cfg["encoder_name"],
        nprobe=int(retriever_cfg.get("nprobe", 8)),
    ).load(paths["index_dir"])
    pipeline = AdaptiveRAGPipeline(llm, retriever)

    adaptive_rows = [None for _ in records]

    for strategy in STRATEGIES:
        indices = grouped_indices[strategy]
        if not indices:
            continue

        group_records = [records[i] for i in indices]
        outputs = _run_group(
            pipeline=pipeline,
            strategy=strategy,
            records=group_records,
            batch_size=batch_size,
            single_k=single_k,
            multi_k=multi_k,
            multi_steps=multi_steps,
            final_k_multi=final_k_multi,
        )

        for idx, output in zip(indices, outputs):
            route = router_rows[idx]
            record = records[idx]
            adaptive_rows[idx] = {
                "id": record.id,
                "question": record.question,
                "gold": record.gold,
                "prediction": output.prediction,
                "strategy": strategy,
                "router_label": route["label"],
                "router_confidence": route["confidence"],
                "router_probabilities": route["probabilities"],
                "context": output.context,
                "retrieval_count": output.retrieval_count,
                "llm_calls": output.llm_calls,
                "latency_s": output.latency_s,
            }

    if any(row is None for row in adaptive_rows):
        raise RuntimeError("Missing adaptive outputs")

    adaptive_scores = _score_predictions([row["prediction"] for row in adaptive_rows], records)
    full_scores = _load_full_strategy_scores(paths, split, records)
    oracle_scores = _oracle_score_from_full_predictions(paths, split, records)
    full_efficiency = _load_full_strategy_efficiency(paths, split)

    strategy_counts = Counter(row["strategy"] for row in adaptive_rows)
    total_retrievals = sum(int(row["retrieval_count"]) for row in adaptive_rows)
    total_llm_calls = sum(int(row["llm_calls"]) for row in adaptive_rows)
    total_latency = sum(float(row["latency_s"]) for row in adaptive_rows)

    stats = {
        "split": split,
        "checkpoint": checkpoint_path,
        "num_examples": len(records),

        "router_distribution": {
            strategy: {
                "count": strategy_counts[strategy],
                "pct": strategy_counts[strategy] / len(records) if records else 0.0,
            }
            for strategy in STRATEGIES
        },

        # -------------------------
        # Accuracy comparison
        # -------------------------
        "adaptive": adaptive_scores,
        "full_strategy_baselines": full_scores,
        "oracle_from_full_generation": oracle_scores,

        # -------------------------
        # Efficiency comparison
        # -------------------------
        "efficiency": {
            "adaptive_routed": {
                "total_retrievals": total_retrievals,
                "total_llm_calls": total_llm_calls,
                "total_latency_s": total_latency,
                "avg_latency_s": total_latency / len(records) if records else 0.0,
            },
            "full_generation": full_efficiency,
        },

        # -------------------------
        # Derived comparison metrics
        # -------------------------
        "comparison": {
            "speedup_vs_full": (
                full_efficiency["all_full_generation"]["total_latency_s"]
                / max(total_latency, 1e-9)
            ),
            "llm_call_reduction": (
                1.0
                - total_llm_calls
                / max(full_efficiency["all_full_generation"]["total_llm_calls"], 1)
            ),
            "retrieval_reduction": (
                1.0
                - total_retrievals
                / max(full_efficiency["all_full_generation"]["total_retrievals"], 1)
            ),
        },
    }

    output_path = _adaptive_output_path(paths, split)
    stats_path = _adaptive_stats_path(paths, split)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in adaptive_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Efficiency: Adaptive Routed ===")
    adaptive_eff = stats["efficiency"]["adaptive_routed"]
    print(f"Total retrievals: {adaptive_eff['total_retrievals']}")
    print(f"Total LLM calls:  {adaptive_eff['total_llm_calls']}")
    print(f"Total latency:    {adaptive_eff['total_latency_s']:.2f}s")
    print(f"Avg latency:      {adaptive_eff['avg_latency_s']:.4f}s")

    print("\n=== Efficiency: Full Generation Baselines ===")
    for strategy in STRATEGIES:
        eff = full_efficiency[strategy]
        missing = " [missing stats]" if eff.get("stats_file_missing") else ""
        print(
            f"{strategy:8s} | "
            f"retrievals: {eff['total_retrievals']} | "
            f"llm calls: {eff['total_llm_calls']} | "
            f"total latency: {eff['total_latency_s']:.2f}s | "
            f"avg latency: {eff['avg_latency_s']:.4f}s"
            f"{missing}"
        )

    total_eff = full_efficiency["all_full_generation"]
    print(
        f"{'all-full':8s} | "
        f"retrievals: {total_eff['total_retrievals']} | "
        f"llm calls: {total_eff['total_llm_calls']} | "
        f"total latency: {total_eff['total_latency_s']:.2f}s | "
        f"avg latency: {total_eff['avg_latency_s']:.4f}s"
    )

    if total_eff["total_latency_s"] > 0:
        speedup = total_eff["total_latency_s"] / max(adaptive_eff["total_latency_s"], 1e-9)
        print(f"\nAdaptive speedup vs full generation: {speedup:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained router on a split")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="validation", choices=sorted(VALID_SPLITS))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_router_evaluation(
        config_path=args.config,
        split=args.split,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
