from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

from src.build_index.retriever import FaissIVFRetriever
from src.file_loader import load_predictions, load_records, load_yaml_config
from src.generate_labels.squad import evaluate_batch, is_correct, mean
from src.generate_responses.llm import LocalLLM
from src.generate_responses.pipeline import AdaptiveRAGPipeline, PipelineOutput


STRATEGY_TO_LABEL = {"no-rag": 0, "single": 1, "multi": 2}


def _load_configs(config_path: str):
    paths = load_yaml_config(config_path, section="paths")
    pipeline_cfg = load_yaml_config(config_path, section="pipeline")

    if "llm_config" in pipeline_cfg and "retriever_config" in pipeline_cfg:
        llm_cfg = load_yaml_config(pipeline_cfg["llm_config"], section="llm")
        retriever_cfg = load_yaml_config(pipeline_cfg["retriever_config"], section="retriever")
    else:
        llm_cfg = load_yaml_config(config_path, section="llm")
        retriever_cfg = load_yaml_config(config_path, section="retriever")

    labels_cfg = load_yaml_config(config_path, section="labels")
    return paths, pipeline_cfg, llm_cfg, retriever_cfg, labels_cfg


def _score_outputs(outputs: List[PipelineOutput], records: List[Any]) -> Dict[str, List[float]]:
    return evaluate_batch(
        [output.prediction for output in outputs],
        [record.gold for record in records],
    )


def _correct_flags(scores: Dict[str, List[float]], threshold: float) -> List[bool]:
    return [
        is_correct(f1=f1, em=em, threshold=threshold)
        for f1, em in zip(scores["f1"], scores["em"])
    ]


def _label_record(
    record: Any,
    output: PipelineOutput,
    strategy: str,
    f1: float,
    em: float,
    acc: float,
) -> Dict[str, Any]:
    weak = not (em == 1.0 or f1 >= 0.8)

    return {
        "id": record.id,
        "question": record.question,
        "gold": record.gold,
        "label": STRATEGY_TO_LABEL[strategy],
        "label_name": strategy,
        "prediction": output.prediction,
        "scores": {strategy: f1},
        "em": {strategy: em},
        "acc": {strategy: acc},
        "correct": {strategy: True},
        "weak": weak,
    }


def _prediction_row(record: Any, output: PipelineOutput, strategy: str) -> Dict[str, Any]:
    return {
        "id": record.id,
        "question": record.question,
        "gold": record.gold,
        "prediction": output.prediction,
        "strategy": strategy,
        "retrieval_count": output.retrieval_count,
        "llm_calls": output.llm_calls,
        "latency_s": output.latency_s,
    }


def _write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_strategy(
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

    for start in tqdm(range(0, len(records), batch_size), desc=strategy, unit="batch"):
        batch = records[start : start + batch_size]
        questions = [record.question for record in batch]

        if strategy == "no-rag":
            batch_outputs = pipeline.no_retrieval(questions)
        elif strategy == "single":
            batch_outputs = pipeline.single_step(questions, k=single_k)
        elif strategy == "multi":
            batch_outputs = pipeline.multi_step(
                questions,
                steps=multi_steps,
                k=multi_k,
                final_k=final_k_multi,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        outputs.extend(batch_outputs)

    return outputs


def run_cascaded_label_generation(
    config_path: str = "config.yaml",
    split: str = "train",
    limit: int | None = None,
    save_stage_predictions: bool = True,
) -> None:
    paths, pipeline_cfg, llm_cfg, retriever_cfg, labels_cfg = _load_configs(config_path)

    data_path = paths["train_data"] if split == "train" else paths["validation_data"]
    records = load_records(data_path)

    if limit is not None:
        records = records[: max(0, int(limit))]

    batch_size = int(pipeline_cfg.get("pipeline_batch_size", labels_cfg.get("batch_size", 8)))
    threshold = float(labels_cfg.get("correct_threshold", 0.8))

    single_k = int(retriever_cfg.get("top_k_single", 6))
    multi_k = int(retriever_cfg.get("top_k_multi", 6))
    final_k_multi = int(retriever_cfg.get("final_k_multi", multi_k))
    multi_steps = int((pipeline_cfg.get("multi", {}) or {}).get("steps", 2))

    output_path = Path(paths["labeled_train"])
    stats_path = Path(paths["labeled_stats"])
    prediction_base = Path(paths["predictions_base"])

    print(f"[cascade] Loaded {len(records)} records from {data_path}", flush=True)
    print("[cascade] Initializing LLM...", flush=True)
    llm = LocalLLM(llm_cfg)

    print("[cascade] Loading retriever...", flush=True)
    retriever = FaissIVFRetriever(
        encoder_name=retriever_cfg["encoder_name"],
        nprobe=int(retriever_cfg.get("nprobe", 8)),
    ).load(paths["index_dir"])

    pipeline = AdaptiveRAGPipeline(llm, retriever)

    remaining_indices = list(range(len(records)))
    labels: List[Dict[str, Any]] = []
    unlabeled: List[Dict[str, Any]] = []

    label_counts = {strategy: 0 for strategy in STRATEGY_TO_LABEL}
    weak_count = 0
    total_latency = 0.0
    total_retrievals = 0
    total_llm_calls = 0

    attempted_metrics = {
        "no-rag": {"f1": [], "em": [], "acc": []},
        "single": {"f1": [], "em": [], "acc": []},
        "multi": {"f1": [], "em": [], "acc": []},
    }

    stage_prediction_rows = {
        "no-rag": [],
        "single": [],
        "multi": [],
    }

    for strategy in ("no-rag", "single", "multi"):
        if not remaining_indices:
            break

        stage_records = [records[i] for i in remaining_indices]
        print(f"\n[cascade] Stage={strategy} examples={len(stage_records)}", flush=True)

        outputs = _run_strategy(
            pipeline=pipeline,
            strategy=strategy,
            records=stage_records,
            batch_size=batch_size,
            single_k=single_k,
            multi_k=multi_k,
            multi_steps=multi_steps,
            final_k_multi=final_k_multi,
        )

        scores = _score_outputs(outputs, stage_records)
        correct = _correct_flags(scores, threshold=threshold)

        for metric_name in ("f1", "em", "acc"):
            attempted_metrics[strategy][metric_name].extend(scores[metric_name])

        next_remaining: List[int] = []

        for local_i, global_i in enumerate(remaining_indices):
            record = records[global_i]
            output = outputs[local_i]

            total_latency += output.latency_s
            total_retrievals += output.retrieval_count
            total_llm_calls += output.llm_calls

            if save_stage_predictions:
                stage_prediction_rows[strategy].append(_prediction_row(record, output, strategy))

            if correct[local_i]:
                row = _label_record(
                    record=record,
                    output=output,
                    strategy=strategy,
                    f1=scores["f1"][local_i],
                    em=scores["em"][local_i],
                    acc=scores["acc"][local_i],
                )
                labels.append(row)
                label_counts[strategy] += 1
                weak_count += int(row["weak"])
            else:
                next_remaining.append(global_i)

        print(
            f"[cascade] {strategy}: newly_labeled={label_counts[strategy]} "
            f"remaining={len(next_remaining)} "
            f"attempted_f1={mean(scores['f1']):.4f} "
            f"attempted_em={mean(scores['em']):.4f}",
            flush=True,
        )

        remaining_indices = next_remaining

    for idx in remaining_indices:
        record = records[idx]
        unlabeled.append(
            {
                "id": record.id,
                "question": record.question,
                "gold": record.gold,
            }
        )

    total = len(records)
    labeled_total = len(labels)

    stats = {
        "mode": "cascaded_cheapest_correct",
        "correct_threshold": threshold,
        "total_examples": total,
        "labeled_examples": labeled_total,
        "unlabeled": len(unlabeled),
        "unlabeled_pct": len(unlabeled) / total if total else 0.0,
        "weak_labels": weak_count,
        "weak_label_pct": weak_count / labeled_total if labeled_total else 0.0,
        "total_latency_s": total_latency,
        "avg_latency_s": total_latency / total if total else 0.0,
        "total_retrievals": total_retrievals,
        "total_llm_calls": total_llm_calls,
        "attempted_strategy_metrics": {
            strategy: {metric: mean(values) for metric, values in metrics.items()}
            for strategy, metrics in attempted_metrics.items()
        },
        "label_distribution": {
            strategy: {
                "count": count,
                "pct_of_labeled": count / labeled_total if labeled_total else 0.0,
                "pct_of_total": count / total if total else 0.0,
            }
            for strategy, count in label_counts.items()
        },
    }

    _write_json(output_path, labels)
    _write_json(stats_path, stats)
    _write_json(stats_path.with_name(stats_path.stem + "_unlabeled.json"), unlabeled)

    if save_stage_predictions:
        for strategy, rows in stage_prediction_rows.items():
            out_path = prediction_base.with_name(f"{prediction_base.stem}-{strategy}-cascade.jsonl")
            _write_jsonl(out_path, rows)

    print("\n=== Cascaded Label Distribution ===")
    for strategy, item in stats["label_distribution"].items():
        print(
            f"{strategy:8s} | {item['count']} "
            f"({item['pct_of_labeled']:.2%} of labeled, {item['pct_of_total']:.2%} of total)"
        )

    print("\n=== Cascaded Summary ===")
    print(f"Total examples:   {total}")
    print(f"Labeled examples: {labeled_total}")
    print(f"Unlabeled:        {len(unlabeled)} ({stats['unlabeled_pct']:.2%})")
    print(f"Weak labels:      {weak_count} ({stats['weak_label_pct']:.2%} of labeled)")
    print(f"Total LLM calls:  {total_llm_calls}")
    print(f"Total retrievals: {total_retrievals}")
    print(f"Avg latency:      {stats['avg_latency_s']:.4f}s/example")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate router labels with cascaded strategy execution")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-stage-predictions", action="store_true")
    args = parser.parse_args()

    run_cascaded_label_generation(
        config_path=args.config,
        split=args.split,
        limit=args.limit,
        save_stage_predictions=not args.no_stage_predictions,
    )


if __name__ == "__main__":
    main()
