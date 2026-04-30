from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.file_loader import load_predictions, load_yaml_config
from src.generate_labels.squad import evaluate_batch, mean


STRATEGIES = ["no-rag", "single", "multi"]


def _data_path(paths: dict, split: str) -> str:
    key = f"{split}_data"
    if key not in paths:
        raise KeyError(f"Missing paths.{key} in config")
    return paths[key]


def _prediction_base(paths: dict, split: str) -> Path:
    split_key = f"predictions_{split}_base"
    if split_key in paths:
        return Path(paths[split_key])
    base = Path(paths["predictions_base"])
    return base.with_name(f"{base.stem}-{split}{base.suffix}")


def _adaptive_output_path(paths: dict, split: str) -> Path:
    if f"adaptive_predictions_{split}" in paths:
        return Path(paths[f"adaptive_predictions_{split}"])
    base = _prediction_base(paths, split)
    return base.with_name(f"{base.stem}-adaptive.jsonl")


def _adaptive_stats_path(paths: dict, split: str) -> Path:
    if f"adaptive_stats_{split}" in paths:
        return Path(paths[f"adaptive_stats_{split}"])
    out = _adaptive_output_path(paths, split)
    return out.with_name(f"{out.stem}_stats.json")


def _load_configs(config_path: str):
    paths = load_yaml_config(config_path, section="paths")
    pipeline_cfg = load_yaml_config(config_path, section="pipeline")

    if "llm_config" in pipeline_cfg and "retriever_config" in pipeline_cfg:
        llm_cfg = load_yaml_config(pipeline_cfg["llm_config"], section="llm")
        retriever_cfg = load_yaml_config(pipeline_cfg["retriever_config"], section="retriever")
    else:
        llm_cfg = load_yaml_config(config_path, section="llm")
        retriever_cfg = load_yaml_config(config_path, section="retriever")

    train_cfg = load_yaml_config(config_path, section="train")
    model_cfg = load_yaml_config(config_path, section="model")
    return paths, pipeline_cfg, llm_cfg, retriever_cfg, train_cfg, model_cfg


def _find_checkpoint(paths: dict, train_cfg: dict, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    for key in ("router_checkpoint", "classifier_checkpoint", "best_router_checkpoint"):
        value = paths.get(key)
        if value and Path(value).exists():
            return str(value)

    ckpt_dir = Path(train_cfg.get("output_dir", "outputs/router")) / "checkpoints"
    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No router checkpoint found. Pass --checkpoint or set paths.router_checkpoint.")

    non_last = [p for p in candidates if p.name != "last.ckpt"]
    return str(non_last[0] if non_last else candidates[0])



def _score_predictions(predictions: List[str], records: List[Any]) -> Dict[str, float]:
    scores = evaluate_batch(predictions, [record.gold for record in records])
    return {name: mean(values) for name, values in scores.items()}


def _load_full_strategy_scores(paths: dict, split: str, records: List[Any]) -> Dict[str, Dict[str, float]]:
    loaded = load_predictions(_prediction_base(paths, split))
    output = {}

    for strategy in STRATEGIES:
        attr = strategy.replace("-", "_")
        strategy_preds = getattr(loaded, attr)
        n = min(len(strategy_preds), len(records))
        output[strategy] = _score_predictions(
            [item.prediction for item in strategy_preds[:n]],
            records[:n],
        )

    return output


def _oracle_score_from_full_predictions(paths: dict, split: str, records: List[Any]) -> Dict[str, float]:
    loaded = load_predictions(_prediction_base(paths, split))
    n = min(len(records), len(loaded.no_rag), len(loaded.single), len(loaded.multi))
    golds = [record.gold for record in records[:n]]

    strategy_preds = {
        "no-rag": [item.prediction for item in loaded.no_rag[:n]],
        "single": [item.prediction for item in loaded.single[:n]],
        "multi": [item.prediction for item in loaded.multi[:n]],
    }

    scored = {strategy: evaluate_batch(preds, golds) for strategy, preds in strategy_preds.items()}

    return {
        "f1": mean([max(scored[s]["f1"][i] for s in STRATEGIES) for i in range(n)]),
        "em": mean([max(scored[s]["em"][i] for s in STRATEGIES) for i in range(n)]),
        "acc": mean([max(scored[s]["acc"][i] for s in STRATEGIES) for i in range(n)]),
    }


def _load_full_strategy_efficiency(paths: dict, split: str) -> Dict[str, Dict[str, float]]:
    base = _prediction_base(paths, split)
    output = {}

    for strategy in STRATEGIES:
        stats_path = base.with_name(f"{base.stem}-{strategy}_stats.json")

        if not stats_path.exists():
            output[strategy] = {
                "num_examples": 0,
                "total_retrievals": 0,
                "total_llm_calls": 0,
                "total_latency_s": 0.0,
                "avg_latency_s": 0.0,
                "stats_file_missing": True,
            }
            continue

        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        output[strategy] = {
            "num_examples": int(stats.get("num_examples", 0)),
            "total_retrievals": int(stats.get("total_retrievals", 0)),
            "total_llm_calls": int(stats.get("total_llm_calls", 0)),
            "total_latency_s": float(stats.get("total_latency_s", 0.0)),
            "avg_latency_s": float(stats.get("avg_latency_s", 0.0)),
            "stats_file_missing": False,
        }

    output["all_full_generation"] = {
        "num_examples": max(v["num_examples"] for v in output.values()) if output else 0,
        "total_retrievals": sum(v["total_retrievals"] for v in output.values()),
        "total_llm_calls": sum(v["total_llm_calls"] for v in output.values()),
        "total_latency_s": sum(v["total_latency_s"] for v in output.values()),
    }

    n = output["all_full_generation"]["num_examples"]
    output["all_full_generation"]["avg_latency_s"] = (
        output["all_full_generation"]["total_latency_s"] / n if n else 0.0
    )

    return output