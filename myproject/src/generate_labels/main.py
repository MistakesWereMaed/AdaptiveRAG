from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.file_loader import load_predictions, load_records, load_yaml_config
from src.generate_labels.squad import evaluate_batch, is_correct, mean


STRATEGY_TO_LABEL = {"no-rag": 0, "single": 1, "multi": 2}
STRATEGY_ORDER = list(STRATEGY_TO_LABEL)
COST_PRIORITY = ["no-rag", "single", "multi"]


def _strategy_attr(strategy: str) -> str:
    return strategy.replace("-", "_")


def _prediction_texts(predictions, strategy: str, start: int, end: int) -> List[str]:
    strategy_predictions = getattr(predictions, _strategy_attr(strategy))
    return [item.prediction for item in strategy_predictions[start:end]]


def _rank(scores: Dict[str, float], ems: Dict[str, float]) -> List[str]:
    return sorted(STRATEGY_ORDER, key=lambda s: (ems[s], scores[s]), reverse=True)


def choose_label(
    scores: Dict[str, float],
    ems: Dict[str, float],
    min_f1: float,
    margin: float,
) -> Tuple[Optional[str], bool]:
    ranked = _rank(scores, ems)
    best, second = ranked[0], ranked[1]

    if scores[best] < min_f1 and ems[best] < 1.0:
        return None, False

    # Cost-aware tie-break for near-equal strategy quality.
    if ems[best] == ems[second] and scores[best] - scores[second] < margin:
        tied = {best, second}
        best = next(strategy for strategy in COST_PRIORITY if strategy in tied)

    weak = scores[best] < 0.8 and ems[best] < 1.0
    return best, weak


def _available_examples(dataset_size: int, predictions) -> int:
    available = [dataset_size]

    for strategy in STRATEGY_ORDER:
        available.append(len(getattr(predictions, _strategy_attr(strategy))))

    return min(available)


def run_generate_labels(config_path: str = "config.yaml") -> None:
    cfg = load_yaml_config(config_path)
    paths = cfg["paths"]
    label_cfg = cfg["labels"]

    min_f1 = float(label_cfg.get("min_f1", 0.5))
    margin = float(label_cfg.get("margin", 0.05))
    batch_size = int(label_cfg.get("batch_size", 64))

    dataset = load_records(paths["train_data"])
    predictions = load_predictions(paths["predictions_base"])

    evaluated_total = _available_examples(len(dataset), predictions)
    dataset = dataset[:evaluated_total]

    output_path = Path(paths["labeled_train"])
    stats_path = Path(paths["labeled_stats"])

    strategy_metrics = {s: {"f1": [], "em": [], "acc": []} for s in STRATEGY_ORDER}
    label_counts = {s: 0 for s in STRATEGY_ORDER}

    labeled_records = []
    unlabeled = 0
    weak_labels = 0
    oracle_f1s = []

    for start in tqdm(range(0, evaluated_total, batch_size), desc="Labeling"):
        end = min(start + batch_size, evaluated_total)
        batch = dataset[start:end]

        golds = [item.gold for item in batch]
        pred_texts = {
            strategy: _prediction_texts(predictions, strategy, start, end)
            for strategy in STRATEGY_ORDER
        }

        batch_scores = {
            strategy: evaluate_batch(pred_texts[strategy], golds)
            for strategy in STRATEGY_ORDER
        }

        for strategy, metrics in batch_scores.items():
            for metric_name in ("f1", "em", "acc"):
                strategy_metrics[strategy][metric_name].extend(metrics[metric_name])

        for i, item in enumerate(batch):
            scores = {s: batch_scores[s]["f1"][i] for s in STRATEGY_ORDER}
            ems = {s: batch_scores[s]["em"][i] for s in STRATEGY_ORDER}
            accs = {s: batch_scores[s]["acc"][i] for s in STRATEGY_ORDER}
            correct = {s: is_correct(scores[s], ems[s]) for s in STRATEGY_ORDER}

            oracle_f1s.append(max(scores.values()))

            label_name, weak = choose_label(scores, ems, min_f1=min_f1, margin=margin)

            if label_name is None:
                unlabeled += 1
                continue

            label_counts[label_name] += 1
            weak_labels += int(weak)

            labeled_records.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "gold": item.gold,
                    "label": STRATEGY_TO_LABEL[label_name],
                    "label_name": label_name,
                    "scores": scores,
                    "em": ems,
                    "acc": accs,
                    "correct": correct,
                    "weak": weak,
                }
            )

    labeled_total = len(labeled_records)

    stats = {
        "total_examples": len(load_records(paths["train_data"])),
        "evaluated_examples": evaluated_total,
        "labeled_examples": labeled_total,
        "unlabeled": unlabeled,
        "unlabeled_pct_of_evaluated": unlabeled / evaluated_total if evaluated_total else 0.0,
        "weak_labels": weak_labels,
        "weak_label_pct": weak_labels / labeled_total if labeled_total else 0.0,
        "min_f1": min_f1,
        "margin": margin,
        "oracle_f1": mean(oracle_f1s),
        "strategy_metrics": {
            strategy: {
                metric: mean(values)
                for metric, values in metrics.items()
            }
            for strategy, metrics in strategy_metrics.items()
        },
        "label_distribution": {
            strategy: {
                "count": count,
                "pct_of_labeled": count / labeled_total if labeled_total else 0.0,
                "pct_of_evaluated": count / evaluated_total if evaluated_total else 0.0,
            }
            for strategy, count in label_counts.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(labeled_records, f, ensure_ascii=False, indent=2)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Strategy Metrics ===")
    for strategy, metrics in stats["strategy_metrics"].items():
        print(
            f"{strategy:8s} | "
            f"F1: {metrics['f1']:.4f} | "
            f"EM: {metrics['em']:.4f} | "
            f"Acc: {metrics['acc']:.4f}"
        )

    print("\n=== Label Distribution ===")
    for strategy, dist in stats["label_distribution"].items():
        print(
            f"{strategy:8s} | "
            f"{dist['count']} "
            f"({dist['pct_of_labeled']:.2%} of labeled, "
            f"{dist['pct_of_evaluated']:.2%} of evaluated)"
        )

    print("\n=== Oracle F1 ===")
    print(f"{stats['oracle_f1']:.4f}")

    print("\n=== Labeling Summary ===")
    print(f"Total examples:   {stats['total_examples']}")
    print(f"Evaluated:        {evaluated_total}")
    print(f"Labeled examples: {labeled_total}")
    print(f"Unlabeled:        {unlabeled} ({stats['unlabeled_pct_of_evaluated']:.2%} of evaluated)")
    print(f"Weak labels:      {weak_labels} ({stats['weak_label_pct']:.2%} of labeled)")


def main() -> None:
    run_generate_labels("config.yaml")


if __name__ == "__main__":
    main()
