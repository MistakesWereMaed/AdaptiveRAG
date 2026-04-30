import json
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from myproject.src.file_loader import load_predictions, load_records, load_yaml_config
from myproject.src.generate_labels.squad import evaluate_batch, is_correct, mean


STRATEGY_TO_LABEL = {"no-rag": 0, "single": 1, "multi": 2}
STRATEGY_PRIORITY = ["no-rag", "single", "multi"]


def _strategy_attr(strategy: str) -> str:
    return strategy.replace("-", "_")


def _rank_strategies(scores: Dict[str, float], ems: Dict[str, float]) -> List[str]:
    return sorted(
        STRATEGY_TO_LABEL.keys(),
        key=lambda s: (ems[s], scores[s]),
        reverse=True,
    )


def _choose_label(
    scores: Dict[str, float],
    ems: Dict[str, float],
    min_f1: float,
    margin: float,
) -> Tuple[str | None, bool]:
    # EM/F1-based routing targets with cost-aware tie-break near margin.
    ranked = _rank_strategies(scores, ems)

    best = ranked[0]
    second = ranked[1]

    best_score = scores[best]
    second_score = scores[second]

    if best_score < min_f1 and ems[best] < 1.0:
        return None, False

    if best_score - second_score < margin and ems[best] == ems[second]:
        tied = {best, second}
        for strategy in STRATEGY_PRIORITY:
            if strategy in tied:
                best = strategy
                break

    weak = best_score < 0.8 and ems[best] < 1.0
    return best, weak


def run_generate_labels(config_path: str = "config.yaml") -> None:
    cfg = load_yaml_config(config_path)

    paths = cfg["paths"]
    cfg_labels = cfg["labels"]

    min_f1 = float(cfg_labels.get("min_f1", 0.5))
    margin = float(cfg_labels.get("margin", 0.05))
    batch_size = int(cfg_labels["batch_size"])

    dataset = load_records(paths["train_data"])
    predictions = load_predictions(paths["predictions_base"])

    output_path = Path(paths["labeled_train"])
    stats_path = Path(paths["labeled_stats"])

    total = len(dataset)

    strategy_metrics = {s: {"f1": [], "em": [], "acc": []} for s in STRATEGY_TO_LABEL}

    label_counts = {s: 0 for s in STRATEGY_TO_LABEL}
    unlabeled_count = 0
    weak_count = 0
    oracle_scores = []
    results = []
    evaluated_examples = 0

    for start in tqdm(range(0, total, batch_size), desc="Labeling"):
        batch = dataset[start : start + batch_size]

        ids = [x.id for x in batch]
        questions = [x.question for x in batch]
        golds = [x.gold for x in batch]

        batch_eval = {}
        pred_texts_by_strategy = {}
        available_sizes = []

        for strategy in STRATEGY_TO_LABEL:
            preds_list = getattr(predictions, _strategy_attr(strategy))
            preds = preds_list[start : start + batch_size]
            available_sizes.append(len(preds))

            pred_texts = [p.prediction for p in preds]
            pred_texts_by_strategy[strategy] = pred_texts

            res = evaluate_batch(pred_texts, golds[: len(preds)])

            strategy_metrics[strategy]["f1"].extend(res["f1"])
            strategy_metrics[strategy]["em"].extend(res["em"])
            strategy_metrics[strategy]["acc"].extend(res["acc"])

            batch_eval[strategy] = res

        actual_batch_size = min(len(batch), *available_sizes)
        evaluated_examples += actual_batch_size

        for i in range(actual_batch_size):
            scores = {strategy: batch_eval[strategy]["f1"][i] for strategy in STRATEGY_TO_LABEL}
            ems = {strategy: batch_eval[strategy]["em"][i] for strategy in STRATEGY_TO_LABEL}
            accs = {strategy: batch_eval[strategy]["acc"][i] for strategy in STRATEGY_TO_LABEL}

            correct_flags = {
                strategy: is_correct(
                    pred_texts_by_strategy[strategy][i],
                    golds[i],
                    scores[strategy],
                    ems[strategy],
                )
                for strategy in STRATEGY_TO_LABEL
            }

            label_name, weak = _choose_label(scores=scores, ems=ems, min_f1=min_f1, margin=margin)

            oracle_scores.append(max(scores.values()))

            if label_name is None:
                unlabeled_count += 1
                continue

            label_counts[label_name] += 1
            weak_count += int(weak)

            results.append(
                {
                    "id": ids[i],
                    "question": questions[i],
                    "gold": golds[i],
                    "label": STRATEGY_TO_LABEL[label_name],
                    "label_name": label_name,
                    "scores": scores,
                    "em": ems,
                    "acc": accs,
                    "correct": correct_flags,
                    "weak": weak,
                }
            )

    labeled_total = len(results)
    label_denom = max(1, labeled_total)
    unlabeled_among_evaluated = max(0, evaluated_examples - labeled_total)

    stats = {
        "total_examples": total,
        "evaluated_examples": evaluated_examples,
        "labeled_examples": labeled_total,
        "unlabeled": unlabeled_count,
        "unlabeled_among_evaluated": unlabeled_among_evaluated,
        "unlabeled_pct_of_total": unlabeled_count / total if total else 0.0,
        "unlabeled_pct_of_evaluated": unlabeled_among_evaluated / max(1, evaluated_examples),
        "weak_labels": weak_count,
        "weak_label_pct": weak_count / label_denom,
        "min_f1": min_f1,
        "margin": margin,
        "strategy_metrics": {},
        "label_distribution": {},
        "oracle_f1": mean(oracle_scores),
    }

    for strategy in STRATEGY_TO_LABEL:
        stats["strategy_metrics"][strategy] = {
            "f1": mean(strategy_metrics[strategy]["f1"]),
            "em": mean(strategy_metrics[strategy]["em"]),
            "acc": mean(strategy_metrics[strategy]["acc"]),
        }

    for strategy, count in label_counts.items():
        stats["label_distribution"][strategy] = {
            "count": count,
            "pct_of_labeled": count / label_denom,
            "pct_of_total": count / total if total else 0.0,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n=== Strategy Metrics ===")
    for strategy, metrics in stats["strategy_metrics"].items():
        print(f"{strategy:8s} | F1: {metrics['f1']:.4f} | EM: {metrics['em']:.4f} | Acc: {metrics['acc']:.4f}")

    print("\n=== Label Distribution ===")
    for strategy, dist in stats["label_distribution"].items():
        print(
            f"{strategy:8s} | {dist['count']} ({dist['pct_of_labeled']:.2%} of labeled, {dist['pct_of_total']:.2%} of total)"
        )

    print("\n=== Oracle F1 ===")
    print(f"{stats['oracle_f1']:.4f}")

    print("\n=== Labeling Summary ===")
    print(f"Total examples:   {total}")
    print(f"Evaluated:        {evaluated_examples}")
    print(f"Labeled examples: {labeled_total}")
    print(f"Unlabeled:        {unlabeled_count} ({stats['unlabeled_pct_of_total']:.2%} of total)")
    print(
        f"Unlabeled eval:   {unlabeled_among_evaluated} ({stats['unlabeled_pct_of_evaluated']:.2%} of evaluated)"
    )
    print(f"Weak labels:      {weak_count} ({stats['weak_label_pct']:.2%} of labeled)")
