import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs6263_template.src.myproject.src.data.file_loader import load_records, load_predictions
from cs6263_template.src.myproject.src.data.squad import evaluate_batch, is_correct, mean


STRATEGY_TO_LABEL = {"no-rag": 0, "single": 1, "multi": 2}
STRATEGY_PRIORITY = ["no-rag", "single", "multi"]  # ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = __import__("yaml").safe_load(open(args.config))
    paths = cfg["paths"]
    cfg_labels = cfg["labels"]

    dataset = load_records(paths["train_data"])
    predictions = load_predictions(paths["predictions_base"])

    batch_size = int(cfg_labels["batch_size"])
    output_path = Path(paths["labeled_train"])
    stats_path = Path(paths["labeled_stats"])

    total = len(dataset)

    # -----------------------------
    # accumulators
    # -----------------------------
    strategy_metrics = {
        s: {"f1": [], "em": [], "acc": []}
        for s in STRATEGY_TO_LABEL
    }

    label_counts = {s: 0 for s in STRATEGY_TO_LABEL}
    unlabeled_count = 0
    oracle_scores = []

    results = []

    # -----------------------------
    # batching
    # -----------------------------
    for start in tqdm(range(0, total, batch_size), desc="Labeling"):
        batch = dataset[start:start + batch_size]

        ids = [x.id for x in batch]
        questions = [x.question for x in batch]
        golds = [x.gold for x in batch]

        batch_eval = {}

        # ---- evaluate each strategy ----
        for strategy in STRATEGY_TO_LABEL:
            preds_list = getattr(predictions, strategy.replace("-", "_"))
            preds = preds_list[start:start + batch_size]
            pred_texts = [p.prediction for p in preds]

            res = evaluate_batch(pred_texts, golds)

            strategy_metrics[strategy]["f1"].extend(res["f1"])
            strategy_metrics[strategy]["em"].extend(res["em"])
            strategy_metrics[strategy]["acc"].extend(res["acc"])

            batch_eval[strategy] = res

        # ---- labeling (PRIORITY-BASED) ----
        for i in range(len(batch)):
            correct_flags = {}

            for strategy in STRATEGY_TO_LABEL:
                f1 = batch_eval[strategy]["f1"][i]
                em = batch_eval[strategy]["em"][i]

                pred = getattr(predictions, strategy.replace("-", "_"))[start + i].prediction
                gold = golds[i]

                correct_flags[strategy] = is_correct(pred, gold, f1, em)

            # ---- apply priority ----
            label_name = None
            for s in STRATEGY_PRIORITY:
                if correct_flags[s]:
                    label_name = s
                    break

            if label_name is None:
                unlabeled_count += 1
                label_name = "no-rag"  # fallback (can change)

            label_counts[label_name] += 1

            oracle_scores.append(
                max(batch_eval[s]["f1"][i] for s in STRATEGY_TO_LABEL)
            )

            results.append({
                "id": ids[i],
                "question": questions[i],
                "gold": golds[i],
                "label": STRATEGY_TO_LABEL[label_name],
                "label_name": label_name,
                "correct": correct_flags,
            })

    # -----------------------------
    # stats
    # -----------------------------
    stats = {
        "total": total,
        "unlabeled": unlabeled_count,
        "strategy_metrics": {},
        "label_distribution": {},
        "oracle_f1": mean(oracle_scores),
    }

    for s in STRATEGY_TO_LABEL:
        stats["strategy_metrics"][s] = {
            "f1": mean(strategy_metrics[s]["f1"]),
            "em": mean(strategy_metrics[s]["em"]),
            "acc": mean(strategy_metrics[s]["acc"]),
        }

    for s, c in label_counts.items():
        stats["label_distribution"][s] = {
            "count": c,
            "pct": c / total,
        }

    # -----------------------------
    # save
    # -----------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    # -----------------------------
    # print
    # -----------------------------
    print("\n=== Strategy Metrics ===")
    for s, m in stats["strategy_metrics"].items():
        print(f"{s:8s} | F1: {m['f1']:.4f} | EM: {m['em']:.4f} | Acc: {m['acc']:.4f}")

    print("\n=== Label Distribution ===")
    for s, d in stats["label_distribution"].items():
        print(f"{s:8s} | {d['count']} ({d['pct']:.2%})")

    print("\n=== Oracle F1 ===")
    print(f"{stats['oracle_f1']:.4f}")

    print(f"\nUnlabeled: {unlabeled_count}")


if __name__ == "__main__":
    main()