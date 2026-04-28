import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.file_loader import load_records, load_predictions
from src.data.squad import evaluate_batch, mean


# ============================================================
# Label mapping
# ============================================================
STRATEGY_TO_LABEL = {"no-rag": 0, "single": 1, "multi": 2}
STRATEGY_PRIORITY = {"no-rag": 0, "single": 1, "multi": 2}


# ============================================================
# Utilities
# ============================================================
def _best_strategy(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda x: (x[1], -STRATEGY_PRIORITY[x[0]]))[0]


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = __import__("yaml").safe_load(open(args.config))
    cfg = cfg.get("labels", cfg)

    dataset = load_records(cfg["dataset"])
    predictions = load_predictions(cfg["predictions"])

    batch_size = int(cfg["batch_size"])
    output_path = Path(cfg["output"])
    stats_path = output_path.with_suffix(".stats.json")

    total = len(dataset)

    # --------------------------------------------------------
    # validation
    # --------------------------------------------------------
    for s in STRATEGY_TO_LABEL:
        attr = s.replace("-", "_")
        preds_list = getattr(predictions, attr)
        if len(preds_list) != total:
            raise ValueError(f"{s} prediction length mismatch")

    # --------------------------------------------------------
    # accumulators
    # --------------------------------------------------------
    strategy_em = {s: [] for s in STRATEGY_TO_LABEL}
    strategy_f1 = {s: [] for s in STRATEGY_TO_LABEL}
    strategy_combined = {s: [] for s in STRATEGY_TO_LABEL}

    label_counts = {s: 0 for s in STRATEGY_TO_LABEL}
    oracle_scores = []

    results = []

    # --------------------------------------------------------
    # batching
    # --------------------------------------------------------
    for start in tqdm(range(0, total, batch_size), desc="Labeling"):
        batch = dataset[start:start + batch_size]

        ids = [x.id for x in batch]
        questions = [x.question for x in batch]
        golds = [x.gold for x in batch]

        batch_scores = {}

        # ----------------------------------------------------
        # evaluate each strategy in vectorized batch
        # ----------------------------------------------------
        for strategy in STRATEGY_TO_LABEL:
            attr = strategy.replace("-", "_")
            preds_list = getattr(predictions, attr)
            preds = preds_list[start:start + batch_size]
            preds_texts = [p.prediction for p in preds]

            res = evaluate_batch(preds_texts, golds)
            em = res["em"]
            f1 = res["f1"]
            combined = f1 # baseline test - TODO: change
            #combined = [(e + f) / 2 for e, f in zip(em, f1)]

            strategy_em[strategy].extend(em)
            strategy_f1[strategy].extend(f1)
            strategy_combined[strategy].extend(combined)

            batch_scores[strategy] = combined

        # ----------------------------------------------------
        # label assignment
        # ----------------------------------------------------
        for i in range(len(batch)):
            scores = {s: batch_scores[s][i] for s in STRATEGY_TO_LABEL}

            best = _best_strategy(scores)

            label_counts[best] += 1
            oracle_scores.append(max(scores.values()))

            results.append({
                "id": ids[i],
                "question": questions[i],
                "gold": golds[i],
                "label": STRATEGY_TO_LABEL[best],
                "label_name": best,
                "scores": scores,
            })

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------
    stats = {
        "total": total,
        "strategy_metrics": {},
        "label_distribution": {},
        "oracle_mean": mean(oracle_scores),
    }

    for s in STRATEGY_TO_LABEL:
        stats["strategy_metrics"][s] = {
            "em": mean(strategy_em[s]),
            "f1": mean(strategy_f1[s]),
            "combined": mean(strategy_combined[s]),
        }

    for s, c in label_counts.items():
        stats["label_distribution"][s] = {
            "count": c,
            "pct": c / total,
        }

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # --------------------------------------------------------
    # print summary
    # --------------------------------------------------------
    print("\n=== Strategy Metrics ===")
    for s, m in stats["strategy_metrics"].items():
        print(f"{s:8s} | EM: {m['em']:.4f} | F1: {m['f1']:.4f} | Combined: {m['combined']:.4f}")

    print("\n=== Label Distribution ===")
    for s, d in stats["label_distribution"].items():
        print(f"{s:8s} | {d['count']} ({d['pct']:.2%})")

    print("\n=== Oracle ===")
    print(f"{stats['oracle_mean']:.4f}")

    print(f"\nSaved: {output_path}")
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()