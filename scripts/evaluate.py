import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.file_loader import load_records, load_single_predictions, normalize_text, load_yaml_config


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def compute_f1(prediction: str, reference: str) -> float:
    p = normalize_text(prediction).split()
    r = normalize_text(reference).split()
    if not p or not r:
        return 0.0
    common = set(p) & set(r)
    num_same = sum(min(p.count(w), r.count(w)) for w in common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(r)
    return 2 * precision * recall / (precision + recall)


def main():
    print("[evaluate] Starting evaluation", flush=True)
    parser = argparse.ArgumentParser(description="Evaluate QA predictions")
    parser.add_argument("--config", default="config.yaml", help="Path to evaluation config")
    args = parser.parse_args()

    config = load_yaml_config(args.config, section="evaluate")
    predictions_path = str(config["predictions"])
    references_path = str(config["references"])

    # references -> List[QAItem]
    references = load_records(references_path)
    # predictions -> List[PredictionItem]
    predictions = load_single_predictions(predictions_path)

    print(f"[evaluate] Loaded {len(predictions)} predictions and {len(references)} references", flush=True)

    refs_by_id = {r.id: r for r in references}
    preds_by_id = {p.id: p for p in predictions}

    if set(refs_by_id.keys()) != set(preds_by_id.keys()):
        raise ValueError("Prediction ids and reference ids do not match")

    exact_match_scores = []
    f1_scores = []
    for _id in tqdm(sorted(refs_by_id.keys()), desc="Evaluating", unit="example"):
        reference = refs_by_id[_id]
        prediction = preds_by_id[_id]
        pred_text = prediction.prediction
        gold_text = reference.gold
        exact_match_scores.append(exact_match(pred_text, gold_text))
        f1_scores.append(compute_f1(pred_text, gold_text))

    metrics = {
        "exact_match": sum(exact_match_scores) / max(1, len(exact_match_scores)),
        "f1": sum(f1_scores) / max(1, len(f1_scores)),
    }

    print(json.dumps(metrics, indent=2))
    print("[evaluate] Finished evaluation", flush=True)


if __name__ == "__main__":
    main()
