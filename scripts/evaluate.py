import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import extract_qa_records, load_records, normalize_answer
from src.utils.config import load_yaml_config
from src.utils.eval import compute_f1


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def main():
    print("[evaluate] Starting evaluation", flush=True)
    parser = argparse.ArgumentParser(description="Evaluate QA predictions")
    parser.add_argument("--config", default="configs/evaluate.yaml", help="Path to evaluation config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    predictions_path = str(config["predictions"])
    references_path = str(config["references"])

    predictions = load_records(predictions_path)
    references = extract_qa_records(load_records(references_path))
    print(f"[evaluate] Loaded {len(predictions)} predictions and {len(references)} references", flush=True)

    if len(predictions) != len(references):
        raise ValueError("Predictions and references must contain the same number of records")

    exact_match_scores = []
    f1_scores = []
    for prediction, reference in tqdm(zip(predictions, references), total=len(predictions), desc="Evaluating", unit="example"):
        pred_text = prediction.get("prediction") or prediction.get("answer") or ""
        gold_text = reference["answer"]
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
