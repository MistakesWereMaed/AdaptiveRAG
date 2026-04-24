import argparse
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from src.data.preprocessing import extract_qa_records, load_records, normalize_answer
from src.utils.eval import compute_f1
from src.utils.config import load_yaml_config


def exact_match(prediction: str, reference: str) -> float:
	return float(normalize_answer(prediction) == normalize_answer(reference))


def main():
	parser = argparse.ArgumentParser(description="Evaluate QA predictions")
	parser.add_argument("--config", default="configs/train.yaml", help="Path to evaluation config")
	parser.add_argument("--predictions", default=None, help="Path to predictions JSON/JSONL")
	parser.add_argument("--references", default=None, help="Path to reference QA JSON/JSONL")
	args = parser.parse_args()

	config = load_yaml_config(args.config)
	predictions_path = args.predictions or config.get("predictions")
	references_path = args.references or config.get("references")

	if predictions_path is None:
		raise ValueError("A predictions path must be provided via --predictions or the config file")
	if references_path is None:
		raise ValueError("A references path must be provided via --references or the config file")

	predictions = load_records(predictions_path)
	references = extract_qa_records(load_records(references_path))

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


if __name__ == "__main__":
	main()

