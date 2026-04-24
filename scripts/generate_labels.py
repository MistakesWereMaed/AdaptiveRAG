import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import extract_qa_records, load_records
from src.utils.config import load_yaml_config
from src.utils.eval import compute_f1, normalize


STRATEGY_TO_LABEL = {"no": 0, "single": 1, "multi": 2}
STRATEGY_PRIORITY = {"no": 0, "single": 1, "multi": 2}


def _best_strategy(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: (item[1], -STRATEGY_PRIORITY[item[0]]))[0]


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def _load_predictions(path: str | Path) -> Dict[str, list]:
    predictions_path = Path(path)
    with predictions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Predictions file must be a JSON object with strategy keys: no/single/multi")

    required_keys = {"no", "single", "multi"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(
            f"Predictions file is missing required strategy outputs: {sorted(missing)}. "
            "Run run_rag_pipelines.py with --strategy all."
        )

    for key in required_keys:
        if not isinstance(payload[key], list):
            raise ValueError(f"Predictions for strategy '{key}' must be a list")

    return payload


def main():
    print("[generate_labels] Starting label generation", flush=True)
    parser = argparse.ArgumentParser(description="Generate weak labels for Adaptive-RAG routing")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to runtime config")
    parser.add_argument("--dataset", default=None, help="Path to QA dataset JSON/JSONL")
    parser.add_argument("--predictions", default=None, help="Path to predictions JSON from run_rag_pipelines.py")
    parser.add_argument("--output", default=None, help="Path to output labeled JSON file")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for score aggregation")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    dataset_path = args.dataset or config.get("train_data")
    predictions_path = args.predictions or config.get("output", "outputs/predictions.json")
    output_path = Path(args.output or config.get("labels_output", "outputs/labeled_train.json"))
    batch_size = args.batch_size or int(config.get("batch_size", 8))

    if dataset_path is None:
        raise ValueError("A dataset path must be provided via --dataset or config train_data")
    if predictions_path is None:
        raise ValueError("A predictions path must be provided via --predictions or config output")

    dataset = extract_qa_records(load_records(dataset_path))
    outputs = _load_predictions(predictions_path)
    print(f"[generate_labels] Loaded {len(dataset)} examples and predictions from {predictions_path}", flush=True)

    total_examples = len(dataset)
    for strategy in ("no", "single", "multi"):
        if len(outputs[strategy]) != total_examples:
            raise ValueError(
                f"Mismatch between dataset size ({total_examples}) and '{strategy}' predictions "
                f"({len(outputs[strategy])}). Ensure predictions were generated from the same dataset."
            )

    results = []
    for start in tqdm(range(0, len(dataset), batch_size), desc="Labeling", unit="batch"):
        batch = dataset[start:start + batch_size]
        for batch_index, item in enumerate(batch):
            global_index = start + batch_index
            question = item["question"]
            gold = item["answer"]
            scores = {}
            for strategy in ("no", "single", "multi"):
                prediction = outputs[strategy][global_index]
                scores[strategy] = max(compute_f1(prediction, gold), exact_match(prediction, gold))

            label_name = _best_strategy(scores)
            results.append(
                {
                    "question": question,
                    "answer": gold,
                    "label": STRATEGY_TO_LABEL[label_name],
                    "label_name": label_name,
                    "scores": scores,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[generate_labels] Wrote labels to {output_path}", flush=True)


if __name__ == "__main__":
    main()
