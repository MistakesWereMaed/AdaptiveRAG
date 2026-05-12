#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASETS = ["musique", "2wikimultihopqa", "hotpotqa", "nq", "trivia", "squad"]
ID2LABEL = {0: "A", 1: "B", 2: "C"}
LABEL_TO_SYSTEM = {"A": "nor_qa", "B": "oner_qa", "C": "ircot_qa"}


def split_to_processed_filename(split: str) -> str:
    if split in {"train", "dev"}:
        return "dev_500_subsampled.jsonl"
    if split in {"validation", "test"}:
        return "test_subsampled.jsonl"
    raise ValueError(f"Unsupported split: {split}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_qid(row: Dict[str, Any]) -> str:
    for key in ("question_id", "id", "_id", "qid", "source_id"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    raise KeyError(f"Could not find question id in keys: {sorted(row.keys())}")


def get_question(row: Dict[str, Any]) -> str:
    for key in ("question_text", "question", "query"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return ""


def load_examples(processed_root: Path, split: str, datasets: Iterable[str]) -> List[Dict[str, Any]]:
    filename = split_to_processed_filename(split)
    examples = []
    for dataset in datasets:
        path = processed_root / dataset / filename
        if not path.exists():
            raise FileNotFoundError(path)

        for row in read_jsonl(path):
            examples.append({
                "id": get_qid(row),
                "dataset_name": dataset,
                "question": get_question(row),
                "source_file": str(path),
            })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--split", default="validation", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--out-root", default="router/router_predictions/deberta_v3_large/flan_t5_xl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-template", default="{question}")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.max_length, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    examples = load_examples(Path(args.processed_root), args.split, args.datasets)
    predictions = []

    with torch.no_grad():
        for start in tqdm(range(0, len(examples), args.batch_size), desc="Routing"):
            batch = examples[start:start + args.batch_size]
            texts = [
                args.text_template.format(question=" ".join(ex["question"].split()))
                for ex in batch
            ]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).detach().cpu().tolist()
            probs_cpu = probs.detach().cpu().tolist()

            for ex, pred_id, prob in zip(batch, pred_ids, probs_cpu):
                label = ID2LABEL[int(pred_id)]
                predictions.append({
                    **ex,
                    "router_label": label,
                    "router_system": LABEL_TO_SYSTEM[label],
                    "scores": {
                        "A": float(prob[0]),
                        "B": float(prob[1]),
                        "C": float(prob[2]),
                    },
                })

    out_root = Path(args.out_root) / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    route_id_dir = out_root / "route_ids"
    route_id_dir.mkdir(parents=True, exist_ok=True)

    with (out_root / "router_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    by_dataset = defaultdict(list)
    for row in predictions:
        by_dataset[row["dataset_name"]].append(row)

    for dataset, rows in by_dataset.items():
        with (out_root / f"{dataset}_router_predictions.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        for label in ["A", "B", "C"]:
            with (route_id_dir / f"{dataset}_{label}.txt").open("w", encoding="utf-8") as f:
                for row in rows:
                    if row["router_label"] == label:
                        f.write(row["id"] + "\n")

    summary = {
        "split": args.split,
        "num_examples": len(predictions),
        "label_counts": dict(Counter(row["router_label"] for row in predictions)),
        "system_counts": dict(Counter(row["router_system"] for row in predictions)),
        "dataset_label_counts": {
            dataset: dict(Counter(row["router_label"] for row in rows))
            for dataset, rows in by_dataset.items()
        },
    }

    with (out_root / "router_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
