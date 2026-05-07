#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

VALID_LABELS = {"A", "B", "C"}


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list input: {path}")
    return payload


def clean_label(text: str) -> str:
    text = str(text).strip().upper()
    return text[0] if text and text[0] in VALID_LABELS else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prompt-template", default="Question: {question} Complexity:")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=256,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()
    records = read_json_or_jsonl(Path(args.input))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    with torch.no_grad():
        for start in tqdm(range(0, len(records), args.batch_size), desc="Predicting"):
            batch_records = records[start:start + args.batch_size]
            prompts = [
                args.prompt_template.format(question=" ".join(str(row.get("question", "")).split()))
                for row in batch_records
            ]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, raw_pred in zip(batch_records, decoded):
                pred = clean_label(raw_pred)
                gold = clean_label(row.get("answer", ""))
                predictions.append({
                    "id": row.get("id"),
                    "dataset_name": row.get("dataset_name"),
                    "question": row.get("question"),
                    "gold": gold,
                    "prediction": pred,
                    "raw_prediction": raw_pred,
                    "correct": bool(gold and pred == gold),
                })
    with out_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    gold_rows = [row for row in predictions if row["gold"] in VALID_LABELS]
    if gold_rows:
        correct = sum(row["correct"] for row in gold_rows)
        print(f"Accuracy: {correct / len(gold_rows):.4f} ({correct}/{len(gold_rows)})")
        print("Gold counts:", dict(Counter(row["gold"] for row in gold_rows)))
        print("Pred counts:", dict(Counter(row["prediction"] for row in gold_rows)))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
