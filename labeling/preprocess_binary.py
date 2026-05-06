#!/usr/bin/env python3
"""
Build binary/inductive-bias router data for both train and validation.

Paper-close behavior:
  train:
    - uses raw/processed training data
    - keeps up to --train-limit examples per dataset, default 400
  validation:
    - uses processed_data/{dataset}/test_subsampled.jsonl
    - keeps all available validation examples by default

Labels:
  single-hop datasets: nq, trivia, squad -> B
  multi-hop datasets: musique, 2wikimultihopqa, hotpotqa -> C

Outputs:
  classifier/data/binary/{dataset}_train.json
  classifier/data/binary/{dataset}_validation.json
  classifier/data/binary/total_data_train.json
  classifier/data/binary/total_data_validation.json

Usage:
  python scripts/adaptive_labeling/preprocess_binary_train_val_official.py \
    --processed-root processed_data \
    --out-root classifier/data/binary
"""

from __future__ import annotations

import argparse
from pathlib import Path

from labeling.preprocess_utils import (
    ALL_DATASETS,
    make_inductive_bias_from_records,
    save_json,
)


def find_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in paths))


def find_train_file(processed_root: Path, raw_root: Path, dataset: str) -> Path:
    """
    Prefer official raw training files when available, but fall back to processed files.
    This keeps the script usable in your reimplementation even if raw paths differ.
    """
    candidates = {
        "musique": [
            raw_root / "musique" / "musique_ans_v1.0_train.jsonl",
            processed_root / "musique" / "train.jsonl",
            processed_root / "musique" / "train.json",
        ],
        "2wikimultihopqa": [
            raw_root / "2wikimultihopqa" / "train.json",
            processed_root / "2wikimultihopqa" / "train.jsonl",
            processed_root / "2wikimultihopqa" / "train.json",
        ],
        "hotpotqa": [
            raw_root / "hotpotqa" / "hotpot_train_v1.1.json",
            processed_root / "hotpotqa" / "train.jsonl",
            processed_root / "hotpotqa" / "train.json",
        ],
        "nq": [
            raw_root / "nq" / "biencoder-nq-train.json",
            processed_root / "nq" / "train.jsonl",
            processed_root / "nq" / "train.json",
        ],
        "trivia": [
            raw_root / "trivia" / "biencoder-trivia-train.json",
            processed_root / "trivia" / "train.jsonl",
            processed_root / "trivia" / "train.json",
        ],
        "squad": [
            raw_root / "squad" / "biencoder-squad1-train.json",
            processed_root / "squad" / "train.jsonl",
            processed_root / "squad" / "train.json",
        ],
    }
    return find_first_existing(candidates[dataset])


def find_validation_file(processed_root: Path, dataset: str) -> Path:
    candidates = [
        processed_root / dataset / "test_subsampled.jsonl",
        processed_root / dataset / "test.jsonl",
        processed_root / dataset / "test.json",
    ]
    return find_first_existing(candidates)


def build_split(
    split: str,
    processed_root: Path,
    raw_root: Path,
    out_root: Path,
    train_limit: int,
    validation_limit: int | None,
) -> list[dict]:
    total = []

    for dataset in ALL_DATASETS:
        if split == "train":
            input_file = find_train_file(processed_root, raw_root, dataset)
            limit = train_limit
        elif split == "validation":
            input_file = find_validation_file(processed_root, dataset)
            limit = validation_limit
        else:
            raise ValueError(split)

        records = make_inductive_bias_from_records(
            input_file=input_file,
            dataset_name=dataset,
            set_name=split,
            limit=limit,
        )

        save_json(out_root / f"{dataset}_{split}.json", records)
        total.extend(records)
        print(f"{dataset} {split}: {len(records)} from {input_file}")

    save_json(out_root / f"total_data_{split}.json", total)

    counts = {}
    for row in total:
        counts[row["answer"]] = counts.get(row["answer"], 0) + 1

    print(f"{split} total:", len(total))
    print(f"{split} label counts:", counts)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--raw-root", default="raw_data")
    parser.add_argument("--out-root", default="classifier/data/binary")
    parser.add_argument("--train-limit", type=int, default=400)
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=None,
        help="Default keeps all validation examples. Set to 500 to force 500/dataset.",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    build_split(
        split="train",
        processed_root=processed_root,
        raw_root=raw_root,
        out_root=out_root,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
    )

    build_split(
        split="validation",
        processed_root=processed_root,
        raw_root=raw_root,
        out_root=out_root,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
    )


if __name__ == "__main__":
    main()
