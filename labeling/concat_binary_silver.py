#!/usr/bin/env python3
"""
Concatenate binary + silver router data for both train and validation.

Official behavior preserved:
  - remove binary records whose id appears in silver
  - append silver records
  - optionally truncate silver to a minimum length

Expected inputs:
  binary:
    .../binary/total_data_train.json
    .../binary/total_data_validation.json
  silver:
    .../{model}/silver/train.json
    .../{model}/silver/validation.json

Outputs:
  .../{model}/train.json
  .../{model}/validation.json

Usage:
  python scripts/adaptive_labeling/concat_binary_silver_train_val_official.py \
    --model flan_t5_xl \
    --root classifier/data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from labeling.preprocess_utils import concat_binary_and_silver, load_json, save_json


def count_labels(records: list[dict]) -> dict[str, int]:
    counts = {}
    for row in records:
        label = row.get("answer", "")
        counts[label] = counts.get(label, 0) + 1
    return counts


def concat_split(
    split: str,
    root: Path,
    model: str,
    silver_limit: int | None,
) -> None:
    binary_file = root / "binary" / f"total_data_{split}.json"
    silver_file = root / model / "silver" / f"{split}.json"
    output_file = root / model / f"{split}.json"

    if not binary_file.exists():
        raise FileNotFoundError(binary_file)
    if not silver_file.exists():
        raise FileNotFoundError(silver_file)

    binary = load_json(binary_file)
    silver = load_json(silver_file)

    final = concat_binary_and_silver(binary, silver, silver_limit=silver_limit)
    save_json(output_file, final)

    print()
    print(f"{split}:")
    print("  binary:", len(binary), count_labels(binary))
    print("  silver:", len(silver), count_labels(silver))
    print("  final :", len(final), count_labels(final))
    print("  out   :", output_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["flan_t5_xl", "flan_t5_xxl", "gpt"])
    parser.add_argument("--root", default="classifier/data/")
    parser.add_argument("--train-silver-limit", type=int, default=None)
    parser.add_argument("--validation-silver-limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)

    concat_split(
        split="train",
        root=root,
        model=args.model,
        silver_limit=args.train_silver_limit,
    )

    concat_split(
        split="validation",
        root=root,
        model=args.model,
        silver_limit=args.validation_silver_limit,
    )


if __name__ == "__main__":
    main()
