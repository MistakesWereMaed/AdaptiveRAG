#!/usr/bin/env python3
"""
Build silver router training data from official AdaptiveRAG
zero_single_multi_classification__*.json files.

This replaces scoring raw prediction__*.json files.

Usage:
  python scripts/adaptive_labeling/preprocess_silver_train_official.py \
    --model flan_t5_xl \
    --processed-root processed_data \
    --predictions-root predictions \
    --out classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train.json

Model names should use the repo folder convention:
  flan_t5_xl
  flan_t5_xxl
  gpt
"""

from __future__ import annotations

import argparse
import os
from labeling.preprocess_utils import (
    label_complexity_from_classification_files,
    save_json,
    ALL_DATASETS,
)

def get_paths(pred_root, model, dataset, split):
    if split == "train":
        pred_dir = "dev_500"
        suffix = "dev_500_subsampled"
    elif split == "validation":
        pred_dir = "test"
        suffix = "test_subsampled"
    else:
        raise ValueError(split)

    def sys_dir(system):
        if system == "ircot_qa":
            return f"ircot_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1"
        if system == "oner_qa":
            return f"oner_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1"
        if system == "nor_qa":
            return f"nor_qa_{model}_{dataset}____prompt_set_1"

    base = os.path.join(pred_root, pred_dir)

    zero = os.path.join(base, sys_dir("nor_qa"),
        f"zero_single_multi_classification__{dataset}_to_{dataset}__{suffix}.json")

    one = os.path.join(base, sys_dir("oner_qa"),
        f"zero_single_multi_classification__{dataset}_to_{dataset}__{suffix}.json")

    multi = os.path.join(base, sys_dir("ircot_qa"),
        f"zero_single_multi_classification__{dataset}_to_{dataset}__{suffix}.json")

    return zero, one, multi


def get_processed_file(processed_root, dataset, split):
    if split == "train":
        return os.path.join(processed_root, dataset, "dev_500_subsampled.jsonl")
    elif split == "validation":
        return os.path.join(processed_root, dataset, "test_subsampled.jsonl")
    else:
        raise ValueError(split)


def build_split(model, split, processed_root, pred_root, out_dir):
    all_data = []

    for dataset in ALL_DATASETS:
        orig_file = get_processed_file(processed_root, dataset, split)
        zero, one, multi = get_paths(pred_root, model, dataset, split)

        for p in [orig_file, zero, one, multi]:
            if not os.path.exists(p):
                raise FileNotFoundError(p)

        data = label_complexity_from_classification_files(
            orig_file,
            zero,
            one,
            multi,
            dataset
        )

        print(f"{dataset} ({split}): {len(data)}")
        all_data.extend(data)

    save_json(os.path.join(out_dir, f"{split}.json"), all_data)

    counts = {}
    for x in all_data:
        counts[x["answer"]] = counts.get(x["answer"], 0) + 1

    print(f"{split} total:", len(all_data))
    print("label counts:", counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--predictions-root", default="predictions")
    parser.add_argument("--out-dir", default="classifier/data/silver/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    build_split(args.model, "train", args.processed_root, args.predictions_root, args.out_dir)
    build_split(args.model, "validation", args.processed_root, args.predictions_root, args.out_dir)


if __name__ == "__main__":
    main()