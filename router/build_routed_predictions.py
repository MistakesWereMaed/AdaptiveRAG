#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

LABEL_TO_SYSTEM = {"A": "nor_qa", "B": "oner_qa", "C": "ircot_qa"}


def split_to_pred_dir_and_suffix(split: str) -> tuple[str, str]:
    if split in {"train", "dev"}:
        return "dev_500", "dev_500_subsampled"
    if split in {"validation", "test"}:
        return "test", "test_subsampled"
    raise ValueError(f"Unsupported split: {split}")


def system_dir_name(system: str, model: str, dataset: str) -> str:
    if system == "ircot_qa":
        return f"ircot_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1"
    if system == "oner_qa":
        return f"oner_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1"
    if system == "nor_qa":
        return f"nor_qa_{model}_{dataset}____prompt_set_1"
    raise ValueError(system)


def prediction_path(predictions_root: Path, model: str, dataset: str, system: str, split: str) -> Path:
    pred_dir, suffix = split_to_pred_dir_and_suffix(split)
    run_dir = predictions_root / pred_dir / system_dir_name(system, model, dataset)
    return run_dir / f"prediction__{dataset}_to_{dataset}__{suffix}.json"


def load_prediction_dict(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected qid->prediction dict: {path}")
    return {str(k): str(v) for k, v in payload.items()}


def load_router_predictions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected router prediction list: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Repo folder model token, e.g. flan_t5_xl")
    parser.add_argument("--split", default="validation", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--router-predictions", required=True)
    parser.add_argument("--predictions-root", default="predictions")
    parser.add_argument("--out-root", default="predictions/adaptive_rag")
    args = parser.parse_args()

    predictions_root = Path(args.predictions_root)
    router_rows = load_router_predictions(Path(args.router_predictions))
    _, suffix = split_to_pred_dir_and_suffix(args.split)

    by_dataset = defaultdict(list)
    for row in router_rows:
        by_dataset[row["dataset_name"]].append(row)

    all_routed = {}
    all_metadata = []
    missing = []

    for dataset, rows in by_dataset.items():
        strategy_predictions = {
            system: load_prediction_dict(prediction_path(predictions_root, args.model, dataset, system, args.split))
            for system in ["nor_qa", "oner_qa", "ircot_qa"]
        }

        routed_prediction = {}
        metadata = []

        for row in rows:
            qid = str(row["id"])
            label = row["router_label"]
            system = LABEL_TO_SYSTEM.get(label, "oner_qa")
            pred = strategy_predictions[system].get(qid)
            if pred is None:
                missing.append({"dataset": dataset, "qid": qid, "label": label, "system": system})
                pred = ""
            routed_prediction[qid] = pred
            all_routed[qid] = pred
            metadata.append({
                "id": qid,
                "dataset_name": dataset,
                "question": row.get("question", ""),
                "router_label": label,
                "router_system": system,
                "selected_prediction": pred,
            })

        out_dir = Path(args.out_root) / args.model / args.split / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_file = out_dir / f"prediction__{dataset}_adaptive_rag__{suffix}.json"
        meta_file = out_dir / f"routing_metadata__{dataset}_adaptive_rag__{suffix}.json"
        with pred_file.open("w", encoding="utf-8") as f:
            json.dump(routed_prediction, f, indent=2, sort_keys=True, ensure_ascii=False)
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        all_metadata.extend(metadata)
        print(f"{dataset}: wrote {len(routed_prediction)} routed predictions -> {pred_file}")
        print(f"{dataset}: route counts {dict(Counter(x['router_label'] for x in metadata))}")

    out_split_dir = Path(args.out_root) / args.model / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    with (out_split_dir / "all_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(all_routed, f, indent=2, sort_keys=True, ensure_ascii=False)
    with (out_split_dir / "all_routing_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    summary = {
        "model": args.model,
        "split": args.split,
        "num_predictions": len(all_routed),
        "route_counts": dict(Counter(row["router_label"] for row in all_metadata)),
        "system_counts": dict(Counter(row["router_system"] for row in all_metadata)),
        "missing_selected_predictions": len(missing),
        "missing_examples": missing[:20],
    }
    with (out_split_dir / "routing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected strategy predictions. See routing_summary.json.")


if __name__ == "__main__":
    main()
