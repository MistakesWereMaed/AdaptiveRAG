#!/usr/bin/env python3
"""
Evaluate routed AdaptiveRAG predictions with repo-compatible metrics.

This version fixes a non-official evaluation crash for single-hop datasets.

Problem in repo evaluate.py:
  evaluate_by_dicts() converts answer predictions to a list, then calls
  SquadAnswerEmF1(prediction, [ground_truth]).
  In this environment, SquadAnswerEmF1 expects predicted_answer to be a string,
  so it crashes with:
      AttributeError: 'list' object has no attribute 'find'

Fix:
  - For --official, still call official_evaluate_by_dicts from evaluate.py.
  - For non-official evaluation, use the same metric classes and answer_extractor
    from evaluate.py, but pass a string to SquadAnswerEmF1 for single-hop data.

Run from repo root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from evaluate import (
    answer_extractor,
    official_evaluate_by_dicts,
    load_experiment_config,
    load_ground_truths,
)
from metrics.drop_answer_em_f1 import DropAnswerEmAndF1
from metrics.support_em_f1 import SupportEmF1Metric
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric

DATASETS = ["musique", "2wikimultihopqa", "hotpotqa", "nq", "trivia", "squad"]
MULTI_HOP_DATASETS = {"hotpotqa", "2wikimultihopqa", "musique", "iirc"}


def split_to_suffix(split: str) -> str:
    if split in {"train", "dev"}:
        return "dev_500_subsampled"
    if split in {"validation", "test"}:
        return "test_subsampled"
    raise ValueError(split)


def split_to_repo_set_name(split: str) -> str:
    if split in {"train", "dev"}:
        return "dev_500"
    if split in {"validation", "test"}:
        return "test"
    raise ValueError(split)


def split_to_eval_file(processed_root: Path, dataset: str, split: str) -> Path:
    if split in {"train", "dev"}:
        return processed_root / dataset / "dev_500_subsampled.jsonl"
    if split in {"validation", "test"}:
        return processed_root / dataset / "test_subsampled.jsonl"
    raise ValueError(split)


def system_dir_name(system: str, model: str, dataset: str) -> str:
    if system == "ircot_qa":
        return f"ircot_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1"
    if system == "oner_qa":
        return f"oner_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1"
    if system == "nor_qa":
        return f"nor_qa_{model}_{dataset}____prompt_set_1"
    raise ValueError(system)


def config_path_for_dataset(
    model: str,
    dataset: str,
    split: str,
    predictions_root: Path,
    preferred_system: str,
) -> str:
    repo_set = split_to_repo_set_name(split)
    suffix = split_to_suffix(split)

    systems = []
    for system in [preferred_system, "oner_qa", "nor_qa", "ircot_qa"]:
        if system not in systems:
            systems.append(system)

    tried = []
    for system in systems:
        exp_name = system_dir_name(system, model, dataset)
        run_dir = predictions_root / repo_set / exp_name

        exact = run_dir / f"config__{dataset}_to_{dataset}__{suffix}.jsonnet"
        tried.append(exact)
        if exact.exists():
            return str(exact)

        if run_dir.exists():
            matches = sorted(run_dir.glob(f"config__*__{suffix}.jsonnet"))
            tried.extend(matches)
            if matches:
                return str(matches[0])

    raise FileNotFoundError(
        "Could not find generated config for "
        f"dataset={dataset}, split={split}, model={model}. Tried:\n"
        + "\n".join(str(p) for p in tried[:50])
    )


def routed_prediction_file(prediction_root: Path, model: str, split: str, dataset: str) -> Path:
    suffix = split_to_suffix(split)
    return prediction_root / model / split / dataset / f"prediction__{dataset}_adaptive_rag__{suffix}.json"


def load_predictions(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected qid->prediction dict: {path}")
    return {str(k): v for k, v in payload.items()}


def to_prediction_string(prediction: Any) -> str:
    if isinstance(prediction, list):
        if len(prediction) == 1:
            return str(prediction[0])
        return " ".join(str(x) for x in prediction)
    return str(prediction)


def to_answer_list(prediction: Any) -> list[str]:
    if isinstance(prediction, str):
        stripped = prediction.strip()
        if stripped.startswith("[") or stripped.endswith("]"):
            return [
                e.strip()
                for e in stripped.replace('"', "").replace("[", "").replace("]", "").split(",")
                if e.strip()
            ]
        return [prediction]
    if isinstance(prediction, (list, tuple)):
        return [str(x) for x in prediction]
    return [str(prediction)]


def evaluate_by_dicts_compatible(
    prediction_type: str,
    id_to_ground_truths: Dict[str, Any],
    id_to_predictions: Dict[str, Any],
    dataset: str,
) -> Dict[str, Any]:
    """
    Repo-compatible non-official evaluation with fixed single-hop Squad metric call.
    """
    if prediction_type != "answer":
        raise NotImplementedError(
            "This routed evaluator currently supports answer predictions. "
            f"Got prediction_type={prediction_type}"
        )

    if dataset in MULTI_HOP_DATASETS:
        answer_metric = DropAnswerEmAndF1()
        support_metric = SupportEmF1Metric(do_normalize_answer=True)

        for qid in set(id_to_ground_truths.keys()):
            ground_truth = id_to_ground_truths[qid]
            prediction = [answer_extractor(x) for x in to_answer_list(id_to_predictions[qid])]

            answer_metric(prediction, [ground_truth])
            support_metric(prediction, ground_truth)

        results = answer_metric.get_metric()
        support_results = support_metric.get_metric()
        results["sp_em"] = support_results["title_em"]
        results["sp_f1"] = support_results["title_f1"]
        results["sp_precision"] = support_results["title_precision"]
        results["sp_recall"] = support_results["title_recall"]
        return results

    # Single-hop datasets: SquadAnswerEmF1Metric expects predicted_answer as str.
    answer_metric = SquadAnswerEmF1Metric()
    support_metric = SupportEmF1Metric(do_normalize_answer=True)

    for qid in set(id_to_ground_truths.keys()):
        ground_truth = id_to_ground_truths[qid]
        prediction_str = answer_extractor(to_prediction_string(id_to_predictions[qid]))

        # Try the most likely expected shape first.
        try:
            answer_metric(prediction_str, ground_truth)
        except Exception:
            answer_metric(prediction_str, [ground_truth])

        # Keep support metric behavior close to repo, but protect against odd gt shapes.
        try:
            support_metric([prediction_str], ground_truth)
        except Exception:
            support_metric([prediction_str], [ground_truth])

    results = answer_metric.get_metric()
    support_results = support_metric.get_metric()
    results["sp_em"] = support_results["title_em"]
    results["sp_f1"] = support_results["title_f1"]
    results["sp_precision"] = support_results["title_precision"]
    results["sp_recall"] = support_results["title_recall"]
    return results


def evaluate_dataset(
    dataset: str,
    model: str,
    split: str,
    processed_root: Path,
    routed_prediction_root: Path,
    original_predictions_root: Path,
    llm_port_num: str,
    official: bool,
    config_system: str,
) -> Dict[str, Any]:
    eval_file = split_to_eval_file(processed_root, dataset, split)
    pred_file = routed_prediction_file(routed_prediction_root, model, split, dataset)

    if not eval_file.exists():
        raise FileNotFoundError(eval_file)

    config_path = config_path_for_dataset(
        model=model,
        dataset=dataset,
        split=split,
        predictions_root=original_predictions_root,
        preferred_system=config_system,
    )

    class Args:
        pass

    args = Args()
    args.llm_port_num = llm_port_num

    experiment_config = load_experiment_config(config_path, args)
    prediction_type = experiment_config["prediction_type"]

    id_to_ground_truths = load_ground_truths(
        experiment_config=experiment_config,
        ground_truth_file_path=str(eval_file),
    )
    id_to_predictions = load_predictions(pred_file)

    gt_ids = set(id_to_ground_truths)
    pred_ids = set(id_to_predictions)
    if gt_ids != pred_ids:
        missing = sorted(gt_ids - pred_ids)
        extra = sorted(pred_ids - gt_ids)
        raise ValueError(
            f"ID mismatch for {dataset}: missing={len(missing)}, extra={len(extra)}, "
            f"missing_sample={missing[:5]}, extra_sample={extra[:5]}"
        )

    if official:
        results = official_evaluate_by_dicts(
            prediction_type=prediction_type,
            id_to_predictions=id_to_predictions,
            id_to_ground_truths=id_to_ground_truths,
            dataset=dataset,
        )
    else:
        results = evaluate_by_dicts_compatible(
            prediction_type=prediction_type,
            id_to_ground_truths=id_to_ground_truths,
            id_to_predictions=id_to_predictions,
            dataset=dataset,
        )

    results = dict(results)
    results["dataset"] = dataset
    results["prediction_type"] = prediction_type
    results["count"] = int(results.get("count", len(id_to_predictions)))
    results["prediction_file"] = str(pred_file)
    results["evaluation_file"] = str(eval_file)
    results["config_used"] = str(config_path)
    return results


def weighted_average(per_dataset: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(int(m.get("count", 0)) for m in per_dataset.values())
    out = {"count": total}

    keys = sorted({
        k
        for m in per_dataset.values()
        for k, v in m.items()
        if isinstance(v, (int, float)) and k != "count"
    })

    for key in keys:
        numerator = 0.0
        denom = 0
        for m in per_dataset.values():
            value = m.get(key)
            count = int(m.get("count", 0))
            if isinstance(value, (int, float)):
                numerator += value * count
                denom += count
        if denom:
            out[key] = numerator / denom

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="folder token, e.g. flan_t5_xl")
    parser.add_argument("--split", default="validation", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--prediction-root", default="predictions/adaptive_rag")
    parser.add_argument("--original-predictions-root", default="predictions")
    parser.add_argument("--out", default=None)
    parser.add_argument("--llm-port-num", default="8010")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--config-system", default="oner_qa", choices=["nor_qa", "oner_qa", "ircot_qa"])
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    routed_prediction_root = Path(args.prediction_root)
    original_predictions_root = Path(args.original_predictions_root)

    per_dataset = {}
    for dataset in args.datasets:
        result = evaluate_dataset(
            dataset=dataset,
            model=args.model,
            split=args.split,
            processed_root=processed_root,
            routed_prediction_root=routed_prediction_root,
            original_predictions_root=original_predictions_root,
            llm_port_num=args.llm_port_num,
            official=args.official,
            config_system=args.config_system,
        )
        per_dataset[dataset] = result
        print(json.dumps(result, indent=2))

    summary = {
        "model": args.model,
        "split": args.split,
        "official": args.official,
        "weighted_average": weighted_average(per_dataset),
        "per_dataset": per_dataset,
    }

    out_path = (
        Path(args.out)
        if args.out
        else routed_prediction_root / args.model / args.split / (
            "official_evaluation_summary.json" if args.official else "evaluation_summary.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Routed AdaptiveRAG Summary ===")
    print(json.dumps(summary["weighted_average"], indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
