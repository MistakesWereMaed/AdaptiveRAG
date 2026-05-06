#!/usr/bin/env python3
"""
Utilities for paper-close AdaptiveRAG router preprocessing.

This matches the official repo's important behavior:

1. Silver labels are built from zero_single_multi_classification__*.json files,
   not from raw prediction__*.json files.
2. A question receives a silver label if its qid appears in the corresponding
   classification file:
      nor_qa   -> A / zero
      oner_qa  -> B / one
      ircot_qa -> C / multiple
3. If a qid appears in multiple files, the simplest successful strategy wins:
      A overrides B overrides C.
4. Binary/inductive-bias labels are separate:
      single-hop datasets -> B
      multi-hop datasets  -> C
5. Final training can concatenate binary + silver while removing duplicate ids
   from binary if they also appear in silver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SINGLE_HOP_DATASETS = {"nq", "trivia", "squad"}
MULTI_HOP_DATASETS = {"musique", "hotpotqa", "2wikimultihopqa"}
ALL_DATASETS = ["musique", "2wikimultihopqa", "hotpotqa", "nq", "trivia", "squad"]


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)
    print(path)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)

    payload = load_json(path)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "instances"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported input structure: {path}")


def get_question_id(record: Dict[str, Any]) -> str:
    for key in ("question_id", "id", "_id", "qid", "source_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise KeyError(f"Could not find question id in keys: {sorted(record.keys())}")


def get_question_text(record: Dict[str, Any]) -> str:
    for key in ("question_text", "question", "query"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def load_classification_qids(path: str | Path) -> set[str]:
    """
    Official zero_single_multi_classification files are JSON dictionaries keyed
    by qid. Values are not needed for labeling; membership is the signal.
    """
    payload = load_json(path)
    if isinstance(payload, dict):
        return set(str(k) for k in payload.keys())

    if isinstance(payload, list):
        qids = set()
        for record in payload:
            if isinstance(record, dict):
                qids.add(get_question_id(record))
            else:
                qids.add(str(record))
        return qids

    raise ValueError(f"Unsupported classification file structure: {path}")


def label_complexity_from_classification_files(
    orig_file: str | Path,
    zero_file: str | Path,
    one_file: str | Path,
    multi_file: str | Path,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """
    Paper-close version of official label_complexity.

    Difference from a naive scorer:
    - Does NOT compare generated answer strings to gold answers.
    - Uses the classification files produced by repo evaluation.
    - Applies simplest-success priority: zero > one > multi.

    Output fields follow official classifier data:
      id, question, answer_description, answer, dataset_name, total_answer
    """
    records = read_json_or_jsonl(orig_file)

    zero_qids = load_classification_qids(zero_file)
    one_qids = load_classification_qids(one_file)
    multi_qids = load_classification_qids(multi_file)

    labeled = []

    for record in records:
        qid = get_question_id(record)

        if qid not in zero_qids and qid not in one_qids and qid not in multi_qids:
            continue

        total_answer = []
        label = ""
        answer_description = ""

        # Preserve official total_answer ordering from broad to simple.
        if qid in multi_qids:
            total_answer.append("multiple")
        if qid in one_qids:
            total_answer.append("one")
        if qid in zero_qids:
            total_answer.append("zero")

        # But make priority explicit and less error-prone:
        # simplest successful strategy wins.
        if qid in zero_qids:
            label = "A"
            answer_description = "zero"
        elif qid in one_qids:
            label = "B"
            answer_description = "one"
        elif qid in multi_qids:
            label = "C"
            answer_description = "multiple"

        labeled.append(
            {
                "id": qid,
                "question": get_question_text(record),
                "answer_description": answer_description,
                "answer": label,
                "dataset_name": dataset_name,
                "total_answer": total_answer,
            }
        )

    return labeled


def prepare_predict_file(orig_file: str | Path, dataset_name: str) -> List[Dict[str, Any]]:
    records = read_json_or_jsonl(orig_file)
    return [
        {
            "id": get_question_id(record),
            "question": get_question_text(record),
            "dataset_name": dataset_name,
            "answer": "",
            "total_answer": [],
        }
        for record in records
    ]


def make_inductive_bias_from_records(
    input_file: str | Path,
    dataset_name: str,
    set_name: str,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Create binary/inductive labels.

    For processed IRCoT files, preserve question_id.
    For raw single-hop DPR files, official code used synthetic ids:
      single_{dataset}_{set}_{idx}
    This function uses real ids if available, otherwise falls back to synthetic.
    """
    records = read_json_or_jsonl(input_file)

    if limit is not None:
        records = records[:limit]

    if dataset_name in SINGLE_HOP_DATASETS:
        label = "B"
        description = "single"
    elif dataset_name in MULTI_HOP_DATASETS:
        label = "C"
        description = "multi"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    output = []
    for idx, record in enumerate(records):
        try:
            qid = get_question_id(record)
        except KeyError:
            qid = f"single_{dataset_name}_{set_name}_{idx}"

        output.append(
            {
                "id": qid,
                "question": get_question_text(record),
                "answer_description": description,
                "answer": label,
                "dataset_name": dataset_name,
            }
        )

    return output


def concat_binary_and_silver(
    binary_records: List[Dict[str, Any]],
    silver_records: List[Dict[str, Any]],
    silver_limit: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Official behavior:
      - Remove binary examples whose id appears in silver.
      - Append silver[:min_len].
    """
    silver_ids = {record["id"] for record in silver_records}
    filtered_binary = [record for record in binary_records if record["id"] not in silver_ids]

    if silver_limit is not None:
        silver_records = silver_records[:silver_limit]

    return filtered_binary + silver_records
