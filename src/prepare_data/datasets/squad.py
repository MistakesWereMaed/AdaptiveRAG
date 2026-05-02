from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm.auto import tqdm

from src.prepare_data.common import answer_list, document, first_answer


def load_squad_split(split: str, dataset_name: str = "squad"):
    return load_dataset(dataset_name, split=split)


def record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    answers = answer_list(record.get("answers"))
    answer = first_answer(record.get("answers"))

    return {
        "source_id": str(record.get("id", "")),
        "question": str(record.get("question", "")).strip(),
        "answer": answer,
        "answers": answers,
        "gold": answer,
        "context_documents": [
            document(
                title=record.get("title", ""),
                text=record.get("context", ""),
                paragraph_index=0,
            )
        ],
        "type": "single-hop",
        "dataset": "squad",
    }


def dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record_to_example(record)
        for record in tqdm(dataset, desc="Converting SQuAD", unit="example")
    ]
