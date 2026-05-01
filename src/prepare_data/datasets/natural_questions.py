from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm.auto import tqdm

from src.prepare_datasets.common import answer_list, document, first_answer


def load_natural_questions_split(
    split: str,
    dataset_name: str = "natural_questions",
    config_name: str | None = None,
):
    # Some environments use the canonical HF dataset. Others use local/processed
    # NQ files. Keep this loader configurable from the calling script.
    if config_name:
        return load_dataset(dataset_name, config_name, split=split)
    return load_dataset(dataset_name, split=split)


def _extract_context(record: Dict[str, Any]) -> str:
    # Supports several common NQ schemas.
    if isinstance(record.get("context"), str):
        return record["context"]

    if isinstance(record.get("document"), dict):
        doc = record["document"]
        if isinstance(doc.get("tokens"), dict) and "token" in doc["tokens"]:
            return " ".join(str(t) for t in doc["tokens"]["token"])
        if isinstance(doc.get("html"), str):
            return doc["html"]
        if isinstance(doc.get("text"), str):
            return doc["text"]

    if isinstance(record.get("long_answer_candidates"), list):
        texts = [
            str(x.get("text", "")).strip()
            for x in record["long_answer_candidates"]
            if isinstance(x, dict) and str(x.get("text", "")).strip()
        ]
        if texts:
            return "\n".join(texts[:5])

    return ""


def _extract_answer(record: Dict[str, Any]) -> List[str]:
    for key in ("answers", "answer", "short_answers"):
        vals = answer_list(record.get(key))
        if vals:
            return vals

    annotations = record.get("annotations")
    if isinstance(annotations, dict):
        vals = answer_list(annotations.get("short_answers"))
        if vals:
            return vals
    if isinstance(annotations, list):
        vals = answer_list(annotations)
        if vals:
            return vals

    return []


def record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    answers = _extract_answer(record)
    answer = answers[0] if answers else ""

    return {
        "source_id": str(record.get("id") or record.get("example_id") or record.get("question_id") or ""),
        "question": str(record.get("question") or record.get("question_text") or "").strip(),
        "answer": answer,
        "answers": answers,
        "gold": answer,
        "context_documents": [
            document(
                title=record.get("title", record.get("document_title", "")),
                text=_extract_context(record),
                paragraph_index=0,
            )
        ],
        "type": "single-hop",
        "dataset": "natural_questions",
    }


def dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record_to_example(record)
        for record in tqdm(dataset, desc="Converting Natural Questions", unit="example")
    ]
