from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm.auto import tqdm

from src.prepare_data.common import answer_list, document, first_answer


def load_musique_split(
    split: str,
    dataset_name: str = "dgslibisey/MuSiQue",
    config_name: str | None = None,
):
    if config_name:
        return load_dataset(dataset_name, config_name, split=split)
    return load_dataset(dataset_name, split=split)


def _extract_paragraphs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    paragraphs = record.get("paragraphs") or record.get("contexts") or record.get("context") or []
    docs: List[Dict[str, Any]] = []

    if isinstance(paragraphs, list):
        for idx, para in enumerate(paragraphs):
            if isinstance(para, dict):
                docs.append(
                    document(
                        title=para.get("title", ""),
                        text=para.get("paragraph_text") or para.get("text") or para.get("context") or "",
                        paragraph_index=idx,
                        is_supporting=para.get("is_supporting"),
                    )
                )
            elif isinstance(para, str):
                docs.append(document(text=para, paragraph_index=idx))

    return [d for d in docs if d["text"]]


def _supporting_titles(record: Dict[str, Any]) -> List[str]:
    titles: List[str] = []
    for para in record.get("paragraphs", []) or []:
        if isinstance(para, dict) and para.get("is_supporting"):
            title = str(para.get("title", "")).strip()
            if title and title not in titles:
                titles.append(title)
    return titles


def record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    answers = answer_list(record.get("answer") or record.get("answers"))
    answer = answers[0] if answers else first_answer(record.get("answer"))

    return {
        "source_id": str(record.get("id") or record.get("_id") or ""),
        "question": str(record.get("question", "")).strip(),
        "answer": answer,
        "answers": answers or ([answer] if answer else []),
        "gold": answer,
        "context_documents": _extract_paragraphs(record),
        "supporting_titles": _supporting_titles(record),
        "question_decomposition": record.get("question_decomposition"),
        "type": "multi-hop",
        "dataset": "musique",
    }


def dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record_to_example(record)
        for record in tqdm(dataset, desc="Converting MuSiQue", unit="example")
    ]
