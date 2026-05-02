from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm.auto import tqdm

from src.prepare_data.common import answer_list, document, first_answer


def load_triviaqa_split(
    split: str,
    dataset_name: str = "trivia_qa",
    config_name: str = "rc.nocontext",
):
    return load_dataset(dataset_name, config_name, split=split)


def _extract_answer(record: Dict[str, Any]) -> List[str]:
    answer = record.get("answer")
    if isinstance(answer, dict):
        aliases = answer_list(answer.get("aliases"))
        value = first_answer(answer.get("value"))
        normalized = first_answer(answer.get("normalized_value"))
        out = []
        for ans in [value, normalized, *aliases]:
            if ans and ans not in out:
                out.append(ans)
        return out
    return answer_list(answer)


def _extract_docs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    entity_pages = record.get("entity_pages")
    if isinstance(entity_pages, dict):
        titles = entity_pages.get("title", [])
        wiki_context = entity_pages.get("wiki_context", [])
        for idx, (title, text) in enumerate(zip(titles, wiki_context)):
            docs.append(document(title=title, text=text, paragraph_index=idx, source="entity_pages"))

    search_results = record.get("search_results")
    if isinstance(search_results, dict):
        titles = search_results.get("title", [])
        descriptions = search_results.get("description", [])
        for idx, (title, text) in enumerate(zip(titles, descriptions)):
            docs.append(document(title=title, text=text, paragraph_index=len(docs) + idx, source="search_results"))

    return [d for d in docs if d["text"]]


def record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    answers = _extract_answer(record)
    answer = answers[0] if answers else ""

    return {
        "source_id": str(record.get("question_id") or record.get("id") or ""),
        "question": str(record.get("question", "")).strip(),
        "answer": answer,
        "answers": answers,
        "gold": answer,
        "context_documents": _extract_docs(record),
        "type": "single-hop",
        "dataset": "triviaqa",
    }


def dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record_to_example(record)
        for record in tqdm(dataset, desc="Converting TriviaQA", unit="example")
    ]
