from __future__ import annotations

from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm.auto import tqdm

from src.prepare_datasets.common import answer_list, document, first_answer


def load_twowiki_split(
    split: str,
    dataset_name: str = "voidful/2WikiMultihopQA",
    config_name: str | None = None,
):
    if config_name:
        return load_dataset(dataset_name, config_name, split=split)
    return load_dataset(dataset_name, split=split)


def _extract_context(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    context = record.get("context") or record.get("contexts") or []
    docs: List[Dict[str, Any]] = []

    if isinstance(context, list):
        for idx, item in enumerate(context):
            if isinstance(item, dict):
                docs.append(
                    document(
                        title=item.get("title", ""),
                        text=item.get("text") or item.get("context") or item.get("sentences") or "",
                        paragraph_index=idx,
                    )
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                text = item[1]
                if isinstance(text, list):
                    text = " ".join(str(s).strip() for s in text if str(s).strip())
                docs.append(document(title=title, text=text, paragraph_index=idx))
            elif isinstance(item, str):
                docs.append(document(text=item, paragraph_index=idx))

    return [d for d in docs if d["text"]]


def _supporting_titles(record: Dict[str, Any]) -> List[str]:
    supporting_facts = record.get("supporting_facts") or []
    titles: List[str] = []

    if isinstance(supporting_facts, list):
        for fact in supporting_facts:
            title = ""
            if isinstance(fact, dict):
                title = str(fact.get("title", "")).strip()
            elif isinstance(fact, (list, tuple)) and fact:
                title = str(fact[0]).strip()
            if title and title not in titles:
                titles.append(title)

    return titles


def record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    answers = answer_list(record.get("answer") or record.get("answers"))
    answer = answers[0] if answers else first_answer(record.get("answer"))

    return {
        "source_id": str(record.get("_id") or record.get("id") or ""),
        "question": str(record.get("question", "")).strip(),
        "answer": answer,
        "answers": answers or ([answer] if answer else []),
        "gold": answer,
        "context_documents": _extract_context(record),
        "supporting_facts": record.get("supporting_facts"),
        "supporting_titles": _supporting_titles(record),
        "type": "multi-hop",
        "dataset": "twowikimultihopqa",
    }


def dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record_to_example(record)
        for record in tqdm(dataset, desc="Converting 2WikiMultiHopQA", unit="example")
    ]
