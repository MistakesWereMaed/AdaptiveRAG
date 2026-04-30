from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm.auto import tqdm


def load_hotpotqa_split(split: str, config_name: str = "distractor"):
    """Load a HotpotQA split from Hugging Face."""
    return load_dataset("hotpot_qa", config_name, split=split)


def _join_sentences(sentences: Any) -> str:
    """Join a HotpotQA sentence list into one paragraph string."""
    if isinstance(sentences, str):
        return sentences.strip()

    if isinstance(sentences, list):
        return " ".join(str(s).strip() for s in sentences if str(s).strip())

    return ""


def context_to_documents(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Hugging Face HotpotQA context into title-aware paragraph docs.

    Expected HF schema:
        context = {
            "title": [title_1, title_2, ...],
            "sentences": [[sent_1, sent_2, ...], ...]
        }
    """
    titles = context.get("title", [])
    sentence_groups = context.get("sentences", [])

    docs: List[Dict[str, Any]] = []
    for idx, (title, sentences) in enumerate(zip(titles, sentence_groups)):
        text = _join_sentences(sentences)
        if not text:
            continue

        docs.append(
            {
                "title": str(title).strip(),
                "text": text,
                "paragraph_index": idx,
            }
        )

    return docs


def supporting_titles(supporting_facts: Any) -> List[str]:
    """Extract unique supporting titles from Hugging Face HotpotQA records."""
    titles = supporting_facts.get("title", []) if isinstance(supporting_facts, dict) else []

    seen = set()
    result = []
    for title in titles:
        title = str(title).strip()
        if title and title not in seen:
            seen.add(title)
            result.append(title)

    return result


def record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    docs = context_to_documents(record.get("context", {}))
    answer = record.get("answer", "")

    return {
        "id": record.get("id", record.get("_id")),
        "question": record.get("question", ""),
        "answer": answer,
        "gold": answer,
        "context_documents": docs,
        "supporting_facts": record.get("supporting_facts", {}),
        "supporting_titles": supporting_titles(record.get("supporting_facts", {})),
        "type": record.get("type"),
        "level": record.get("level"),
    }


def dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        record_to_example(record)
        for record in tqdm(dataset, desc="Converting HotpotQA", unit="example")
    ]


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
