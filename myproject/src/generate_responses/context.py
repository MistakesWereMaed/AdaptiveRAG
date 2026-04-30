from __future__ import annotations

from typing import Iterable, List, Sequence

from myproject.src.schemas import RetrievedDocument


def format_passage(doc: RetrievedDocument, idx: int) -> str:
    title = doc.title.strip()
    if title:
        return f"[{idx}] Title: {title}\n{doc.text.strip()}"
    return f"[{idx}] Title: {doc.doc_id}\n{doc.text.strip()}"


def build_context(docs: Sequence[RetrievedDocument], max_docs: int = 6) -> str:
    selected = list(docs[:max_docs])
    return "\n\n".join(format_passage(doc, i + 1) for i, doc in enumerate(selected))


def _dedupe_key(doc: RetrievedDocument) -> str:
    if doc.doc_id:
        return doc.doc_id
    return f"{doc.title.strip()}::{doc.text.strip()[:100]}"


def deduplicate_documents(docs: Iterable[RetrievedDocument]) -> List[RetrievedDocument]:
    seen = set()
    unique_docs: List[RetrievedDocument] = []
    for doc in docs:
        key = _dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs
