from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
from datasets import load_dataset
from tqdm.auto import tqdm


def load_hotpotqa_split(split: str = "train", config_name: str = "distractor"):
    return load_dataset("hotpot_qa", config_name, split=split)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else []


def _join_sentences(sentences: Any) -> str:
    if isinstance(sentences, str):
        return sentences.strip()

    if isinstance(sentences, list):
        return " ".join(str(sentence).strip() for sentence in sentences if str(sentence).strip())

    return ""


def hotpotqa_context_to_structured_documents(context: Any) -> List[Dict[str, Any]]:
    """
    Convert HotpotQA context into paragraph-level structured documents.

    Handles the Hugging Face schema:

        context = {
            "title": [...],
            "sentences": [[...], [...], ...]
        }

    Returns:
        [
            {
                "title": str,
                "text": str,
                "sentences": list[str],
                "paragraph_index": int
            }
        ]
    """
    documents: List[Dict[str, Any]] = []

    if context is None:
        return documents

    # Hugging Face hotpot_qa schema.
    if isinstance(context, dict):
        titles = context.get("title")
        sentence_groups = context.get("sentences")

        if isinstance(titles, list) and isinstance(sentence_groups, list):
            for paragraph_index, (title, sentences) in enumerate(zip(titles, sentence_groups)):
                title_text = str(title).strip() if title is not None else ""

                if isinstance(sentences, list):
                    clean_sentences = [
                        str(sentence).strip()
                        for sentence in sentences
                        if str(sentence).strip()
                    ]
                elif isinstance(sentences, str):
                    clean_sentences = [sentences.strip()] if sentences.strip() else []
                else:
                    clean_sentences = []

                text = " ".join(clean_sentences).strip()

                if title_text or text:
                    documents.append(
                        {
                            "title": title_text,
                            "text": text,
                            "sentences": clean_sentences,
                            "paragraph_index": paragraph_index,
                        }
                    )

            return documents

        # Fallback for other dict-like schemas.
        for key in ("paragraphs", "documents", "passages"):
            value = context.get(key)
            if isinstance(value, list):
                documents.extend(hotpotqa_context_to_structured_documents(value))

        return documents

    # Fallback schema: list of [title, sentences] pairs.
    if isinstance(context, list):
        for paragraph_index, entry in enumerate(context):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                title, sentences = entry
                title_text = str(title).strip() if title is not None else ""

                if isinstance(sentences, list):
                    clean_sentences = [
                        str(sentence).strip()
                        for sentence in sentences
                        if str(sentence).strip()
                    ]
                elif isinstance(sentences, str):
                    clean_sentences = [sentences.strip()] if sentences.strip() else []
                else:
                    clean_sentences = []

                text = " ".join(clean_sentences).strip()

                if title_text or text:
                    documents.append(
                        {
                            "title": title_text,
                            "text": text,
                            "sentences": clean_sentences,
                            "paragraph_index": paragraph_index,
                        }
                    )

            elif isinstance(entry, dict):
                documents.extend(hotpotqa_context_to_structured_documents(entry))

            elif isinstance(entry, str) and entry.strip():
                documents.append(
                    {
                        "title": "",
                        "text": entry.strip(),
                        "sentences": [entry.strip()],
                        "paragraph_index": paragraph_index,
                    }
                )

        return documents

    # Fallback for plain string context.
    if isinstance(context, str):
        for paragraph_index, segment in enumerate(context.split("\n")):
            text = segment.strip()
            if text:
                documents.append(
                    {
                        "title": "",
                        "text": text,
                        "sentences": [text],
                        "paragraph_index": paragraph_index,
                    }
                )

    return documents


def hotpotqa_context_to_documents(context: Any) -> List[str]:
    """
    Backward-compatible plain-text view of context.

    Prefer `hotpotqa_context_to_structured_documents` for retrieval.
    """
    structured = hotpotqa_context_to_structured_documents(context)

    documents = []
    for doc in structured:
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()

        if title and text:
            documents.append(f"{title}: {text}")
        elif text:
            documents.append(text)
        elif title:
            documents.append(title)

    return documents


def _extract_supporting_titles(supporting_facts: Any) -> List[str]:
    titles: List[str] = []
    seen = set()

    # Hugging Face schema:
    # supporting_facts = {"title": [...], "sent_id": [...]}
    if isinstance(supporting_facts, dict):
        raw_titles = supporting_facts.get("title", [])
        for title in raw_titles:
            title_text = str(title).strip()
            if title_text and title_text not in seen:
                seen.add(title_text)
                titles.append(title_text)
        return titles

    # Fallback schema:
    # supporting_facts = [[title, sentence_id], ...]
    if isinstance(supporting_facts, list):
        for fact in supporting_facts:
            if isinstance(fact, (list, tuple)) and len(fact) >= 1:
                title_text = str(fact[0]).strip()
                if title_text and title_text not in seen:
                    seen.add(title_text)
                    titles.append(title_text)

    return titles


def hotpotqa_record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_context = record.get("context")
    structured_documents = hotpotqa_context_to_structured_documents(raw_context)

    context_documents = []
    for doc in structured_documents:
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()

        if title and text:
            context_documents.append(
                {
                    "title": title,
                    "text": text,
                    "paragraph_index": doc.get("paragraph_index"),
                }
            )
        elif text:
            context_documents.append(
                {
                    "title": "",
                    "text": text,
                    "paragraph_index": doc.get("paragraph_index"),
                }
            )

    supporting_facts = record.get("supporting_facts", [])
    supporting_titles = _extract_supporting_titles(supporting_facts)

    context_text = "\n\n".join(
        f"Title: {doc['title']}\n{doc['text']}" if doc["title"] else doc["text"]
        for doc in context_documents
    )

    return {
        "id": record.get("id", record.get("_id")),
        "source_id": record.get("id", record.get("_id")),
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "gold": record.get("answer", ""),
        "context": context_text,
        "raw_context": raw_context,
        "context_documents": context_documents,
        "supporting_facts": supporting_facts,
        "supporting_titles": supporting_titles,
        "type": record.get("type"),
        "level": record.get("level"),
    }


def hotpotqa_dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        hotpotqa_record_to_example(record)
        for record in tqdm(dataset, desc="Converting HotpotQA", unit="example")
    ]


def save_jsonl(records: Iterable[Dict[str, Any]], output_path: str | Path) -> None:
    print(f"[hotpotqa] Saving JSONL to {output_path}", flush=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")