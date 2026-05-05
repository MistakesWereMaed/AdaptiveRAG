import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from lib import read_json, read_jsonl, write_jsonl


DATASET_NAME = "nq"
SET_NAMES = ["train", "dev", "test"]
MAX_NUM_TOKENS = 1000


def _existing_input_path(input_directory: str, set_name: str) -> Optional[str]:
    """Return the first plausible raw file path for this split."""
    candidates = [
        os.path.join(input_directory, f"{set_name}.json"),
        os.path.join(input_directory, f"{set_name}.jsonl"),
        os.path.join(input_directory, f"nq-{set_name}.json"),
        os.path.join(input_directory, f"nq-{set_name}.jsonl"),
        os.path.join(input_directory, f"biencoder-nq-{set_name}.json"),
        os.path.join(input_directory, f"biencoder-nq-{set_name}.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _load_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    for key in ("data", "records", "examples"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Unsupported raw NQ structure: {path}")


def _normalize_answers(raw_instance: Dict[str, Any]) -> List[str]:
    answers = raw_instance.get("answers")
    if answers is None:
        answers = raw_instance.get("answer")
    if answers is None:
        answers = raw_instance.get("gold")

    if isinstance(answers, str):
        answers = [answers]
    elif answers is None:
        answers = []

    normalized = []
    for answer in answers:
        if answer is None:
            continue
        answer = str(answer).strip()
        if answer and answer not in normalized:
            normalized.append(answer)
    return normalized


def _context_text(raw_context: Dict[str, Any]) -> str:
    text = raw_context.get("text")
    if text is None:
        text = raw_context.get("paragraph_text")
    if text is None:
        text = raw_context.get("passage")
    if text is None:
        text = raw_context.get("contents")
    return " ".join(str(text or "").strip().split()[:MAX_NUM_TOKENS])


def _context_title(raw_context: Dict[str, Any]) -> str:
    title = raw_context.get("title")
    if title is None:
        title = raw_context.get("wikipedia_title")
    if title is None:
        title = raw_context.get("doc_title")
    return str(title or "").strip()


def _iter_contexts(raw_instance: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], bool]]:
    """Yield DPR-style contexts with support labels."""
    for context in raw_instance.get("positive_ctxs", []) or []:
        yield context, True

    # Include negatives only as distractors for gold-context/direct QA compatibility.
    for key in ("negative_ctxs", "hard_negative_ctxs"):
        for context in raw_instance.get(key, []) or []:
            yield context, False

    # Fallback for already-normalized or custom records.
    for context in raw_instance.get("contexts", []) or raw_instance.get("context_documents", []) or []:
        yield context, bool(context.get("is_supporting", context.get("has_answer", False)))


def _process_instance(raw_instance: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    question_text = str(raw_instance.get("question") or raw_instance.get("question_text") or "").strip()
    answers = _normalize_answers(raw_instance)

    if not question_text or not answers:
        return None

    seen = set()
    processed_contexts = []
    for raw_context, is_supporting in _iter_contexts(raw_instance):
        title = _context_title(raw_context)
        paragraph_text = _context_text(raw_context)
        if not paragraph_text:
            continue
        key = (title, paragraph_text)
        if key in seen:
            continue
        seen.add(key)
        processed_contexts.append(
            {
                "idx": len(processed_contexts),
                "title": title,
                "paragraph_text": paragraph_text,
                "is_supporting": bool(is_supporting),
            }
        )

    answers_object = {
        "number": "",
        "date": {"day": "", "month": "", "year": ""},
        "spans": answers,
    }

    return {
        "dataset": DATASET_NAME,
        "question_id": str(raw_instance.get("id") or raw_instance.get("_id") or index),
        "question_text": question_text,
        "answers_objects": [answers_object],
        "contexts": processed_contexts,
    }


def main():
    input_directory = os.path.join("raw_data", DATASET_NAME)
    output_directory = os.path.join("processed_data", DATASET_NAME)
    os.makedirs(output_directory, exist_ok=True)

    for set_name in SET_NAMES:
        input_filepath = _existing_input_path(input_directory, set_name)
        if input_filepath is None:
            print(f"Skipping {set_name}: no raw file found in {input_directory}")
            continue

        print(f"Processing {set_name}: {input_filepath}")
        raw_instances = _load_records(input_filepath)

        processed_instances = []
        for index, raw_instance in enumerate(raw_instances):
            processed_instance = _process_instance(raw_instance, index)
            if processed_instance is not None:
                processed_instances.append(processed_instance)

        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")
        write_jsonl(processed_instances, output_filepath)
        print(f"Wrote {len(processed_instances)} instances to {output_filepath}")


if __name__ == "__main__":
    main()
