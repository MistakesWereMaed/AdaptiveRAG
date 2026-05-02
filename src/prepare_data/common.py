from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def first_answer(value: Any) -> str:
    """Return a single answer string from common QA answer schemas."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        # SQuAD / NQ-style: {"text": [...]} or {"answer": ...}
        for key in ("text", "answer", "answers", "aliases"):
            if key in value:
                out = first_answer(value[key])
                if out:
                    return out
        return ""

    if isinstance(value, (list, tuple)):
        for item in value:
            out = first_answer(item)
            if out:
                return out
        return ""

    return str(value).strip()


def answer_list(value: Any) -> List[str]:
    """Return all usable answer strings from common answer schemas."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if isinstance(value, dict):
        for key in ("text", "answer", "answers", "aliases"):
            if key in value:
                return answer_list(value[key])
        return []

    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(answer_list(item))
        seen = set()
        deduped = []
        for ans in out:
            if ans and ans not in seen:
                seen.add(ans)
                deduped.append(ans)
        return deduped

    text = str(value).strip()
    return [text] if text else []


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_sample(
    records: Sequence[Dict[str, Any]],
    n: int,
    seed: int,
    dataset_name: str,
    split_name: str,
) -> List[Dict[str, Any]]:
    if len(records) < n:
        raise ValueError(
            f"{dataset_name}/{split_name} has only {len(records)} records; "
            f"cannot sample {n}."
        )

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])

    sampled: List[Dict[str, Any]] = []
    for sample_idx, source_idx in enumerate(chosen):
        item = dict(records[source_idx])
        item["dataset"] = dataset_name
        item["source_split"] = split_name
        item["sample_index"] = sample_idx
        item["source_index"] = source_idx
        sampled.append(item)

    return sampled


def normalize_numeric_ids(records: Iterable[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        item = dict(record)
        raw_id = item.get("id") or item.get("_id") or item.get("qid") or item.get("question_id") or idx
        item["source_id"] = str(raw_id)
        item["id"] = idx
        item["dataset"] = dataset_name
        out.append(item)

    return out


def document(title: str = "", text: str = "", paragraph_index: int = 0, **metadata: Any) -> Dict[str, Any]:
    return {
        "title": str(title or "").strip(),
        "text": str(text or "").strip(),
        "paragraph_index": paragraph_index,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }
