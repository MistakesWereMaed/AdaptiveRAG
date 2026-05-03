from __future__ import annotations

from typing import Any, Dict, List


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


def document(title: str = "", text: str = "", paragraph_index: int = 0, **metadata: Any) -> Dict[str, Any]:
    return {
        "title": str(title or "").strip(),
        "text": str(text or "").strip(),
        "paragraph_index": paragraph_index,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }