from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List


PUNCT_TABLE = str.maketrans("", "", string.punctuation)
ARTICLES = {"a", "an", "the"}


def normalize_tokens(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()

    return [token for token in text.split() if token and token not in ARTICLES]


def normalize_text(text: str) -> str:
    return " ".join(normalize_tokens(text))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_tokens(prediction) == normalize_tokens(reference))


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_tokens(prediction)
    gold_tokens = normalize_tokens(reference)

    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def loose_accuracy(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction)
    gold = normalize_text(reference)

    if not pred or not gold:
        return 0.0

    return float(pred == gold or gold in pred)


def evaluate_batch(predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
    f1s, ems, accs = [], [], []

    for pred, gold in zip(predictions, references):
        f1s.append(f1_score(pred, gold))
        ems.append(exact_match(pred, gold))
        accs.append(loose_accuracy(pred, gold))

    return {"f1": f1s, "em": ems, "acc": accs}


def is_correct(f1: float, em: float, threshold: float = 0.8) -> bool:
    return em == 1.0 or f1 >= threshold


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
