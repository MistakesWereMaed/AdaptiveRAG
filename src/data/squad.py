import string
from collections import Counter
from typing import Dict, List


# -----------------------------
# Normalization
# -----------------------------
_punct_table = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> List[str]:
    """
    Lowercase, remove punctuation, collapse whitespace, return tokens.
    """
    if text is None:
        return []

    text = text.lower()
    text = text.translate(_punct_table)
    return text.split()


# -----------------------------
# F1 computation (vectorized-style per pair)
# -----------------------------
def f1_score(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    return float(pred_tokens == gold_tokens)


# -----------------------------
# Core vectorized evaluator
# -----------------------------
def evaluate_batch(
    predictions: List[str],
    references: List[str],
) -> Dict[str, List[float]]:
    """
    Fully vectorized batch SQuAD evaluation.

    Returns:
        {
            "f1": [...],
            "em": [...]
        }
    """

    f1s = []
    ems = []

    for pred, gold in zip(predictions, references):
        pred_toks = normalize(pred)
        gold_toks = normalize(gold)

        f1s.append(f1_score(pred_toks, gold_toks))
        ems.append(exact_match(pred_toks, gold_toks))

    return {
        "f1": f1s,
        "em": ems,
    }


# -----------------------------
# Aggregation helper
# -----------------------------
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0