import re
import string
import unicodedata
from collections import Counter
from typing import Dict, List

# -----------------------------
# Normalization
# -----------------------------
_punct_table = str.maketrans("", "", string.punctuation)
_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> List[str]:
    if not text:
        return []

    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = text.translate(_punct_table)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]

    return tokens


# -----------------------------
# Metrics
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


def accuracy(pred: str, gold: str) -> float:
    """
    Loose containment metric (for reporting only).
    """
    if not pred or not gold:
        return 0.0
    return float(normalize(gold) == normalize(pred) or " ".join(normalize(gold)) in " ".join(normalize(pred)))


# -----------------------------
# Batch evaluation
# -----------------------------
def evaluate_batch(predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
    f1s, ems, accs = [], [], []

    for pred, gold in zip(predictions, references):
        pred_toks = normalize(pred)
        gold_toks = normalize(gold)

        f1 = f1_score(pred_toks, gold_toks)
        em = exact_match(pred_toks, gold_toks)
        acc = accuracy(pred, gold)

        f1s.append(f1)
        ems.append(em)
        accs.append(acc)

    return {"f1": f1s, "em": ems, "acc": accs}


# -----------------------------
# Correctness
# -----------------------------
def is_correct(pred: str, gold: str, f1: float, em: float) -> bool:
    """
    Binary correctness used for labeling.
    """
    return em == 1.0 or f1 >= 0.8


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0