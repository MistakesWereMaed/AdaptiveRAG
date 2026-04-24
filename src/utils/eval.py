import re
import string
from collections import Counter


def normalize(text):
    print("[eval] Normalizing text", flush=True)
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join(text.split())
    return text


def compute_f1(pred, gold):
    print("[eval] Computing F1", flush=True)
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)