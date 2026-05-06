#!/usr/bin/env python3
"""
Export official-style classifier JSON to simple T5 TSV.

Input records use:
  question
  answer   # A/B/C

Output:
  prompt<TAB>answer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean(text: str) -> str:
    return " ".join(str(text).replace("\t", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-template", default="Question: {question} Complexity:")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in records:
            prompt = clean(args.prompt_template.format(question=row["question"]))
            answer = clean(row["answer"])
            f.write(f"{prompt}\t{answer}\n")

    print(out)
    print(len(records))


if __name__ == "__main__":
    main()
