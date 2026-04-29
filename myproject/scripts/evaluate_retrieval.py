import argparse
import json
import sys
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.file_loader import load_records, load_yaml_config, normalize_text
from src.rag.context import deduplicate_documents
from src.rag.retriever import FaissIVFRetriever


def _load_retriever(config_path: str, retriever_name: str):
    if retriever_name != "dense":
        raise ValueError(f"Only 'dense' retriever is supported, got: {retriever_name}")
    
    paths = load_yaml_config(config_path, section="paths")
    retr_cfg = load_yaml_config(config_path, section="retriever")
    index_dir = Path(paths["index_dir"])

    dense = FaissIVFRetriever(encoder_name=retr_cfg["encoder_name"])
    dense.load(index_dir)
    return dense


def _normalize(text: str) -> str:
    return normalize_text(text)


def _metrics_for_example(question: str, gold: str, support_titles: List[str], retrieved_docs, ks=(1, 3, 5, 10, 20)):
    metrics = {}
    retrieved_titles = [doc.title for doc in retrieved_docs]
    retrieved_text = " ".join(doc.text for doc in retrieved_docs)
    norm_gold = _normalize(gold)
    norm_text = _normalize(retrieved_text)

    for k in ks:
        top_docs = retrieved_docs[:k]
        top_titles = [doc.title for doc in top_docs]
        top_text = " ".join(doc.text for doc in top_docs)
        top_gold_hits = sum(1 for title in support_titles if title in top_titles)
        total_support = max(1, len(set(support_titles)))
        metrics[f"answer_hit@{k}"] = 1.0 if norm_gold and norm_gold in _normalize(top_text) else 0.0
        metrics[f"support_title_hit@{k}"] = 1.0 if top_gold_hits > 0 else 0.0
        metrics[f"support_recall@{k}"] = top_gold_hits / total_support
        metrics[f"support_precision@{k}"] = top_gold_hits / max(1, len(top_titles))
        metrics[f"oracle_context_answerable@{k}"] = 1.0 if (norm_gold and norm_gold in norm_text) or top_gold_hits > 0 else 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality without generation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--retriever", default="dense", choices=["dense"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", default=None)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    paths = load_yaml_config(args.config, section="paths")
    split_path = Path(paths[f"{args.split}_data"])
    records = load_records(split_path)
    retriever = _load_retriever(args.config, args.retriever)

    ks = (1, 3, 5, 10, 20)
    totals = {f"answer_hit@{k}": 0.0 for k in ks}
    totals.update({f"support_title_hit@{k}": 0.0 for k in ks})
    totals.update({f"support_recall@{k}": 0.0 for k in ks})
    totals.update({f"support_precision@{k}": 0.0 for k in ks})
    totals.update({f"oracle_context_answerable@{k}": 0.0 for k in ks})

    diagnostics_path = Path(args.diagnostics) if args.diagnostics else Path(args.output).with_suffix(".jsonl")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    with diagnostics_path.open("w", encoding="utf-8") as diagnostics_file:
        for record in tqdm(records, desc=f"Evaluating {args.retriever}", unit="example"):
            support_titles = []
            metadata = getattr(record, "metadata", None) or {}
            if isinstance(metadata, dict):
                maybe_titles = metadata.get("supporting_titles") or metadata.get("support_titles") or []
                if isinstance(maybe_titles, list):
                    support_titles = [str(title) for title in maybe_titles if str(title).strip()]

            retrieved_docs = retriever.retrieve([record.question], k=args.k)[0]
            retrieved_docs = deduplicate_documents(retrieved_docs)
            per_example = _metrics_for_example(record.question, record.gold, support_titles, retrieved_docs, ks=ks)

            for key, value in per_example.items():
                totals[key] += value

            diagnostics_file.write(
                json.dumps(
                    {
                        "id": record.id,
                        "question": record.question,
                        "gold": record.gold,
                        "retrieved": [doc.model_dump() for doc in retrieved_docs],
                        "metrics": per_example,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    num_examples = len(records)
    output = {
        "retriever_name": f"{args.retriever}",
        "split": args.split,
        "num_examples": num_examples,
        "metrics": {key: (value / num_examples if num_examples else 0.0) for key, value in totals.items()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()