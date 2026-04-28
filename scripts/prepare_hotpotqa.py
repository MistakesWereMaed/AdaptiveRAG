import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from src.data.hotpotqa import hotpotqa_context_to_documents, hotpotqa_dataset_to_records, load_hotpotqa_split
from src.data.file_loader import load_yaml_config


def _ensure_ids(records):
    normalized = []
    for index, record in enumerate(records):
        current = dict(record)
        raw_id = current.get("id")
        if raw_id is None:
            current["id"] = index
        else:
            try:
                current["id"] = int(raw_id)
            except (TypeError, ValueError):
                current["source_id"] = raw_id
                current["id"] = index
        normalized.append(current)
    return normalized


def _write_jsonl(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(_ensure_ids(records), desc=f"Writing {output_path.name}", unit="record"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_corpus(records):
    corpus = []
    seen_texts = set()
    for record in records:
        context_documents = record.get("context_documents")
        if not isinstance(context_documents, list):
            context_documents = hotpotqa_context_to_documents(record.get("context"))

        for passage in context_documents:
            passage = passage.strip()
            if passage and passage not in seen_texts:
                seen_texts.add(passage)
                corpus.append({"text": passage})
    return corpus


def main():
    print("[prepare_hotpotqa] Starting HotpotQA preprocessing", flush=True)
    parser = argparse.ArgumentParser(description="Download and preprocess HotpotQA")
    parser.add_argument("--config", default="config.yaml", help="Path to HotpotQA config")
    args = parser.parse_args()

    config = load_yaml_config(args.config, section="hotpotqa")
    output_dir = Path(config["output_dir"])
    config_name = str(config["config_name"])
    build_corpus = bool(config["build_corpus"])

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in tqdm(("train", "validation"), desc="HotpotQA splits", unit="split"):
        dataset = load_hotpotqa_split(split=split, config_name=config_name)
        records = hotpotqa_dataset_to_records(dataset)
        _write_jsonl(records, output_dir / f"{split}.jsonl")

        if build_corpus and split == "train":
            corpus = _build_corpus(records)
            if not corpus:
                raise ValueError("HotpotQA corpus extraction produced no passages; check the dataset schema or config name")
            _write_jsonl(corpus, output_dir / "corpus.jsonl")


if __name__ == "__main__":
    main()
