from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm.auto import tqdm

from src.file_loader import load_yaml_config
from src.prepare_hotpotqa.hotpotqa import (
    dataset_to_records,
    load_hotpotqa_split,
    write_jsonl,
)


def _stable_id(record: Dict[str, Any], fallback: int) -> str | int:
    return record.get("id") or record.get("source_id") or fallback


def _normalize_ids(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []

    for idx, record in enumerate(records):
        item = dict(record)
        raw_id = _stable_id(item, idx)

        try:
            item["id"] = int(raw_id)
        except (TypeError, ValueError):
            item["source_id"] = str(raw_id)
            item["id"] = idx

        output.append(item)

    return output


def _doc_key(title: str, text: str) -> Tuple[str, str]:
    return title.strip(), text.strip()


def build_corpus(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a deduplicated title-aware paragraph corpus from parsed records."""
    corpus: List[Dict[str, Any]] = []
    seen = set()

    for record in tqdm(records, desc="Building corpus", unit="example"):
        source_id = record.get("source_id", record.get("id"))

        for local_idx, doc in enumerate(record.get("context_documents", [])):
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()

            if not text:
                continue

            key = _doc_key(title, text)
            if key in seen:
                continue

            seen.add(key)
            corpus_id = len(corpus)

            corpus.append(
                {
                    "id": corpus_id,
                    "doc_id": f"hotpotqa_{corpus_id}",
                    "title": title,
                    "text": text,
                    "source": "hotpotqa",
                    "metadata": {
                        "original_example_id": source_id,
                        "paragraph_index": doc.get("paragraph_index", local_idx),
                    },
                }
            )

    return corpus


def main() -> None:
    config_path = "config.yaml"
    paths = load_yaml_config(config_path, section="paths")
    cfg = load_yaml_config(config_path, section="hotpotqa")

    out_dir = Path(paths["hotpotqa_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    config_name = str(cfg.get("config_name", "distractor"))
    build_corpus_flag = bool(cfg.get("build_corpus", True))

    train_records: List[Dict[str, Any]] = []

    for split in ("train", "validation"):
        dataset = load_hotpotqa_split(split=split, config_name=config_name)
        records = _normalize_ids(dataset_to_records(dataset))

        write_jsonl(records, out_dir / f"{split}.jsonl")

        if split == "train":
            train_records = records

    if not build_corpus_flag:
        return

    corpus = build_corpus(train_records)
    if not corpus:
        raise RuntimeError("No corpus passages were extracted from HotpotQA.")

    non_empty_titles = sum(bool(doc["title"]) for doc in corpus)
    if non_empty_titles == 0:
        raise RuntimeError("Corpus was built, but every title is empty.")

    write_jsonl(corpus, Path(paths["corpus"]))


if __name__ == "__main__":
    main()
