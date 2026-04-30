import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm.auto import tqdm

from myproject.src.file_loader import load_yaml_config
from myproject.src.prepare_hotpotqa.hotpotqa import (
    hotpotqa_context_to_structured_documents,
    hotpotqa_dataset_to_records,
    load_hotpotqa_split,
)


def _normalize_id(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    current = dict(record)
    raw_id = current.get("id")

    if raw_id is None:
        current["id"] = index
        return current

    try:
        current["id"] = int(raw_id)
    except (TypeError, ValueError):
        current["source_id"] = str(raw_id)
        current["id"] = index

    return current


def _write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(
            tqdm(records, desc=f"Writing {output_path.name}", unit="record")
        ):
            normalized = _normalize_id(record, index)
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")


def _dedupe_key(doc: Dict[str, Any]) -> Tuple[str, str]:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    return title, text


def _build_corpus(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    seen_keys = set()

    for record in tqdm(records, desc="Building HotpotQA corpus", unit="example"):
        context_documents = record.get("context_documents")

        if not isinstance(context_documents, list):
            context_documents = hotpotqa_context_to_structured_documents(
                record.get("raw_context") or record.get("context")
            )

        for paragraph_index, doc in enumerate(context_documents):
            if not isinstance(doc, dict):
                continue

            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()

            if not text:
                continue

            key = _dedupe_key({"title": title, "text": text})
            if key in seen_keys:
                continue

            seen_keys.add(key)

            doc_id = f"hotpotqa_{len(corpus)}"

            corpus.append(
                {
                    "id": len(corpus),
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "source": "hotpotqa",
                    "metadata": {
                        "original_example_id": record.get("source_id", record.get("id")),
                        "paragraph_index": paragraph_index,
                    },
                }
            )

    return corpus


def run_prepare_hotpotqa(config_path: str = "config.yaml") -> None:
    print("[prepare_hotpotqa] Starting HotpotQA preprocessing", flush=True)

    paths = load_yaml_config(config_path, section="paths")
    config = load_yaml_config(config_path, section="hotpotqa")

    output_dir = Path(paths["hotpotqa_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config_name = str(config.get("config_name", "distractor"))
    build_corpus = bool(config.get("build_corpus", True))

    train_records: List[Dict[str, Any]] = []

    for split in tqdm(("train", "validation"), desc="HotpotQA splits", unit="split"):
        dataset = load_hotpotqa_split(split=split, config_name=config_name)
        records = hotpotqa_dataset_to_records(dataset)

        if split == "train":
            train_records = records

        _write_jsonl(records, output_dir / f"{split}.jsonl")

    if build_corpus:
        corpus = _build_corpus(train_records)

        if not corpus:
            raise ValueError(
                "HotpotQA corpus extraction produced no passages. "
                "Check dataset schema, config name, or context parser."
            )

        _write_jsonl(corpus, Path(paths["corpus"]))

        non_empty_titles = sum(1 for doc in corpus if doc.get("title"))
        print(
            f"[prepare_hotpotqa] Corpus size: {len(corpus)} passages; "
            f"non-empty titles: {non_empty_titles}",
            flush=True,
        )

        if non_empty_titles == 0:
            raise ValueError(
                "Corpus was built, but all titles are empty. "
                "This indicates title extraction is still broken."
            )
