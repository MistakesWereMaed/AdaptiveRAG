from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple
from tqdm.auto import tqdm

from src.prepare_data.common import (
    normalize_numeric_ids,
    read_jsonl,
    stable_sample,
    write_jsonl,
)

from src.prepare_data.datasets.hotpotqa import dataset_to_records as hotpot_to_records
from src.prepare_data.datasets.hotpotqa import load_hotpotqa_split

from src.prepare_data.datasets.squad import dataset_to_records as squad_to_records
from src.prepare_data.datasets.squad import load_squad_split

from src.prepare_data.datasets.natural_questions import dataset_to_records as nq_to_records
from src.prepare_data.datasets.natural_questions import load_natural_questions_split

from src.prepare_data.datasets.triviaqa import dataset_to_records as trivia_to_records
from src.prepare_data.datasets.triviaqa import load_triviaqa_split

from src.prepare_data.datasets.musique import dataset_to_records as musique_to_records
from src.prepare_data.datasets.musique import load_musique_split

from src.prepare_data.datasets.twowiki import dataset_to_records as twowiki_to_records
from src.prepare_data.datasets.twowiki import load_twowiki_split


DatasetSpec = Dict[str, Any]


DATASETS: Dict[str, DatasetSpec] = {
    "squad": {
        "type": "single-hop",
        "train_split": "train",
        "eval_split": "validation",
        "loader": load_squad_split,
        "converter": squad_to_records,
        "loader_kwargs": {},
    },
    "natural_questions": {
        "type": "single-hop",
        "train_split": "train",
        "eval_split": "validation",
        "loader": load_natural_questions_split,
        "converter": nq_to_records,
        "loader_kwargs": {},
    },
    "triviaqa": {
        "type": "single-hop",
        "train_split": "train",
        "eval_split": "validation",
        "loader": load_triviaqa_split,
        "converter": trivia_to_records,
        "loader_kwargs": {"config_name": "rc.nocontext"},
    },
    "musique": {
        "type": "multi-hop",
        "train_split": "train",
        "eval_split": "validation",
        "loader": load_musique_split,
        "converter": musique_to_records,
        "loader_kwargs": {},
    },
    "hotpotqa": {
        "type": "multi-hop",
        "train_split": "train",
        "eval_split": "validation",
        "loader": load_hotpotqa_split,
        "converter": hotpot_to_records,
        "loader_kwargs": {"config_name": "distractor"},
    },
    "twowikimultihopqa": {
        "type": "multi-hop",
        "train_split": "train",
        "eval_split": "validation",
        "loader": load_twowiki_split,
        "converter": twowiki_to_records,
        "loader_kwargs": {},
    },
}


def _convert_dataset(name: str, spec: DatasetSpec, split: str) -> List[Dict[str, Any]]:
    loader: Callable[..., Any] = spec["loader"]
    converter: Callable[[Iterable[Dict[str, Any]]], List[Dict[str, Any]]] = spec["converter"]
    loader_kwargs = dict(spec.get("loader_kwargs", {}))

    dataset = loader(split=split, **loader_kwargs)
    records = converter(dataset)
    records = normalize_numeric_ids(records, dataset_name=name)

    for record in records:
        record["question_type"] = spec["type"]
        record["dataset"] = name

    return records


def _write_split(
    name: str,
    spec: DatasetSpec,
    split_kind: str,
    out_dir: Path,
    force: bool,
) -> Path:
    split = spec[f"{split_kind}_split"]
    output_path = out_dir / name / f"{split_kind}.jsonl"

    if output_path.exists() and not force:
        print(f"[skip] {name}/{split_kind}: {output_path}")
        return output_path

    print(f"[load] {name}/{split_kind} <- HF split={split}")
    records = _convert_dataset(name, spec, split)
    write_jsonl(records, output_path)
    print(f"[write] {len(records)} records -> {output_path}")
    return output_path


def build_deduped_corpus(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    seen = set()

    for record in tqdm(records, desc="Building full corpus", unit="example"):
        dataset = str(record.get("dataset", "unknown"))
        source_id = record.get("source_id", record.get("id"))

        for local_idx, doc in enumerate(record.get("context_documents", [])):
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            if not text:
                continue

            key = (title, text)
            if key in seen:
                continue
            seen.add(key)

            corpus_id = len(corpus)
            corpus.append(
                {
                    "id": corpus_id,
                    "doc_id": f"{dataset}_{corpus_id}",
                    "title": title,
                    "text": text,
                    "source": dataset,
                    "metadata": {
                        "original_example_id": source_id,
                        "paragraph_index": doc.get("paragraph_index", local_idx),
                    },
                }
            )

    return corpus


def prepare_all(
    out_dir: Path,
    sample_size: int,
    seed: int,
    force: bool = False,
    build_corpus: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths: Dict[str, Path] = {}
    eval_paths: Dict[str, Path] = {}

    for name, spec in DATASETS.items():
        train_paths[name] = _write_split(name, spec, "train", out_dir, force)
        eval_paths[name] = _write_split(name, spec, "eval", out_dir, force)

    full_eval: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "sample_size_per_dataset": sample_size,
        "seed": seed,
        "datasets": {},
    }

    for name, path in eval_paths.items():
        records = read_jsonl(path)
        sampled = stable_sample(records, n=sample_size, seed=seed, dataset_name=name, split_name="eval")
        sample_path = out_dir / name / f"eval_{sample_size}.jsonl"
        write_jsonl(sampled, sample_path)

        full_eval.extend(sampled)
        manifest["datasets"][name] = {
            "type": DATASETS[name]["type"],
            "eval_records": len(records),
            "sampled_records": len(sampled),
            "sample_path": str(sample_path),
        }

    full_eval_path = out_dir / f"full_eval_{sample_size}_each.jsonl"
    write_jsonl(full_eval, full_eval_path)
    manifest["full_eval_path"] = str(full_eval_path)
    manifest["full_eval_records"] = len(full_eval)

    if build_corpus:
        # Build a compact corpus from the sampled evaluation contexts.
        # This is useful for debugging but is not a substitute for the paper's
        # official BM25 Wikipedia/IRCoT corpora.
        corpus = build_deduped_corpus(full_eval)
        corpus_path = out_dir / f"full_eval_{sample_size}_each_context_corpus.jsonl"
        write_jsonl(corpus, corpus_path)
        manifest["sample_context_corpus_path"] = str(corpus_path)
        manifest["sample_context_corpus_records"] = len(corpus)

    import json
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest -> {manifest_path}")
    print(f"[done] full eval -> {full_eval_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the six-dataset Adaptive-RAG evaluation mixture."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/full_qa"))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-corpus", action="store_true")
    args = parser.parse_args()

    prepare_all(
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        force=args.force,
        build_corpus=not args.no_corpus,
    )


if __name__ == "__main__":
    main()
