from __future__ import annotations

from pathlib import Path
from typing import List

from src.build_index.retriever import FaissIVFRetriever
from src.file_loader import extract_structured_documents, load_raw_records, load_yaml_config
from src.schemas import RetrievedDocument


def _pick_test_query(documents: List[RetrievedDocument]) -> str:
    """Use a real title/text fragment from the corpus as a reliable smoke-test query."""
    for doc in documents:
        if doc.title:
            return doc.title

    first = documents[0].text.split()
    return " ".join(first[:8]) if first else "test query"


def _smoke_test(index_dir: str | Path, encoder_name: str, nprobe: int, query: str, k: int = 3) -> None:
    print(f"[build_index] Smoke test query: {query!r}", flush=True)

    retriever = FaissIVFRetriever(
        encoder_name=encoder_name,
        nprobe=nprobe,
        use_cache=False,
    ).load(index_dir)

    results = retriever.retrieve([query], k=k)[0]

    if not results:
        raise RuntimeError("Smoke test failed: retrieval returned no documents")

    print("[build_index] Smoke test results:", flush=True)
    for doc in results:
        title = doc.title or "<no title>"
        snippet = doc.text[:160].replace("\n", " ")
        print(f"  rank={doc.rank} score={doc.score:.4f} title={title} text={snippet}", flush=True)


def run_build_index(config_path: str = "config.yaml") -> None:
    print("[build_index] Starting index build", flush=True)

    paths = load_yaml_config(config_path, section="paths")
    cfg = load_yaml_config(config_path, section="retriever")

    corpus_path = Path(paths["corpus"])
    index_dir = Path(paths["index_dir"])

    encoder_name = str(cfg.get("encoder_name", "BAAI/bge-base-en-v1.5"))
    nlist = cfg.get("nlist")
    nprobe = int(cfg.get("nprobe", 8))
    batch_size = int(cfg.get("batch_size", 1024))
    smoke_k = int(cfg.get("smoke_test_k", 3))

    print(f"[build_index] Loading corpus: {corpus_path}", flush=True)
    records = load_raw_records(corpus_path)
    documents = extract_structured_documents(records)

    if not documents:
        raise RuntimeError(f"No documents loaded from corpus: {corpus_path}")

    print(f"[build_index] Loaded {len(documents)} documents", flush=True)

    retriever = FaissIVFRetriever(
        encoder_name=encoder_name,
        nlist=int(nlist) if nlist is not None else None,
        nprobe=nprobe,
    )

    retriever.build(documents, batch_size=batch_size)
    retriever.save(index_dir)

    print(f"[build_index] Wrote index to: {index_dir}", flush=True)

    test_query = _pick_test_query(retriever.documents)
    _smoke_test(index_dir, encoder_name=encoder_name, nprobe=nprobe, query=test_query, k=smoke_k)

    print("[build_index] Done", flush=True)


def main() -> None:
    run_build_index("config.yaml")


if __name__ == "__main__":
    main()
