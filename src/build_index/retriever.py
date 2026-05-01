from __future__ import annotations

import json
import faiss
import numpy as np

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from sentence_transformers import SentenceTransformer

from src.schemas import RetrievedDocument


class FaissIVFRetriever:
    """FAISS IVF dense retriever for title-aware passage retrieval."""

    def __init__(
        self,
        encoder_name: str = "BAAI/bge-base-en-v1.5",
        nlist: Optional[int] = None,
        nprobe: int = 8,
        use_cache: bool = True,
    ):
        self.encoder_name = encoder_name
        self.encoder = SentenceTransformer(encoder_name)

        self.nlist = nlist
        self.nprobe = int(nprobe)
        self.use_cache = bool(use_cache)

        self.index: Optional[faiss.Index] = None
        self.documents: List[RetrievedDocument] = []
        self.dim: Optional[int] = None
        self._query_cache: Dict[str, np.ndarray] = {}

    # --------------------------------------------------
    # Document handling
    # --------------------------------------------------

    @staticmethod
    def _coerce_document(raw: Any, position: int) -> RetrievedDocument:
        if isinstance(raw, RetrievedDocument):
            if not raw.doc_id:
                raw.doc_id = f"doc_{position}"
            return raw

        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict or RetrievedDocument, got {type(raw)!r}")

        text = str(raw.get("text", "")).strip()
        if not text:
            raise ValueError(f"Document at position {position} is missing text")

        return RetrievedDocument(
            doc_id=str(raw.get("doc_id") or raw.get("id") or f"doc_{position}"),
            title=str(raw.get("title", "")).strip(),
            text=text,
            source=str(raw.get("source") or "corpus"),
            metadata=dict(raw.get("metadata") or {}),
        )

    @staticmethod
    def _dedupe_key(doc: RetrievedDocument) -> Tuple[str, str]:
        return doc.title.strip(), doc.text.strip()
    
    @staticmethod
    def deduplicate_documents(docs: Iterable[RetrievedDocument]) -> List[RetrievedDocument]:
        seen = set()
        unique_docs: List[RetrievedDocument] = []
        for i, raw in enumerate(docs):
            doc = FaissIVFRetriever._coerce_document(raw, i)
            key = FaissIVFRetriever._dedupe_key(doc)
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs


    @staticmethod
    def _index_text(doc: RetrievedDocument) -> str:
        return f"{doc.title}. {doc.text}" if doc.title else doc.text

    # --------------------------------------------------
    # Build / save / load
    # --------------------------------------------------
    def build(self, documents: Sequence[Any], batch_size: int = 1024) -> None:
        self.documents = self.deduplicate_documents(documents)

        texts = [self._index_text(doc) for doc in self.documents]
        print(f"[FAISS] Encoding {len(texts)} documents", flush=True)

        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        self.dim = int(embeddings.shape[1])
        self.nlist = self.nlist or max(1, min(2048, int(np.sqrt(len(self.documents)))))

        print(f"[FAISS] Building IVF index dim={self.dim} nlist={self.nlist}", flush=True)

        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(
            quantizer,
            self.dim,
            self.nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = self.nprobe

        self.index = index
        self._query_cache.clear()

        print(f"[FAISS] Build complete: docs={len(self.documents)} nprobe={self.nprobe}", flush=True)

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save before building or loading an index")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with (path / "documents.json").open("w", encoding="utf-8") as f:
            json.dump(
                [doc.model_dump() for doc in self.documents],
                f,
                ensure_ascii=False,
                indent=2,
            )

        meta = {
            "encoder_name": self.encoder_name,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "dim": self.dim,
            "num_documents": len(self.documents),
        }
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str | Path) -> "FaissIVFRetriever":
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))

        with (path / "documents.json").open("r", encoding="utf-8") as f:
            raw_documents = json.load(f)

        self.documents = self.deduplicate_documents(raw_documents)
        self.dim = int(self.index.d)
        self.index.nprobe = self.nprobe
        self._query_cache.clear()

        return self

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    def _encode(self, queries: List[str]) -> np.ndarray:
        vectors = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            batch_size=512,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(vectors)
        return vectors

    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        if not self.use_cache:
            return self._encode(queries)

        vectors: List[Optional[np.ndarray]] = [None] * len(queries)
        misses: List[str] = []
        miss_indices: List[int] = []

        for i, query in enumerate(queries):
            cached = self._query_cache.get(query)
            if cached is None:
                misses.append(query)
                miss_indices.append(i)
            else:
                vectors[i] = cached

        if misses:
            encoded = self._encode(misses)
            for query, idx, vector in zip(misses, miss_indices, encoded):
                self._query_cache[query] = vector
                vectors[idx] = vector

        return np.stack(vectors).astype("float32")

    def retrieve(
        self,
        queries: List[str],
        k: int = 5,
        return_scores: bool = False,
    ):
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded")

        if not queries:
            return []

        k = min(int(k), len(self.documents))
        query_vectors = self._encode_queries(queries)
        scores, indices = self.index.search(query_vectors, k)

        batch_results = []
        for row_scores, row_indices in zip(scores, indices):
            results = []

            for rank, (score, doc_idx) in enumerate(zip(row_scores, row_indices), start=1):
                if doc_idx < 0:
                    continue

                base = self.documents[int(doc_idx)]
                doc = RetrievedDocument(
                    doc_id=base.doc_id,
                    title=base.title,
                    text=base.text,
                    score=float(score),
                    rank=rank,
                    source=base.source or "dense",
                    metadata=dict(base.metadata),
                )

                results.append((doc, float(score)) if return_scores else doc)

            batch_results.append(results)

        return batch_results
