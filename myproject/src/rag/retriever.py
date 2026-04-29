import json
from pathlib import Path
from typing import Any, List, Sequence, Optional, Union, Dict, Tuple, Iterable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.data.schemas import RetrievedDocument
from src.rag.trace import ExecutionTrace


class FaissIVFRetriever:
    def __init__(
        self,
        encoder_name: str = "BAAI/bge-base-en-v1.5",
        nlist: Optional[int] = None,
        nprobe: int = 8,
        use_cache: bool = True,
    ):
        self.encoder = SentenceTransformer(encoder_name)

        self.index = None
        self.documents: List[RetrievedDocument] = []
        self.dim = None

        self.nprobe = nprobe
        self.use_cache = use_cache
        self._query_cache: Dict[str, Tuple[List[str], np.ndarray]] = {}

        # will be inferred if not provided
        self.nlist = nlist

    # --------------------------------------------------
    # Document preprocessing
    # --------------------------------------------------
    def _doc_key(self, doc: RetrievedDocument) -> str:
        if doc.doc_id:
            return doc.doc_id
        return f"{doc.title.strip()}::{doc.text.strip()[:100]}"

    def _coerce_document(self, doc: Any, position: int) -> RetrievedDocument:
        if isinstance(doc, RetrievedDocument):
            if not doc.doc_id:
                doc.doc_id = f"doc_{position}"
            return doc

        if isinstance(doc, str):
            return RetrievedDocument(doc_id=f"doc_{position}", title="", text=doc.strip(), source="corpus")

        if isinstance(doc, dict):
            raw_text = None
            for key in ("text", "content", "passage", "document", "context"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    raw_text = value.strip()
                    break

            if raw_text is None:
                raise ValueError("Document is missing text content")

            raw_doc_id = doc.get("doc_id") or doc.get("id") or doc.get("_id") or f"doc_{position}"
            raw_title = doc.get("title") or doc.get("name") or doc.get("doc_title") or ""
            metadata = dict(doc.get("metadata") or {})
            for key, value in doc.items():
                if key not in {"doc_id", "id", "_id", "title", "name", "doc_title", "text", "content", "passage", "document", "context", "metadata", "score", "rank", "source"}:
                    metadata.setdefault(key, value)

            return RetrievedDocument(
                doc_id=str(raw_doc_id).strip(),
                title=str(raw_title).strip(),
                text=raw_text,
                score=doc.get("score"),
                rank=doc.get("rank"),
                source=str(doc.get("source") or "corpus"),
                metadata=metadata,
            )

        raise TypeError(f"Unsupported document type: {type(doc)!r}")

    def _deduplicate(self, docs: Sequence[Any]) -> List[RetrievedDocument]:
        seen = set()
        out = []
        for position, raw_doc in enumerate(docs):
            doc = self._coerce_document(raw_doc, position)
            key = self._doc_key(doc)
            if key and key not in seen:
                out.append(doc)
                seen.add(key)
        return out

    # --------------------------------------------------
    # Build index
    # --------------------------------------------------
    def _index_text(self, doc: RetrievedDocument) -> str:
        title = doc.title.strip()
        if title:
            return f"{title}. {doc.text.strip()}"
        return doc.text.strip()

    def build(self, documents: Sequence[Any], batch_size: int = 1024):

        self.documents = self._deduplicate(documents)

        if not self.documents:
            raise ValueError("Empty document set")

        print(f"[FAISS] Encoding {len(self.documents)} documents")

        embeddings = self.encoder.encode(
            [self._index_text(doc) for doc in self.documents],
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        ).astype("float32")

        self.dim = embeddings.shape[1]

        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # --------------------------------------------------
        # adaptive nlist
        # --------------------------------------------------
        if self.nlist is None:
            self.nlist = max(1, min(2048, int(np.sqrt(len(self.documents)))))

        print(f"[FAISS] Using nlist={self.nlist}")

        # --------------------------------------------------
        # IVF index
        # --------------------------------------------------
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.dim,
            self.nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        assert not self.index.is_trained

        print("[FAISS] Training index...")
        self.index.train(embeddings)

        assert self.index.is_trained

        print("[FAISS] Adding vectors...")
        self.index.add(embeddings)

        # --------------------------------------------------
        # CRITICAL FIX: retrieval sensitivity
        # --------------------------------------------------
        self.index.nprobe = self.nprobe

        print(f"[FAISS] Build complete (nprobe={self.nprobe})")

    # --------------------------------------------------
    # Cache-aware embedding
    # --------------------------------------------------
    def _encode_queries(self, queries: List[str]) -> np.ndarray:

        if not self.use_cache:
            return self._encode(queries)

        cached_vecs = []
        to_compute = []
        idx_map = []

        for i, q in enumerate(queries):
            if q in self._query_cache:
                cached_vecs.append((i, self._query_cache[q]))
            else:
                to_compute.append(q)
                idx_map.append(i)

        if to_compute:
            vecs = self._encode(to_compute)

            for q, v in zip(to_compute, vecs):
                self._query_cache[q] = v

            cached_vecs.extend(zip(idx_map, vecs))

        # restore order
        cached_vecs.sort(key=lambda x: x[0])
        return np.stack([v for _, v in cached_vecs])

    def _encode(self, queries: List[str]) -> np.ndarray:
        q_emb = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            batch_size=512,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(q_emb)
        return q_emb

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------
    def retrieve(
        self,
        queries: List[str],
        k: int = 5,
        trace: Optional[Union[ExecutionTrace, List[ExecutionTrace]]] = None,
        return_scores: bool = False,
    ) -> Union[List[List[str]], List[List[Tuple[str, float]]]]:

        # -------------------------
        # trace logging (FIXED)
        # -------------------------
        if trace is not None:
            if isinstance(trace, list):
                for t in trace:
                    if t is not None:
                        t.record_retrieval(k)
            else:
                trace.record_retrieval(k)

        # -------------------------
        # embedding
        # -------------------------
        q_emb = self._encode_queries(queries)

        # -------------------------
        # search
        # -------------------------
        D, I = self.index.search(q_emb, k)

        results = []

        for i in range(len(queries)):

            docs = []
            scores = []
            for doc_index, score in zip(I[i].tolist(), D[i].tolist()):
                if doc_index < 0 or doc_index >= len(self.documents):
                    continue
                docs.append(self.documents[doc_index])
                scores.append(score)

            if return_scores:
                result_docs = []
                for rank, (doc, score) in enumerate(zip(docs, scores), start=1):
                    result_docs.append(
                        (
                            RetrievedDocument(
                                doc_id=doc.doc_id,
                                title=doc.title,
                                text=doc.text,
                                score=float(score),
                                rank=rank,
                                source=doc.source or "dense",
                                metadata=dict(doc.metadata),
                            ),
                            float(score),
                        )
                    )
                results.append(result_docs)
            else:
                result_docs = []
                for rank, (doc, score) in enumerate(zip(docs, scores), start=1):
                    result_docs.append(
                        RetrievedDocument(
                            doc_id=doc.doc_id,
                            title=doc.title,
                            text=doc.text,
                            score=float(score),
                            rank=rank,
                            source=doc.source or "dense",
                            metadata=dict(doc.metadata),
                        )
                    )
                results.append(result_docs)

        return results

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "documents.json", "w", encoding="utf-8") as f:
            json.dump([doc.model_dump() for doc in self.documents], f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "documents.json", "r", encoding="utf-8") as f:
            loaded_documents = json.load(f)

        self.documents = self._deduplicate(loaded_documents)

        self.dim = self.index.d

        self.index.nprobe = self.nprobe