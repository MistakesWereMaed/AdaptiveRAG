import json
from pathlib import Path
from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from cs6263_template.src.myproject.src.rag.trace import ExecutionTrace


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
        self.documents: List[str] = []
        self.dim = None

        self.nprobe = nprobe
        self.use_cache = use_cache
        self._query_cache: Dict[str, Tuple[List[str], np.ndarray]] = {}

        # will be inferred if not provided
        self.nlist = nlist

    # --------------------------------------------------
    # Document preprocessing
    # --------------------------------------------------
    def _deduplicate(self, docs: Sequence[str]) -> List[str]:
        seen = set()
        out = []
        for d in docs:
            if d and d not in seen:
                out.append(d)
                seen.add(d)
        return out

    # --------------------------------------------------
    # Build index
    # --------------------------------------------------
    def build(self, documents: Sequence[str], batch_size: int = 1024):

        self.documents = self._deduplicate(documents)

        if not self.documents:
            raise ValueError("Empty document set")

        print(f"[FAISS] Encoding {len(self.documents)} documents")

        embeddings = self.encoder.encode(
            self.documents,
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

        # -------------------------
        # vectorized extraction (FASTER FIX)
        # -------------------------
        docs_arr = np.array(self.documents, dtype=object)
        retrieved_docs = docs_arr[I]

        results = []

        for i in range(len(queries)):

            docs = retrieved_docs[i].tolist()
            scores = D[i]

            if return_scores:
                results.append(list(zip(docs, scores)))
            else:
                results.append(docs)

        return results

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f)

    def load(self, path: str):
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "documents.json", "r") as f:
            self.documents = json.load(f)

        self.dim = self.index.d

        self.index.nprobe = self.nprobe