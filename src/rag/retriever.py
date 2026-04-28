import json
from pathlib import Path
from typing import List, Sequence, Optional, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from src.rag.trace import ExecutionTrace


class FaissIVFRetriever:
    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        nlist: int = 4096,   # number of clusters (tune this)
    ):
        self.encoder = SentenceTransformer(encoder_name)
        self.index = None

        self.nlist = nlist
        self.documents: List[str] = []

        self.dim = None

    # --------------------------------------------------
    # Build index (FAISS IVF training + add)
    # --------------------------------------------------
    def build(self, documents: Sequence[str], batch_size: int = 1024):

        self.documents = [d for d in documents if d]
        if not self.documents:
            raise ValueError("No documents provided")

        print(f"[FAISS] Encoding {len(self.documents)} documents", flush=True)

        embeddings = self.encoder.encode(
            self.documents,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
        ).astype("float32")

        self.dim = embeddings.shape[1]

        # -----------------------------
        # Normalize (important for cosine)
        # -----------------------------
        print("[FAISS] Normalizing embeddings...", flush=True)
        faiss.normalize_L2(embeddings)

        # -----------------------------
        # IVF index
        # -----------------------------
        print("[FAISS] Creating IVF index...", flush=True)
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.dim,
            self.nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        print("[FAISS] Training IVF index...", flush=True)
        self.index.train(embeddings)

        print("[FAISS] Adding vectors...", flush=True)
        self.index.add(embeddings)

        print("[FAISS] Build complete", flush=True)

    # --------------------------------------------------
    # Batch retrieval
    # --------------------------------------------------
    def retrieve(self, queries: List[str], k: int = 5, trace: Optional[Union[ExecutionTrace, List[ExecutionTrace]]] = None) -> List[List[str]]:

        # Record retrieval in trace(s) (before execution)
        if trace is not None:
            if isinstance(trace, list):
                for t in trace:
                    if t is not None:
                        t.record_retrieval(1)
            else:
                trace.record_retrieval(1)

        q_emb = self.encoder.encode(
            queries,
            convert_to_numpy=True,
            batch_size=1024,
            show_progress_bar=False,
        ).astype("float32")

        faiss.normalize_L2(q_emb)

        D, I = self.index.search(q_emb, k)

        results = []
        for row in I:
            docs = [
                self.documents[i]
                for i in row
                if 0 <= i < len(self.documents)
            ]
            results.append(docs)

        return results

    # --------------------------------------------------
    # Save / load
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