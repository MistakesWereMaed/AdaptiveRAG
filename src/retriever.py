import json
from pathlib import Path
from typing import List, Sequence

from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class Retriever:
    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "retrieval_index"
    ):
        print(f"[Retriever] Initializing encoder={encoder_name} collection={collection_name}", flush=True)
        self.encoder = SentenceTransformer(encoder_name)
        self.collection_name = collection_name

        # local in-memory / on-disk Qdrant
        self.client = QdrantClient(path=".qdrant_db")

        self.texts: List[str] = []

    # --------------------------------------------------
    # Build index
    # --------------------------------------------------
    def build_index(self, documents: List[str]):
        print(f"[Retriever] Building index for {len(documents)} documents", flush=True)
        if not documents:
            raise ValueError("No documents provided")

        self.texts = documents

        embeddings = self.encoder.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=256
        ).astype("float32")

        dim = embeddings.shape[1]

        # create collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

        # insert points
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={"text": documents[i]}
            )
            for i in tqdm(range(len(documents)), desc="Indexing", unit="doc")
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # --------------------------------------------------
    # Retrieve
    # --------------------------------------------------
    def retrieve(self, query: str, k: int = 5):
        print(f"[Retriever] Retrieving top-{k} documents", flush=True)
        if not self.texts:
            raise ValueError("Index not built yet")

        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")[0]

        if hasattr(self.client, "query_points"):
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=q_emb,
                limit=k,
                with_payload=True,
            )
            points = results.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=q_emb,
                limit=k,
                with_payload=True,
            )

        return [hit.payload["text"] for hit in points if hit.payload and "text" in hit.payload]

    # --------------------------------------------------
    # Save / Load (Qdrant handles persistence automatically)
    # --------------------------------------------------
    def save_index(self, path: str):
        print(f"[Retriever] Saving index metadata to {path}", flush=True)
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "documents.json", "w") as f:
            json.dump(self.texts, f)

    def load_index(self, path: str):
        print(f"[Retriever] Loading index metadata from {path}", flush=True)
        with open(Path(path) / "documents.json", "r", encoding="utf-8") as f:
            self.texts = json.load(f)

        # Recreate the collection from saved docs so loading works even if
        # .qdrant_db is empty or was removed.
        self.build_index(self.texts)


class QdrantRetrieverIndex:
    def __init__(
        self,
        encoder_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "qdrant",
    ):
        print(f"[QdrantRetrieverIndex] Initializing encoder={encoder_name} collection={collection_name}", flush=True)
        self.encoder = SentenceTransformer(encoder_name)
        self.collection_name = collection_name
        self.client = QdrantClient(path=".qdrant_db")
        self.documents: List[str] = []

    def build(self, documents: Sequence[str]) -> None:
        print(f"[QdrantRetrieverIndex] Building index for {len(documents)} documents", flush=True)
        self.documents = [document for document in documents if document]
        if not self.documents:
            raise ValueError("No documents provided")

        embeddings = self.encoder.encode(
            self.documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=256,
        ).astype("float32")

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
        )

        points = [
            PointStruct(id=i, vector=embeddings[i], payload={"text": self.documents[i]})
            for i in tqdm(range(len(self.documents)), desc="Indexing", unit="doc")
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, k: int = 5) -> List[str]:
        print(f"[QdrantRetrieverIndex] Searching top-{k} documents", flush=True)
        if not self.documents:
            raise ValueError("Index not built yet")

        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")[0]

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=q_emb,
                limit=k,
                with_payload=True,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=q_emb,
                limit=k,
                with_payload=True,
            )

        return [hit.payload["text"] for hit in points if hit.payload and "text" in hit.payload]

    def save(self, output_dir: str | Path) -> None:
        print(f"[QdrantRetrieverIndex] Saving metadata to {output_dir}", flush=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with (output_path / "documents.json").open("w", encoding="utf-8") as f:
            json.dump(self.documents, f)

    @classmethod
    def load(
        cls,
        input_dir: str | Path,
        encoder_name: str = "all-MiniLM-L6-v2",
    ) -> "QdrantRetrieverIndex":
        print(f"[QdrantRetrieverIndex] Loading metadata from {input_dir}", flush=True)
        instance = cls(encoder_name=encoder_name)

        with (Path(input_dir) / "documents.json").open("r", encoding="utf-8") as f:
            instance.documents = json.load(f)

        if instance.documents:
            instance.build(instance.documents)

        return instance
