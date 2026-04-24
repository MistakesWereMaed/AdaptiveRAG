import os
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(encoder_name)
        self.index = None
        self.texts = []
        self._gpu_resources = None

    def _get_gpu_device(self) -> int:
        gpu_count = faiss.get_num_gpus()
        if gpu_count <= 0 or not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("FAISS GPU is required but no GPU resources were detected")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return local_rank % gpu_count

    def _to_gpu_index(self, cpu_index):
        device = self._get_gpu_device()
        self._gpu_resources = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(self._gpu_resources, device, cpu_index)

    def _to_cpu_index(self):
        if self.index is None:
            raise ValueError("The retrieval index has not been built yet")
        return faiss.index_gpu_to_cpu(self.index)

    def build_index(self, documents):
        if not documents:
            raise ValueError("No documents were provided for indexing")
        self.texts = documents
        embeddings = self.encoder.encode(documents, convert_to_numpy=True, show_progress_bar=True).astype("float32")

        dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(dim)
        self.index = self._to_gpu_index(cpu_index)
        self.index.add(embeddings)

    def save_index(self, output_dir):
        if self.index is None:
            raise ValueError("The retrieval index has not been built yet")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._to_cpu_index(), str(output_path / "index.faiss"))
        with (output_path / "documents.json").open("w", encoding="utf-8") as handle:
            json.dump(self.texts, handle, indent=2)

    def load_index(self, input_dir):
        input_path = Path(input_dir)
        index_path = input_path / "index.faiss"
        documents_path = input_path / "documents.json"

        if not index_path.exists() or not documents_path.exists():
            raise FileNotFoundError(
                f"Missing index files in {input_path}. Expected {index_path.name} and {documents_path.name}."
            )

        cpu_index = faiss.read_index(str(index_path))
        self.index = self._to_gpu_index(cpu_index)
        with documents_path.open("r", encoding="utf-8") as handle:
            self.texts = json.load(handle)

    def retrieve(self, query, k=5):
        if self.index is None:
            raise ValueError("The retrieval index has not been built yet")

        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_emb, k)
        return [self.texts[i] for i in indices[0]]