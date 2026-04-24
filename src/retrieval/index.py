import os
from pathlib import Path
from typing import List, Sequence

import faiss
from sentence_transformers import SentenceTransformer


class FaissRetrieverIndex:
	def __init__(self, encoder_name: str = "all-MiniLM-L6-v2"):
		self.encoder = SentenceTransformer(encoder_name)
		self.index = None
		self.documents: List[str] = []
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
			raise ValueError("The FAISS index has not been built yet")
		return faiss.index_gpu_to_cpu(self.index)

	def build(self, documents: Sequence[str]) -> None:
		self.documents = [document for document in documents if document]
		if not self.documents:
			raise ValueError("No documents were provided for indexing")

		embeddings = self.encoder.encode(self.documents, convert_to_numpy=True, show_progress_bar=True)
		embeddings = embeddings.astype("float32")

		cpu_index = faiss.IndexFlatL2(embeddings.shape[1])
		self.index = self._to_gpu_index(cpu_index)
		self.index.add(embeddings)

	def search(self, query: str, k: int = 5) -> List[str]:
		if self.index is None:
			raise ValueError("The FAISS index has not been built yet")

		query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
		_, indices = self.index.search(query_embedding, k)
		return [self.documents[index] for index in indices[0] if 0 <= index < len(self.documents)]

	def save(self, output_dir: str | Path) -> None:
		if self.index is None:
			raise ValueError("The FAISS index has not been built yet")

		output_path = Path(output_dir)
		output_path.mkdir(parents=True, exist_ok=True)
		faiss.write_index(self._to_cpu_index(), str(output_path / "index.faiss"))

		import json

		with (output_path / "documents.json").open("w", encoding="utf-8") as handle:
			json.dump(self.documents, handle, indent=2)

	@classmethod
	def load(cls, input_dir: str | Path, encoder_name: str = "all-MiniLM-L6-v2") -> "FaissRetrieverIndex":
		import json

		input_path = Path(input_dir)
		instance = cls(encoder_name=encoder_name)
		cpu_index = faiss.read_index(str(input_path / "index.faiss"))
		instance.index = instance._to_gpu_index(cpu_index)
		with (input_path / "documents.json").open("r", encoding="utf-8") as handle:
			instance.documents = json.load(handle)
		return instance

