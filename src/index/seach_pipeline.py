from __future__ import annotations

"""
Elasticsearch BM25 retriever adapter for the generation pipeline.

This wraps the IRCoT-style ElasticsearchRetriever and exposes the same batched
interface your old FAISS pipeline expected:

    retrieve(queries: list[str], k: int, datasets: list[str] | None = None)

It also normalizes different index schemas:
- IRCoT-style: paragraph_text
- common/indexing-script style: text
"""

from typing import Any, Dict, List, Optional, Sequence
from src.schemas import RetrievedDocument


DEFAULT_DATASET_TO_INDEX = {
    "squad": "wiki",
    "squad_v1": "wiki",
    "natural_questions": "wiki",
    "nq": "wiki",
    "triviaqa": "wiki",
    "trivia_qa": "wiki",
    "hotpotqa": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "musique": "musique",
    "twowikimultihopqa": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "2wiki": "2wikimultihopqa",
}


class BM25ElasticsearchPipelineRetriever:
    def __init__(
        self,
        es_retriever: Any,
        dataset_to_index: Optional[Dict[str, str]] = None,
        default_index: str = "wiki",
        query_title_field_too: bool = True,
        max_buffer_count: int = 100,
    ):
        self.es_retriever = es_retriever
        self.dataset_to_index = dict(DEFAULT_DATASET_TO_INDEX)
        if dataset_to_index:
            self.dataset_to_index.update(dataset_to_index)

        self.default_index = default_index
        self.query_title_field_too = bool(query_title_field_too)
        self.max_buffer_count = int(max_buffer_count)

    def index_for_dataset(self, dataset: Optional[str]) -> str:
        if not dataset:
            return self.default_index
        key = str(dataset).strip().lower()
        return self.dataset_to_index.get(key, self.default_index)

    @staticmethod
    def _source_text(raw: Dict[str, Any]) -> str:
        return str(
            raw.get("paragraph_text")
            or raw.get("text")
            or raw.get("contents")
            or raw.get("passage")
            or raw.get("context")
            or ""
        ).strip()

    @staticmethod
    def _source_title(raw: Dict[str, Any]) -> str:
        return str(
            raw.get("title")
            or raw.get("wikipedia_title")
            or raw.get("page_title")
            or ""
        ).strip()

    @staticmethod
    def _source_id(raw: Dict[str, Any], fallback: int) -> str:
        return str(
            raw.get("id")
            or raw.get("_id")
            or raw.get("doc_id")
            or raw.get("pid")
            or fallback
        )

    def _to_document(self, raw: Dict[str, Any], fallback: int) -> Optional[RetrievedDocument]:
        text = self._source_text(raw)
        if not text:
            return None

        metadata = {
            k: v
            for k, v in raw.items()
            if k not in {"id", "_id", "doc_id", "pid", "title", "paragraph_text", "text", "contents"}
        }

        return RetrievedDocument(
            doc_id=self._source_id(raw, fallback=fallback),
            title=self._source_title(raw),
            text=text,
            score=float(raw.get("score", 0.0) or 0.0),
            metadata=metadata,
        )

    @staticmethod
    def deduplicate_documents(docs: Sequence[RetrievedDocument]) -> List[RetrievedDocument]:
        seen = set()
        output: List[RetrievedDocument] = []

        for doc in docs:
            key = (
                str(getattr(doc, "title", "") or "").strip().lower(),
                str(getattr(doc, "text", "") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(doc)

        return output

    def retrieve_one(
        self,
        query: str,
        dataset: Optional[str] = None,
        k: int = 6,
        allowed_titles: Optional[List[str]] = None,
    ) -> List[RetrievedDocument]:
        index_name = self.index_for_dataset(dataset)

        try:
            raw_results = self.es_retriever.retrieve_paragraphs(
                corpus_name=index_name,
                query_text=query,
                allowed_titles=allowed_titles,
                query_title_field_too=self.query_title_field_too,
                max_buffer_count=max(self.max_buffer_count, k),
                max_hits_count=k,
            )
        except KeyError:
            raw_results = self._retrieve_paragraphs_flexible(
                index_name=index_name,
                query=query,
                k=k,
                allowed_titles=allowed_titles,
            )

        docs: List[RetrievedDocument] = []
        for i, raw in enumerate(raw_results):
            doc = self._to_document(raw, fallback=i)
            if doc is not None:
                docs.append(doc)

        return self.deduplicate_documents(docs)[:k]

    def _retrieve_paragraphs_flexible(
        self,
        index_name: str,
        query: str,
        k: int,
        allowed_titles: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        should = [
            {"match": {"paragraph_text": query}},
            {"match": {"text": query}},
            {"match": {"title": query}},
        ]

        body: Dict[str, Any] = {
            "size": max(self.max_buffer_count, k),
            "_source": True,
            "query": {"bool": {"should": should}},
        }

        if allowed_titles:
            body["query"]["bool"]["filter"] = [
                {"terms": {"title.keyword": allowed_titles}}
            ]

        result = self.es_retriever._es.search(index=index_name, body=body)
        hits = result.get("hits", {}).get("hits", [])

        output: List[Dict[str, Any]] = []
        for hit in hits:
            src = dict(hit.get("_source", {}))
            src["score"] = hit.get("_score", 0.0)
            src.setdefault("corpus_name", index_name)
            output.append(src)

        output = sorted(output, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        return output[:k]

    def retrieve(
        self,
        queries: Sequence[str],
        k: int = 6,
        datasets: Optional[Sequence[Optional[str]]] = None,
    ) -> List[List[RetrievedDocument]]:
        if datasets is None:
            datasets = [None] * len(queries)

        if len(datasets) != len(queries):
            raise ValueError("queries and datasets must have the same length")

        return [
            self.retrieve_one(query=query, dataset=dataset, k=k)
            for query, dataset in zip(queries, datasets)
        ]
