from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.schemas import RetrievedDocument


_STOP_PHRASES = {
    "What", "Which", "Who", "Whom", "Whose", "When", "Where", "Were", "Was",
    "Are", "Is", "Did", "Do", "Does", "The", "A", "An", "In", "On", "Of",
}


def extract_candidate_queries(question: str, max_queries: int = 2) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', question)
    titled = re.findall(r"\b(?:[A-Z][a-zA-Z0-9'’.-]+(?:\s+|$)){1,5}", question)

    candidates = []
    for item in quoted + titled:
        item = " ".join(item.split()).strip(" ?.,;:")
        if not item:
            continue

        first = item.split()[0]
        if first in _STOP_PHRASES or len(item) < 3:
            continue

        if item not in candidates:
            candidates.append(item)

        if len(candidates) >= max_queries:
            break

    return candidates


@dataclass
class PipelineOutput:
    prediction: str
    context: str
    retrieved_docs: List[RetrievedDocument]
    retrieval_count: int
    llm_calls: int
    latency_s: float


def _doc_to_dict(doc: RetrievedDocument) -> Dict[str, Any]:
    return {
        "doc_id": getattr(doc, "doc_id", ""),
        "title": getattr(doc, "title", ""),
        "text": getattr(doc, "text", ""),
        "score": float(getattr(doc, "score", 0.0) or 0.0),
        "metadata": getattr(doc, "metadata", None) or {},
    }


def format_passage(doc: RetrievedDocument, max_chars: int = 900) -> str:
    title = (getattr(doc, "title", "") or getattr(doc, "doc_id", "") or "").strip()
    text = (getattr(doc, "text", "") or "").strip()

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]

    return f"Wikipedia Title: {title}\n{text}"


def build_context(docs: Sequence[RetrievedDocument], max_docs: int, max_chars_per_doc: int = 900) -> str:
    return "\n\n".join(format_passage(doc, max_chars=max_chars_per_doc) for doc in docs[:max_docs])


class AdaptiveRAGPipeline:
    def __init__(self, llm, retriever, max_chars_per_doc: int = 900):
        self.llm = llm
        self.retriever = retriever
        self.max_chars_per_doc = int(max_chars_per_doc)

    @staticmethod
    def serialize_docs(docs: Sequence[RetrievedDocument]) -> List[Dict[str, Any]]:
        return [_doc_to_dict(doc) for doc in docs]

    def no_retrieval(self, questions: List[str], datasets: Optional[List[str]] = None) -> List[PipelineOutput]:
        start = time.perf_counter()
        predictions = self.llm.answer(questions, contexts=None, strategy="direct")
        latency = (time.perf_counter() - start) / max(1, len(questions))

        return [
            PipelineOutput(
                prediction=prediction,
                context="",
                retrieved_docs=[],
                retrieval_count=0,
                llm_calls=1,
                latency_s=latency,
            )
            for prediction in predictions
        ]

    def single_step(
        self,
        questions: List[str],
        datasets: Optional[List[str]] = None,
        k: int = 6,
    ) -> List[PipelineOutput]:
        start = time.perf_counter()

        batch_docs = self.retriever.retrieve(questions, k=k, datasets=datasets)
        contexts = [
            build_context(docs, max_docs=k, max_chars_per_doc=self.max_chars_per_doc)
            for docs in batch_docs
        ]
        predictions = self.llm.answer(questions, contexts=contexts, strategy="direct")

        latency = (time.perf_counter() - start) / max(1, len(questions))

        return [
            PipelineOutput(
                prediction=prediction,
                context=context,
                retrieved_docs=docs,
                retrieval_count=len(docs),
                llm_calls=1,
                latency_s=latency,
            )
            for prediction, context, docs in zip(predictions, contexts, batch_docs)
        ]

    def multi_step(
        self,
        questions: List[str],
        datasets: Optional[List[str]] = None,
        steps: int = 2,
        k: int = 6,
        final_k: int = 6,
    ) -> List[PipelineOutput]:
        start = time.perf_counter()

        if datasets is None:
            datasets = [None] * len(questions)

        flat_queries: List[str] = []
        flat_datasets: List[Optional[str]] = []
        owners: List[int] = []

        for i, (question, dataset) in enumerate(zip(questions, datasets)):
            queries = [question]

            if hasattr(self.llm, "generate_search_queries"):
                try:
                    generated = self.llm.generate_search_queries([question], max_queries=max(1, steps))[0]
                    for query in generated:
                        if query and query not in queries:
                            queries.append(query)
                except Exception:
                    pass

            for query in extract_candidate_queries(question, max_queries=max(1, steps)):
                if query and query not in queries:
                    queries.append(query)

            queries = queries[: max(1, steps + 1)]

            for query in queries:
                flat_queries.append(query)
                flat_datasets.append(dataset)
                owners.append(i)

        flat_results = self.retriever.retrieve(flat_queries, k=k, datasets=flat_datasets)

        grouped_docs: List[List[RetrievedDocument]] = [[] for _ in questions]
        retrieval_counts = [0 for _ in questions]

        for owner, docs in zip(owners, flat_results):
            grouped_docs[owner].extend(docs)
            retrieval_counts[owner] += len(docs)

        all_docs: List[List[RetrievedDocument]] = []
        contexts: List[str] = []

        for docs in grouped_docs:
            selected = self.retriever.deduplicate_documents(docs)[:final_k]
            all_docs.append(selected)
            contexts.append(
                build_context(
                    selected,
                    max_docs=final_k,
                    max_chars_per_doc=self.max_chars_per_doc,
                )
            )

        predictions = self.llm.answer(questions, contexts=contexts, strategy="cot")

        latency = (time.perf_counter() - start) / max(1, len(questions))

        return [
            PipelineOutput(
                prediction=prediction,
                context=context,
                retrieved_docs=docs,
                retrieval_count=retrieval_count,
                llm_calls=1,
                latency_s=latency,
            )
            for prediction, context, docs, retrieval_count in zip(
                predictions,
                contexts,
                all_docs,
                retrieval_counts,
            )
        ]
