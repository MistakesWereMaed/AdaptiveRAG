from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

from src.schemas import RetrievedDocument
from src.index.utils import FaissIVFRetriever

import re

_STOP_PHRASES = {
    "What", "Which", "Who", "Whom", "Whose", "When", "Where", "Were", "Was",
    "Are", "Is", "Did", "Do", "Does", "The", "A", "An", "In", "On", "Of",
}

def extract_candidate_queries(question: str, max_queries: int = 2) -> list[str]:
    # Captures quoted titles first.
    quoted = re.findall(r'"([^"]+)"', question)

    # Captures title-like spans: Scott Derrickson, Ed Wood, Big Stone Gap.
    titled = re.findall(
        r"\b(?:[A-Z][a-zA-Z0-9'’.-]+(?:\s+|$)){1,5}",
        question,
    )

    candidates = []
    for item in quoted + titled:
        item = " ".join(item.split()).strip(" ?.,;:")
        if not item:
            continue

        first = item.split()[0]
        if first in _STOP_PHRASES:
            continue

        if len(item) < 3:
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

def format_passage(doc, max_chars: int = 400) -> str:
    title = (doc.title or doc.doc_id or "").strip()
    text = doc.text.strip()

    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]

    return f"Wikipedia Title: {title}\n{text}"


def build_context(docs: Sequence[RetrievedDocument], max_docs: int) -> str:
    return "\n\n".join(format_passage(doc) for doc in docs[:max_docs])


class AdaptiveRAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def no_retrieval(self, questions: List[str]) -> List[PipelineOutput]:
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

    def single_step(self, questions: List[str], k: int = 6) -> List[PipelineOutput]:
        start = time.perf_counter()

        batch_docs = self.retriever.retrieve(questions, k=k)
        contexts = [build_context(docs, max_docs=k) for docs in batch_docs]
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
        steps: int = 2,
        k: int = 6,
        final_k: int = 6,
    ) -> List[PipelineOutput]:
        start = time.perf_counter()

        flat_queries: List[str] = []
        owners: List[int] = []

        for i, question in enumerate(questions):
            queries = [question]
            for query in extract_candidate_queries(question, max_queries=max(1, steps)):
                if query and query not in queries:
                    queries.append(query)

            for query in queries:
                flat_queries.append(query)
                owners.append(i)

        flat_results = self.retriever.retrieve(flat_queries, k=k)

        grouped_docs: List[List[RetrievedDocument]] = [[] for _ in questions]
        retrieval_counts = [0 for _ in questions]

        for owner, docs in zip(owners, flat_results):
            grouped_docs[owner].extend(docs)
            retrieval_counts[owner] += len(docs)

        all_docs = []
        contexts = []

        for docs in grouped_docs:
            selected = FaissIVFRetriever.deduplicate_documents(docs)[:final_k]
            all_docs.append(selected)
            contexts.append(build_context(selected, max_docs=final_k))

        predictions = self.llm.answer(
            questions,
            contexts=contexts,
            strategy="cot",
        )

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
