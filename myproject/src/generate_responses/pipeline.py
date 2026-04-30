from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence

from src.schemas import RetrievedDocument
from src.build_index.retriever import FaissIVFRetriever


@dataclass
class PipelineOutput:
    prediction: str
    context: str
    retrieved_docs: List[RetrievedDocument]
    retrieval_count: int
    llm_calls: int
    latency_s: float

def format_passage(doc: RetrievedDocument) -> str:
    title = (doc.title or doc.doc_id or "").strip()
    text = doc.text.strip()
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

        query_batches = self.llm.generate_search_queries(
            questions,
            max_queries=max(1, steps),
        )

        all_docs: List[List[RetrievedDocument]] = []
        retrieval_counts: List[int] = []

        for question, generated_queries in zip(questions, query_batches):
            queries = [question]
            for query in generated_queries:
                if query and query not in queries:
                    queries.append(query)

            docs: List[RetrievedDocument] = []
            retrieval_count = 0

            for query in queries:
                retrieved = self.retriever.retrieve([query], k=k)[0]
                docs.extend(retrieved)
                retrieval_count += len(retrieved)

            selected = FaissIVFRetriever.deduplicate_documents(docs)[:final_k]
            all_docs.append(selected)
            retrieval_counts.append(retrieval_count)

        contexts = [build_context(docs, max_docs=final_k) for docs in all_docs]
        predictions = self.llm.answer(questions, contexts=contexts, strategy="cot")

        latency = (time.perf_counter() - start) / max(1, len(questions))

        return [
            PipelineOutput(
                prediction=prediction,
                context=context,
                retrieved_docs=docs,
                retrieval_count=retrieval_count,
                llm_calls=2,
                latency_s=latency,
            )
            for prediction, context, docs, retrieval_count in zip(
                predictions,
                contexts,
                all_docs,
                retrieval_counts,
            )
        ]
