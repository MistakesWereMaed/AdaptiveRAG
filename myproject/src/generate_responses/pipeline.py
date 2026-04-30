from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from myproject.src.schemas import RetrievedDocument
from myproject.src.generate_responses.context import deduplicate_documents


@dataclass
class PipelineOutput:
    prediction: str
    context: str
    retrieved_docs: List[RetrievedDocument]
    retrieval_count: int
    llm_calls: int
    latency_s: float


class AdaptiveRAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def _make_traces(
        self,
        questions: List[str],
        start_query_id: int,
        enabled: bool,
    ) -> Optional[List[ExecutionTrace]]:
        if not enabled:
            return None

        return [
            ExecutionTrace(start_query_id + i, question)
            for i, question in enumerate(questions)
        ]

    @staticmethod
    def _trace_at(
        traces: Optional[List[ExecutionTrace]],
        index: int,
    ) -> Optional[ExecutionTrace]:
        if traces is None:
            return None
        return traces[index]

    @staticmethod
    def _format_docs_for_flan(docs: List[RetrievedDocument], max_docs: int) -> str:
        blocks = []

        for doc in docs[:max_docs]:
            title = str(getattr(doc, "title", "") or "").strip()
            text = str(getattr(doc, "text", "") or "").strip()

            if not text:
                continue

            blocks.append(f"Wikipedia Title: {title}\n{text}")

        return "\n\n".join(blocks)

    def no_retrieval(
        self,
        questions: List[str],
        start_query_id: int = 0,
    ) -> List[PipelineOutput]:
        traces = self._make_traces(questions, start_query_id, enabled=True)
        outputs = self.llm.answer(
            questions,
            contexts=None,
            strategy="direct",
            trace=traces,
        )

        results: List[PipelineOutput] = []
        for i, output in enumerate(outputs):
            trace = traces[i]
            trace.finalize()
            results.append(
                PipelineOutput(
                    prediction=output,
                    context="",
                    retrieved_docs=[],
                    retrieval_count=trace.retrieval_count,
                    llm_calls=trace.llm_call_count,
                    latency_s=trace.latency_s,
                )
            )

        return results

    def single_step(
        self,
        questions: List[str],
        k: int = 6,
        start_query_id: int = 0,
    ) -> List[PipelineOutput]:
        traces = self._make_traces(questions, start_query_id, enabled=True)

        batch_docs = self.retriever.retrieve(
            questions,
            k=k,
            trace=traces,
        )

        contexts = [
            self._format_docs_for_flan(docs, max_docs=k)
            for docs in batch_docs
        ]

        outputs = self.llm.answer(
            questions,
            contexts=contexts,
            strategy="direct",
            trace=traces,
        )

        results: List[PipelineOutput] = []

        for i, output in enumerate(outputs):
            trace = traces[i]
            trace.finalize()
            results.append(
                PipelineOutput(
                    prediction=output,
                    context=contexts[i],
                    retrieved_docs=batch_docs[i],
                    retrieval_count=trace.retrieval_count,
                    llm_calls=trace.llm_call_count,
                    latency_s=trace.latency_s,
                )
            )

        return results

    def multi_step(
        self,
        questions: List[str],
        steps: int = 2,
        k: int = 6,
        final_k: int = 6,
        start_query_id: int = 0,
    ) -> List[PipelineOutput]:
        traces = self._make_traces(questions, start_query_id, enabled=True)

        query_batches = self.llm.generate_search_queries(
            questions,
            trace=traces,
            max_queries=max(1, steps),
        )

        batch_docs: List[List[RetrievedDocument]] = []
        contexts: List[str] = []

        for i, (question, generated_queries) in enumerate(zip(questions, query_batches)):
            current_trace = self._trace_at(traces, i)

            search_queries = [question]
            for query in generated_queries:
                if query and query not in search_queries:
                    search_queries.append(query)

            retrieved_docs: List[RetrievedDocument] = []

            for search_query in search_queries:
                docs = self.retriever.retrieve(
                    [search_query],
                    k=k,
                    trace=current_trace,
                )[0]
                retrieved_docs.extend(docs)

            fused_docs = deduplicate_documents(retrieved_docs)
            selected_docs = fused_docs[:final_k]

            context = self._format_docs_for_flan(selected_docs, max_docs=final_k)

            batch_docs.append(selected_docs)
            contexts.append(context)

        outputs = self.llm.answer(
            questions,
            contexts=contexts,
            strategy="cot",
            trace=traces,
        )

        results: List[PipelineOutput] = []

        for i, output in enumerate(outputs):
            trace = traces[i]
            trace.finalize()
            results.append(
                PipelineOutput(
                    prediction=output,
                    context=contexts[i],
                    retrieved_docs=batch_docs[i],
                    retrieval_count=trace.retrieval_count,
                    llm_calls=trace.llm_call_count,
                    latency_s=trace.latency_s,
                )
            )

        return results
