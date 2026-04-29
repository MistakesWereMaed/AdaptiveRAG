from __future__ import annotations

from typing import List, Optional, Tuple, Union

from src.data.schemas import RetrievedDocument
from src.rag.context import deduplicate_documents
from src.rag.trace import ExecutionTrace


PipelineResult = Tuple[
    str,
    ExecutionTrace,
    str,
    List[RetrievedDocument],
    dict,
]


class AdaptiveRAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
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

    # --------------------------------------------------
    # No-RAG
    # Prompt:
    # Q: [Question]
    # A:
    # --------------------------------------------------
    def no_retrieval(
        self,
        questions: List[str],
        return_traces: bool = False,
        return_debug: bool = False,
        start_query_id: int = 0,
    ) -> Union[List[str], List[PipelineResult]]:
        traces = self._make_traces(questions, start_query_id, return_traces)

        if return_debug:
            outputs, debug_rows = self.llm.answer(
                questions,
                contexts=None,
                strategy="direct",
                trace=traces,
                return_debug=True,
            )
        else:
            outputs = self.llm.answer(
                questions,
                contexts=None,
                strategy="direct",
                trace=traces,
                return_debug=False,
            )
            debug_rows = [{} for _ in outputs]

        if not return_traces:
            return outputs

        results: List[PipelineResult] = []

        for i, output in enumerate(outputs):
            traces[i].finalize()
            results.append((output, traces[i], "", [], debug_rows[i]))

        return results

    # --------------------------------------------------
    # Single-hop RAG
    # Prompt:
    # Wikipedia Title: [title]
    # [text]
    #
    # Q: [Question]
    # A:
    # --------------------------------------------------
    def single_step(
        self,
        questions: List[str],
        k: int = 6,
        return_traces: bool = False,
        return_debug: bool = False,
        start_query_id: int = 0,
    ) -> Union[List[str], List[PipelineResult]]:
        traces = self._make_traces(questions, start_query_id, return_traces)

        batch_docs = self.retriever.retrieve(
            questions,
            k=k,
            trace=traces,
        )

        contexts = [
            self._format_docs_for_flan(docs, max_docs=k)
            for docs in batch_docs
        ]

        if return_debug:
            outputs, debug_rows = self.llm.answer(
                questions,
                contexts=contexts,
                strategy="direct",
                trace=traces,
                return_debug=True,
            )
        else:
            outputs = self.llm.answer(
                questions,
                contexts=contexts,
                strategy="direct",
                trace=traces,
                return_debug=False,
            )
            debug_rows = [{} for _ in outputs]

        if not return_traces:
            return outputs

        results: List[PipelineResult] = []

        for i, output in enumerate(outputs):
            traces[i].finalize()
            results.append(
                (
                    output,
                    traces[i],
                    contexts[i],
                    batch_docs[i],
                    debug_rows[i],
                )
            )

        return results

    # --------------------------------------------------
    # Multi-hop RAG with CoT
    #
    # Prompt:
    # Wikipedia Title: [Passage 1 title]
    # [Passage 1 text]
    #
    # Wikipedia Title: [Passage 2 title]
    # [Passage 2 text]
    #
    # Q: Answer the following question by reasoning step-by-step.
    # [Question]
    # A:
    #
    # Output extractor:
    # "answer is ..."
    # --------------------------------------------------
    def multi_step(
        self,
        questions: List[str],
        steps: int = 2,
        k: int = 6,
        final_k: int = 6,
        return_traces: bool = False,
        return_debug: bool = False,
        start_query_id: int = 0,
    ) -> Union[List[str], List[PipelineResult]]:
        traces = self._make_traces(questions, start_query_id, return_traces)

        if return_debug:
            query_batches, query_debug_rows = self.llm.generate_search_queries(
                questions,
                trace=traces,
                max_queries=max(1, steps),
                return_debug=True,
            )
        else:
            query_batches = self.llm.generate_search_queries(
                questions,
                trace=traces,
                max_queries=max(1, steps),
                return_debug=False,
            )
            query_debug_rows = [{} for _ in questions]

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

            query_debug_rows[i]["search_queries_used"] = search_queries
            query_debug_rows[i]["num_retrieved_before_dedup"] = len(retrieved_docs)
            query_debug_rows[i]["num_retrieved_after_dedup"] = len(fused_docs)
            query_debug_rows[i]["num_docs_in_final_context"] = len(selected_docs)

        if return_debug:
            outputs, answer_debug_rows = self.llm.answer(
                questions,
                contexts=contexts,
                strategy="cot",
                trace=traces,
                return_debug=True,
            )
        else:
            outputs = self.llm.answer(
                questions,
                contexts=contexts,
                strategy="cot",
                trace=traces,
                return_debug=False,
            )
            answer_debug_rows = [{} for _ in outputs]

        debug_rows = []
        for q_debug, a_debug in zip(query_debug_rows, answer_debug_rows):
            row = {}
            row.update(q_debug)
            row.update(a_debug)
            debug_rows.append(row)

        if not return_traces:
            return outputs

        results: List[PipelineResult] = []

        for i, output in enumerate(outputs):
            traces[i].finalize()
            results.append(
                (
                    output,
                    traces[i],
                    contexts[i],
                    batch_docs[i],
                    debug_rows[i],
                )
            )

        return results