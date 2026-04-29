from typing import List, Union, Optional, Tuple
from cs6263_template.src.myproject.src.rag.trace import ExecutionTrace


class AdaptiveRAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    # --------------------------------------------------
    # No-RAG
    # --------------------------------------------------
    def no_retrieval(
        self,
        questions: List[str],
        return_traces: bool = False,
        start_query_id: int = 0,
    ) -> Union[List[str], List[Tuple[str, ExecutionTrace, str]]]:

        if not return_traces:
            return self.llm.answer(questions)

        traces = [ExecutionTrace(start_query_id + i, q) for i, q in enumerate(questions)]
        results = self.llm.answer(questions, trace=traces)

        for t in traces:
            t.finalize()

        # include empty context for no-retrieval
        return [(r, t, "") for r, t in zip(results, traces)]

    # --------------------------------------------------
    # Single-step RAG (cleaned)
    # --------------------------------------------------
    def single_step(
        self,
        questions: List[str],
        k: int = 5,
        return_traces: bool = False,
        start_query_id: int = 0,
    ) -> Union[List[str], List[Tuple[str, ExecutionTrace, str]]]:

        traces = None
        if return_traces:
            traces = [ExecutionTrace(start_query_id + i, q) for i, q in enumerate(questions)]

        batch_docs = self.retriever.retrieve(questions, k=k, trace=traces)

        contexts = [
            "\n".join(docs[:k]) if docs else ""
            for docs in batch_docs
        ]

        outputs = self.llm.answer(questions, contexts, trace=traces)

        if not return_traces:
            return outputs

        for t in traces:
            t.finalize()

        return [(out, traces[i], contexts[i]) for i, out in enumerate(outputs)]

    # --------------------------------------------------
    # Multi-step RAG (FIXED + paper-aligned)
    # --------------------------------------------------
    def multi_step(
        self,
        questions: List[str],
        steps: int = 2,
        k: int = 4,
        return_traces: bool = False,
        start_query_id: int = 0,
    ) -> Union[List[str], List[Tuple[str, ExecutionTrace, str]]]:

        n = len(questions)

        traces = None
        if return_traces:
            traces = [ExecutionTrace(start_query_id + i, q) for i, q in enumerate(questions)]

        # -----------------------------
        # structured memory (FIXED)
        # -----------------------------
        memory_docs = [[] for _ in range(n)]
        memory_answers = [[] for _ in range(n)]

        batch_docs = [[] for _ in range(n)]

        # -----------------------------
        # multi-step loop
        # -----------------------------
        for step in range(steps):

            # ---- query construction (bounded) ----
            if step == 0:
                queries = questions
            else:
                queries = [
                    (questions[i] + " " + (memory_answers[i][-1] if memory_answers[i] else ""))[:256]
                    for i in range(n)
                ]

            # ---- retrieval ----
            batch_docs = self.retriever.retrieve(queries, k=k, trace=traces)

            # ---- build step context (FIXED: includes memory) ----
            step_contexts = []
            for i in range(n):
                context_parts = (
                    memory_docs[i][-4:] +
                    memory_answers[i][-2:] +
                    batch_docs[i]
                )

                step_contexts.append("\n".join(context_parts))

            step_prompts = [
                self.llm.format_prompt(questions[i], step_contexts[i])
                for i in range(n)
            ]

            # ---- intermediate generation (skip last step) ----
            if step < steps - 1:
                step_outputs = self.llm.generate(step_prompts, trace=traces)

                for i in range(n):
                    memory_answers[i].append(step_outputs[i])
                    memory_docs[i].extend(batch_docs[i])

                    # bounded memory (prevents explosion)
                    memory_answers[i] = memory_answers[i][-2:]
                    memory_docs[i] = memory_docs[i][-6:]

        # -----------------------------
        # FINAL STEP
        # -----------------------------
        final_contexts = [
            "\n".join(memory_docs[i][-6:] + memory_answers[i][-2:])
            for i in range(n)
        ]

        final_outputs = self.llm.answer(questions, final_contexts, trace=traces)

        if not return_traces:
            return final_outputs

        results = []
        for i, out in enumerate(final_outputs):
            traces[i].finalize()
            # include final context for downstream inspection
            results.append((out, traces[i], final_contexts[i]))

        return results