from typing import List, Union
from src.rag.trace import ExecutionTrace


class AdaptiveRAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever


    # --------------------------------------------------
    # Strategy 0: No Retrieval
    # --------------------------------------------------
    def no_retrieval(self, questions: List[str], return_traces: bool = False, start_query_id: int = 0) -> Union[List[str], List[tuple]]:
        """Batched no-retrieval. Returns list[str] by default, or list[(str, ExecutionTrace)] when return_traces=True."""
        if not return_traces:
            return self.llm.answer(questions)

        traces = [ExecutionTrace(query_id=start_query_id + i, question=q) for i, q in enumerate(questions)]
        results = self.llm.answer(questions, trace=traces)

        paired = []
        for i, r in enumerate(results):
            traces[i].finalize()
            paired.append((r, traces[i]))

        return paired


    # --------------------------------------------------
    # Strategy 1: Single-step
    # --------------------------------------------------
    def single_step(self, questions: List[str], k: int = 5, return_traces: bool = False, start_query_id: int = 0) -> Union[List[str], List[tuple]]:
        """Batched single-step. Returns list[str] by default, or list[(str, ExecutionTrace)] when return_traces=True."""
        if not return_traces:
            # existing behavior
            print("Retrieving contexts...", flush=True)
            batch_docs = self.retriever.retrieve(questions, k=k)

            contexts: List[str] = []
            for docs in batch_docs:
                trimmed = docs[:k]
                if len(trimmed) <= 1:
                    context = trimmed[0] if trimmed else ""
                else:
                    context = "\n".join(trimmed)
                contexts.append(context)

            return self.llm.answer(questions, contexts)

        # return_traces == True
        traces = [ExecutionTrace(query_id=start_query_id + i, question=q) for i, q in enumerate(questions)]
        batch_docs = self.retriever.retrieve(questions, k=k, trace=traces)

        contexts: List[str] = []
        for docs in batch_docs:
            trimmed = docs[:k]
            if len(trimmed) <= 1:
                context = trimmed[0] if trimmed else ""
            else:
                context = "\n".join(trimmed)
            contexts.append(context)

        results = self.llm.answer(questions, contexts=contexts, trace=traces)

        paired = []
        for i, r in enumerate(results):
            traces[i].finalize()
            paired.append((r, traces[i]))

        return paired


    # --------------------------------------------------
    # Strategy 2: Multi-step
    # --------------------------------------------------
    def multi_step(
        self,
        questions: List[str],
        steps: int = 2,
        k: int = 3,
        return_traces: bool = False,
        start_query_id: int = 0,
    ):
        """
        Optimized multi-step RAG with ExecutionTrace support.

        - Reduced prompt size
        - No document feedback into retriever
        - Fewer generations (last step skips intermediate)
        - Maintains trace logging
        """

        n = len(questions)

        # ---- Initialize traces if needed ----
        if return_traces:
            traces = [
                ExecutionTrace(query_id=start_query_id + i, question=questions[i])
                for i in range(n)
            ]
        else:
            traces = None

        # ---- Store only intermediate answers ----
        memory_answers = [[] for _ in range(n)]

        # ---- Multi-step loop ----
        for step in range(steps):

            # ---- Build retrieval queries ----
            if step == 0:
                queries = questions
            else:
                queries = [
                    questions[i] + " " + " ".join(memory_answers[i][-1:])
                    if memory_answers[i]
                    else questions[i]
                    for i in range(n)
                ]

            # ---- Retrieval ----
            if return_traces:
                batch_docs = self.retriever.retrieve(queries, k=k, trace=traces)
            else:
                batch_docs = self.retriever.retrieve(queries, k=k)

            # ---- Build prompts (minimal context) ----
            step_prompts = [
                (
                    "Answer the question using the context.\n"
                    "Return a short answer.\n\n"
                    f"Question: {questions[i]}\n"
                    f"Context:\n{chr(10).join(batch_docs[i])}\n\n"
                    "Answer:"
                )
                for i in range(n)
            ]

            # ---- Skip generation on final step ----
            if step < steps - 1:
                if return_traces:
                    step_answers = self.llm.generate(step_prompts, trace=traces)
                else:
                    step_answers = self.llm.generate(step_prompts)

                # ---- Update memory ----
                for i in range(n):
                    memory_answers[i].append(step_answers[i])

                    # keep only last 2 answers (tight bound)
                    memory_answers[i] = memory_answers[i][-2:]

        # ---- Final generation ----
        final_prompts = [
            (
                "Answer the question using the context.\n"
                "Provide ONLY the final answer.\n\n"
                f"Question: {questions[i]}\n"
                f"Context:\n{chr(10).join(batch_docs[i])}\n\n"
                "Answer:"
            )
            for i in range(n)
        ]

        if return_traces:
            final_results = self.llm.generate(final_prompts, trace=traces)

            paired = []
            for i, r in enumerate(final_results):
                traces[i].finalize()
                paired.append((r, traces[i]))

            return paired

        return self.llm.generate(final_prompts)