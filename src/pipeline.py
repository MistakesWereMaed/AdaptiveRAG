from typing import List, Dict
from tqdm.auto import tqdm


class AdaptiveRAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    # --------------------------------------------------
    # Strategy 0: No Retrieval
    # --------------------------------------------------
    def no_retrieval(self, questions: List[str]) -> List[str]:
        return self.llm.answer(questions)

    # --------------------------------------------------
    # Strategy 1: Single-step RAG (Qdrant)
    # --------------------------------------------------
    def single_step(self, questions: List[str], k: int = 5) -> List[str]:
        contexts = []

        for q in tqdm(questions, desc="Retrieving", unit="query"):
            docs = self.retriever.retrieve(q, k=k)
            context = "\n".join(docs)
            contexts.append(context)

        return self.llm.answer(questions, contexts)

    # --------------------------------------------------
    # Strategy 2: Multi-step RAG (Qdrant)
    # --------------------------------------------------
    def multi_step(self, questions: List[str], steps: int = 2, k: int = 3) -> List[str]:
        all_contexts = [[] for _ in questions]

        for step in tqdm(range(steps), desc="Multi-step retrieval", unit="step"):

            # -----------------------------------------
            # Build queries (same logic as before)
            # -----------------------------------------
            if step == 0:
                queries = questions
            else:
                queries = [
                    q + " " + " ".join(all_contexts[i])
                    for i, q in enumerate(questions)
                ]

            # -----------------------------------------
            # Qdrant retrieval (batch loop but simple API)
            # -----------------------------------------
            batch_docs = [
                self.retriever.retrieve(q, k=k)
                for q in queries
            ]

            # -----------------------------------------
            # Accumulate contexts
            # -----------------------------------------
            for i, docs in enumerate(batch_docs):
                all_contexts[i].extend(docs)

        final_contexts = [
            "\n".join(ctx) for ctx in all_contexts
        ]

        return self.llm.answer(questions, final_contexts)

    # --------------------------------------------------
    # Run all strategies (for labeling)
    # --------------------------------------------------
    def run_all(self, questions: List[str]) -> Dict[str, List[str]]:
        outputs = {}

        outputs["no"] = self.no_retrieval(questions)
        outputs["single"] = self.single_step(questions)
        outputs["multi"] = self.multi_step(questions)

        return outputs