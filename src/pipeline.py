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
    # Strategy 1: Single-step
    # --------------------------------------------------
    def single_step(self, questions: List[str], k: int = 5) -> List[str]:
        print("Retrieving contexts (batched)...", flush=True)

        batch_docs = self.retriever.retrieve(questions, k=k)

        contexts = [
            "\n".join(docs)
            for docs in batch_docs
        ]

        print("Generating answers...", flush=True)
        return self.llm.answer(questions, contexts)

    # --------------------------------------------------
    # Strategy 2: Multi-step
    # --------------------------------------------------
    def multi_step(self, questions: List[str], steps: int = 2, k: int = 3) -> List[str]:
        all_contexts = [[] for _ in questions]

        for step in tqdm(range(steps), desc="Multi-step retrieval", unit="step"):

            if step == 0:
                queries = questions
            else:
                queries = [
                    q + " " + " ".join(all_contexts[i])
                    for i, q in enumerate(questions)
                ]

            batch_docs = self.retriever.retrieve(queries, k=k)

            for i, docs in enumerate(batch_docs):
                all_contexts[i].extend(docs)

        final_contexts = [
            "\n".join(ctx)
            for ctx in all_contexts
        ]

        return self.llm.answer(questions, final_contexts)

    # --------------------------------------------------
    # Run all strategies
    # --------------------------------------------------
    def run_all(self, questions: List[str]) -> Dict[str, List[str]]:
        return {
            "no": self.no_retrieval(questions),
            "single": self.single_step(questions),
            "multi": self.multi_step(questions),
        }