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
    # Strategy 1: Single-step RAG
    # --------------------------------------------------
    def single_step(self, questions: List[str], k: int = 5) -> List[str]:
        # -----------------------------
        # 1. Encode all queries (progress bar)
        # -----------------------------
        q_embeddings = self.retriever.encoder.encode(
            questions,
            convert_to_numpy=True,
            batch_size=256,
            show_progress_bar=True  # sentence-transformers internal tqdm
        )

        # -----------------------------
        # 2. FAISS retrieval
        # -----------------------------
        _, I = self.retriever.index.search(q_embeddings, k)

        # -----------------------------
        # 3. Build contexts
        # -----------------------------
        contexts = []

        for idxs in tqdm(I, desc="Building contexts", unit="query"):
            docs = [
                self.retriever.texts[int(i)]
                for i in idxs
                if 0 <= int(i) < len(self.retriever.texts)
            ]
            context = "\n".join(docs)
            contexts.append(context)

        # -----------------------------
        # 4. LLM generation
        # -----------------------------
        return self.llm.answer(questions, contexts)

    # --------------------------------------------------
    # Strategy 2: Multi-step RAG (batched)
    # --------------------------------------------------
    def multi_step(self, questions: List[str], steps: int = 2, k: int = 3) -> List[str]:
        all_contexts = [[] for _ in questions]

        for step in tqdm(range(steps), desc="Multi-step retrieval", unit="step"):
            queries = []

            for i, q in enumerate(questions):
                if step == 0:
                    queries.append(q)
                else:
                    partial_context = " ".join(all_contexts[i])
                    queries.append(f"{q} {partial_context}")

            # batch retrieval
            batch_docs = [self.retriever.retrieve(q, k=k) for q in queries]

            for i, docs in enumerate(batch_docs):
                all_contexts[i].extend(docs)

        final_contexts = ["\n".join(ctx) for ctx in all_contexts]

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