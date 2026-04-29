from typing import Any, List, Optional, Union
from vllm import LLM, SamplingParams
from cs6263_template.src.myproject.src.rag.trace import ExecutionTrace


class LocalLLM:
    def __init__(self, config: dict):
        self.model_name = config["model_name"]

        self.default_max_new_tokens = int(config["max_new_tokens"])
        self.default_temperature = float(config["temperature"])
        self.default_top_p = float(config["top_p"])
        self.use_tqdm = bool(config.get("use_tqdm", False))

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=int(config["tensor_parallel_size"]),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            dtype=config.get("dtype", "float16"),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
            max_model_len=int(config.get("max_model_len", 2048)),
        )

    # -----------------------------
    # Prompt
    # -----------------------------
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        if context:
            return (
                "You are a precise question answering system.\n"
                "Answer using ONLY the provided context.\n"
                "If the answer is not contained in the context, respond: unknown.\n"
                "Be concise.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

        return (
            "You are a precise question answering system.\n"
            "Answer the question concisely.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    # -----------------------------
    # Generation
    # -----------------------------
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        trace: Optional[Union[ExecutionTrace, List[ExecutionTrace]]] = None,
    ) -> List[str]:

        if isinstance(prompts, str):
            prompts = [prompts]

        if trace is not None:
            if isinstance(trace, list):
                for t in trace:
                    t.record_llm_call(1)
            else:
                trace.record_llm_call(1)

        sampling = SamplingParams(
            temperature=temperature or self.default_temperature,
            top_p=top_p or self.default_top_p,
            max_tokens=max_new_tokens or self.default_max_new_tokens,
            stop=["\n\n", "Question:", "Context:"],
        )

        outputs = self.llm.generate(prompts, sampling_params=sampling, use_tqdm=self.use_tqdm)

        return [o.outputs[0].text if o.outputs else "" for o in outputs]

    # -----------------------------
    # QA wrapper
    # -----------------------------
    def answer(
        self,
        questions: List[str],
        contexts: Optional[List[str]] = None,
        trace: Optional[ExecutionTrace] = None,
    ) -> List[str]:

        if contexts is None:
            contexts = [None] * len(questions)

        prompts = [
            self.format_prompt(q, c)
            for q, c in zip(questions, contexts)
        ]

        return self.generate(prompts, trace=trace)