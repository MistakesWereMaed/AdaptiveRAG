from typing import Any, List, Optional, Union
import re

from vllm import LLM, SamplingParams


class LocalLLM:
    def __init__(self, config: dict):
        model_name = str(config["model_name"])
        tensor_parallel_size = int(config["tensor_parallel_size"])

        self.default_max_new_tokens = int(config["max_new_tokens"])
        self.default_temperature = float(config["temperature"])
        self.default_top_p = float(config["top_p"])
        self.default_use_tqdm = bool(config.get("use_tqdm"))

        llm_kwargs: dict[str, Any] = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": bool(config["trust_remote_code"]),
        }

        dtype = config.get("dtype")
        if dtype:
            llm_kwargs["dtype"] = str(dtype)

        gpu_memory_utilization = config.get("gpu_memory_utilization")
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)

        max_model_len = config.get("max_model_len")
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = int(max_model_len)

        self.llm = LLM(**llm_kwargs)
        self._sampling_params_cls: Any = SamplingParams

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        if context:
            return (
                "You are a question answering system.\n"
                "Answer the question using ONLY the provided context.\n"
                "Return ONLY a short, final answer. No explanation.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
        else:
            return (
                "You are a question answering system.\n"
                "Return ONLY a short, final answer. No explanation.\n\n"
                f"Question: {question}\n"
                "Answer:"
            )

    # --------------------------------------------------
    # Normalization (NEW)
    # --------------------------------------------------
    def _normalize_answer(self, text: str) -> str:
        text = text.lower()

        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)

        # collapse whitespace
        text = " ".join(text.split())

        return text

    def _clean_prediction(self, text: str) -> str:
        text = text.strip()

        # take first line only
        text = text.split("\n")[0].strip()

        # remove trailing punctuation
        text = text.rstrip(".,;: ")

        return text

    def _postprocess(self, text: str) -> str:
        text = self._clean_prediction(text)
        return text

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[str]:

        if isinstance(prompts, str):
            prompts = [prompts]

        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=["\n", "\n\n", "Question:", "Context:"]
        )

        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        results: List[str] = []
        for output in outputs:
            if output.outputs:
                raw = output.outputs[0].text
                cleaned = self._postprocess(raw)
                results.append(cleaned)
            else:
                results.append("")

        return results

    # --------------------------------------------------
    # QA API
    # --------------------------------------------------
    def answer(
        self,
        questions: Union[str, List[str]],
        contexts: Optional[Union[str, List[str]]] = None,
        normalize: bool = False,
        **kwargs,
    ) -> List[str]:

        if isinstance(questions, str):
            questions = [questions]

        if contexts is None:
            contexts = [None] * len(questions)
        elif isinstance(contexts, str):
            contexts = [contexts]

        prompts = [
            self.format_prompt(question, context)
            for question, context in zip(questions, contexts)
        ]

        outputs = self.generate(prompts, **kwargs)

        # optional normalization for evaluation
        if normalize:
            outputs = [self._normalize_answer(o) for o in outputs]

        return outputs