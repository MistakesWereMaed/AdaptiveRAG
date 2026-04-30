from __future__ import annotations

import re
import torch

from typing import List, Optional, Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ANSWER_RE = re.compile(r".*answer is:?\s*(.*)", re.IGNORECASE | re.DOTALL)


def clean_answer(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^(answer|a)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.split("\n")[0].strip()
    return text.strip(" .")


def extract_cot_answer(text: str) -> str:
    text = str(text or "").strip()
    match = ANSWER_RE.match(text)
    if match:
        return clean_answer(match.group(1))
    return clean_answer(text)


def parse_generated_queries(text: str, fallback: str, max_queries: int = 2) -> List[str]:
    queries: List[str] = []

    for line in str(text or "").splitlines():
        line = line.strip()
        line = line.lstrip("-•* ")
        line = line.lstrip("0123456789. ").strip()

        if not line:
            continue

        if line.lower().startswith(("q:", "a:", "question:", "answer:")):
            continue

        queries.append(line)

        if len(queries) >= max_queries:
            break

    return queries or [fallback]


class LocalLLM:
    """Minimal FLAN-T5 wrapper for Adaptive-RAG response generation."""

    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "google/flan-t5-xl")
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        dtype_name = str(config.get("dtype", "float16")).lower()
        if self.device.type == "cuda" and dtype_name in {"float16", "fp16", "half"}:
            dtype = torch.float16
        elif self.device.type == "cuda" and dtype_name in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        self.batch_size = int(config.get("batch_size", 8))
        self.max_input_length = int(config.get("max_input_length", 2048))
        self.max_new_tokens = int(config.get("max_new_tokens", 32))
        self.cot_max_new_tokens = int(config.get("cot_max_new_tokens", 128))
        self.query_max_new_tokens = int(config.get("query_max_new_tokens", 48))
        self.num_beams = int(config.get("num_beams", 1))
        self.do_sample = bool(config.get("do_sample", False))
        self.temperature = float(config.get("temperature", 1.0))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    # --------------------------------------------------
    # Prompt formats
    # --------------------------------------------------
    @staticmethod
    def direct_prompt(question: str) -> str:
        return f"Q: {question}\nA:"

    @staticmethod
    def context_prompt(question: str, context: str) -> str:
        return f"{context}\n\nQ: {question}\nA:"

    @staticmethod
    def cot_prompt(question: str, context: str) -> str:
        return (
            f"{context}\n\n"
            "Q: Answer the following question by reasoning step-by-step.\n"
            f"{question}\n"
            "A:"
        )

    @staticmethod
    def query_prompt(question: str) -> str:
        return (
            "Generate short Wikipedia search queries for the following question.\n"
            "Return one query per line.\n\n"
            f"Q: {question}\n"
            "A:"
        )

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        outputs: List[str] = []

        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
            ).to(self.device)

            kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": self.num_beams,
                "do_sample": self.do_sample,
            }
            if self.do_sample:
                kwargs["temperature"] = self.temperature

            generated = self.model.generate(**encoded, **kwargs)
            outputs.extend(
                self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )

        return outputs

    def answer(
        self,
        questions: List[str],
        contexts: Optional[List[Optional[str]]] = None,
        strategy: str = "direct",
    ) -> List[str]:
        contexts = contexts or [None] * len(questions)

        if len(contexts) != len(questions):
            raise ValueError("questions and contexts must have the same length")

        prompts = []
        for question, context in zip(questions, contexts):
            if context and strategy == "cot":
                prompts.append(self.cot_prompt(question, context))
            elif context:
                prompts.append(self.context_prompt(question, context))
            else:
                prompts.append(self.direct_prompt(question))

        max_tokens = self.cot_max_new_tokens if strategy == "cot" else self.max_new_tokens
        raw = self.generate(prompts, max_new_tokens=max_tokens)

        if strategy == "cot":
            cleaned = [extract_cot_answer(x) for x in raw]
        else:
            cleaned = [clean_answer(x) for x in raw]

        return [x if x else "unknown" for x in cleaned]

    def generate_search_queries(self, questions: List[str], max_queries: int = 2) -> List[List[str]]:
        prompts = [self.query_prompt(question) for question in questions]
        raw = self.generate(prompts, max_new_tokens=self.query_max_new_tokens)

        return [
            parse_generated_queries(text, fallback=question, max_queries=max_queries)
            for text, question in zip(raw, questions)
        ]
