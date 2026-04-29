from __future__ import annotations

import re
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.rag.postprocess import clean_answer
from src.rag.trace import ExecutionTrace


ANSWER_IS_RE = re.compile(r".*answer is:?\s*(.*)", re.IGNORECASE | re.DOTALL)


def extract_cot_answer(text: str) -> str:
    if not text:
        return ""

    match = ANSWER_IS_RE.match(text.strip())
    if match:
        return clean_answer(match.group(1))

    return clean_answer(text)


def parse_generated_queries(
    text: str,
    original_question: str,
    max_queries: int = 2,
) -> List[str]:
    queries: List[str] = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = line.lstrip("-•* ")
        line = line.lstrip("0123456789. ")
        line = line.strip()

        if line.lower().startswith(("q:", "a:", "question:", "answer:")):
            continue

        queries.append(line)

        if len(queries) >= max_queries:
            break

    return queries or [original_question]


class LocalLLM:
    """
    FLAN-T5 / T5 seq2seq wrapper using the paper-style prompt formats.

    Supported strategies:
      - no_context_direct
      - gold_context_direct
      - cot
      - query_generation
    """

    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "google/flan-t5-xl")
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        dtype_name = str(config.get("dtype", "float16")).lower()
        if dtype_name in {"fp16", "float16", "half"} and self.device.type == "cuda":
            self.dtype = torch.float16
        elif dtype_name in {"bf16", "bfloat16"} and self.device.type == "cuda":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.default_max_new_tokens = int(config.get("max_new_tokens", 32))
        self.cot_max_new_tokens = int(config.get("cot_max_new_tokens", 128))
        self.query_max_new_tokens = int(config.get("query_max_new_tokens", 48))
        self.default_num_beams = int(config.get("num_beams", 1))
        self.default_do_sample = bool(config.get("do_sample", False))
        self.default_temperature = float(config.get("temperature", 1.0))
        self.batch_size = int(config.get("batch_size", 8))
        self.max_input_length = int(config.get("max_input_length", 2048))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    # --------------------------------------------------
    # Trace helpers
    # --------------------------------------------------
    def _record_llm_call(
        self,
        trace: Optional[Union[ExecutionTrace, List[ExecutionTrace]]],
        num_calls: int = 1,
    ) -> None:
        if trace is None:
            return

        if isinstance(trace, list):
            for t in trace:
                if t is not None:
                    t.record_llm_call(num_calls)
        else:
            trace.record_llm_call(num_calls)

    # --------------------------------------------------
    # Prompt formatting
    # --------------------------------------------------
    @staticmethod
    def format_passages(docs) -> str:
        blocks = []

        for doc in docs or []:
            title = str(getattr(doc, "title", "") or "").strip()
            text = str(getattr(doc, "text", "") or "").strip()

            if not text:
                continue

            if title:
                blocks.append(f"Wikipedia Title: {title}\n{text}")
            else:
                blocks.append(f"Wikipedia Title: \n{text}")

        return "\n\n".join(blocks)

    def format_no_context_direct_prompt(self, question: str) -> str:
        return f"Q: {question}\nA:"

    def format_gold_context_direct_prompt(self, question: str, context: str) -> str:
        return f"{context}\n\nQ: {question}\nA:"

    def format_cot_prompt(self, question: str, context: str) -> str:
        return (
            f"{context}\n\n"
            "Q: Answer the following question by reasoning step-by-step.\n"
            f"{question}\n"
            "A:"
        )

    def format_query_generation_prompt(self, question: str, max_queries: int = 2) -> str:
        return (
            "Generate short Wikipedia search queries for the following question.\n"
            "Return one query per line.\n\n"
            f"Q: {question}\n"
            f"A:"
        )

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        trace: Optional[Union[ExecutionTrace, List[ExecutionTrace]]] = None,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        self._record_llm_call(trace, num_calls=1)

        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        num_beams = self.default_num_beams if num_beams is None else num_beams
        do_sample = self.default_do_sample if do_sample is None else do_sample
        temperature = self.default_temperature if temperature is None else temperature

        all_outputs: List[str] = []

        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]

            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
            ).to(self.device)

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": do_sample,
            }

            if do_sample:
                generation_kwargs["temperature"] = temperature

            output_ids = self.model.generate(
                **encoded,
                **generation_kwargs,
            )

            decoded = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            all_outputs.extend(decoded)

        return all_outputs

    # --------------------------------------------------
    # QA wrappers
    # --------------------------------------------------
    def answer(
        self,
        questions: List[str],
        contexts: Optional[List[Optional[str]]] = None,
        strategy: str = "direct",
        trace: Optional[Union[ExecutionTrace, List[ExecutionTrace]]] = None,
        return_debug: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[dict]]]:
        if contexts is None:
            contexts = [None] * len(questions)

        if len(contexts) != len(questions):
            raise ValueError("questions and contexts must have the same length")

        prompts: List[str] = []

        for question, context in zip(questions, contexts):
            if context:
                if strategy == "cot":
                    prompts.append(self.format_cot_prompt(question, context))
                else:
                    prompts.append(self.format_gold_context_direct_prompt(question, context))
            else:
                prompts.append(self.format_no_context_direct_prompt(question))

        max_tokens = self.cot_max_new_tokens if strategy == "cot" else self.default_max_new_tokens

        raw_outputs = self.generate(
            prompts,
            max_new_tokens=max_tokens,
            trace=trace,
        )

        if strategy == "cot":
            cleaned_outputs = [extract_cot_answer(text) for text in raw_outputs]
        else:
            cleaned_outputs = [clean_answer(text) for text in raw_outputs]

        final_outputs = [text if text else "unknown" for text in cleaned_outputs]

        if return_debug:
            debug_rows = []

            for prompt, raw, cleaned, final in zip(
                prompts,
                raw_outputs,
                cleaned_outputs,
                final_outputs,
            ):
                debug_rows.append(
                    {
                        "prompt": prompt,
                        "raw_generation": raw,
                        "cleaned_generation": cleaned,
                        "final_generation": final,
                        "strategy_prompt_type": strategy,
                    }
                )

            return final_outputs, debug_rows

        return final_outputs

    def generate_search_queries(
        self,
        questions: List[str],
        trace: Optional[Union[ExecutionTrace, List[ExecutionTrace]]] = None,
        max_queries: int = 2,
        return_debug: bool = False,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[dict]]]:
        prompts = [
            self.format_query_generation_prompt(question, max_queries=max_queries)
            for question in questions
        ]

        raw_outputs = self.generate(
            prompts,
            max_new_tokens=self.query_max_new_tokens,
            trace=trace,
            num_beams=1,
            do_sample=False,
        )

        parsed = [
            parse_generated_queries(output, question, max_queries=max_queries)
            for output, question in zip(raw_outputs, questions)
        ]

        if return_debug:
            debug_rows = []

            for prompt, raw, queries in zip(prompts, raw_outputs, parsed):
                debug_rows.append(
                    {
                        "search_query_prompt": prompt,
                        "search_query_raw_generation": raw,
                        "search_queries": queries,
                    }
                )

            return parsed, debug_rows

        return parsed