import os
import json
import hashlib
from pathlib import Path
from typing import List, Union, Optional

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from src.utils.config import load_yaml_config


class LocalLLM:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        max_input_length: Optional[int] = None,
        cache_path: Optional[str] = None,
        config_path: Union[str, Path] = "configs/llm.yaml",
        compile_model: Optional[bool] = None,
    ):
        config = load_yaml_config(config_path)

        if model_name is None:
            model_name = str(config.get("model_name", "mistralai/Mistral-7B-v0.1"))
        if device is None:
            device = str(config.get("device", "cuda"))
        if load_in_4bit is None:
            load_in_4bit = bool(config.get("load_in_4bit", True))
        if max_input_length is None:
            max_input_length = int(config.get("max_input_length", 4096))
        if compile_model is None:
            compile_model = bool(config.get("compile_model", True))

        self.model_name = model_name
        self.device = device
        self.max_input_length = max_input_length
        self.default_max_new_tokens = int(config.get("max_new_tokens", 32))
        self.default_batch_size = int(config.get("batch_size", 128))
        self.default_use_cache = bool(config.get("use_cache", True))

        bnb_4bit_compute_dtype = str(config.get("bnb_4bit_compute_dtype", "float16"))
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)
        quant_type = str(config.get("bnb_4bit_quant_type", "nf4"))
        use_double_quant = bool(config.get("bnb_4bit_use_double_quant", True))
        device_map = config.get("device_map", "auto")

        # ---- Quantization config ----
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_type=quant_type,
            )

        # ---- Tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Model ----
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            dtype=torch.float16 if not load_in_4bit else None,
            device_map=device_map,
        )

        self.model.eval()
        if compile_model:
            self.model = torch.compile(self.model)

        # ---- Cache ----
        self.cache_path = cache_path
        self.cache = {}

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    # --------------------------------------------------
    # Prompt Formatting
    # --------------------------------------------------
    def format_prompt(self, question: str, context: Optional[str] = None) -> str:
        if context:
            return (
                "[INST]\n"
                "Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n"
                "[/INST]"
            )
        else:
            return (
                "[INST]\n"
                "Answer the question.\n\n"
                f"Question:\n{question}\n"
                "[/INST]"
            )

    # --------------------------------------------------
    # Cache utilities
    # --------------------------------------------------
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cached(self, prompt: str):
        key = self._hash(prompt)
        return self.cache.get(key, None)

    def _set_cache(self, prompt: str, output: str):
        key = self._hash(prompt)
        self.cache[key] = output

    def save_cache(self):
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)

    # --------------------------------------------------
    # Core generation
    # --------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
    ) -> List[str]:

        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        if batch_size is None:
            batch_size = self.default_batch_size
        if use_cache is None:
            use_cache = self.default_use_cache

        if isinstance(prompts, str):
            prompts = [prompts]

        results = []

        # ---- batching ----
        prompts = sorted(prompts, key=len)
        batch_starts = range(0, len(prompts), batch_size)
        for i in tqdm(batch_starts, desc="Generating", unit="batch"):
            batch_prompts = prompts[i:i + batch_size]

            uncached_prompts = []
            uncached_indices = []
            batch_outputs = [None] * len(batch_prompts)

            # ---- check cache ----
            for j, p in enumerate(batch_prompts):
                if use_cache:
                    cached = self._get_cached(p)
                    if cached is not None:
                        batch_outputs[j] = cached
                        continue

                uncached_prompts.append(p)
                uncached_indices.append(j)

            # ---- run model on uncached ----
            if uncached_prompts:
                inputs = self.tokenizer(
                    uncached_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length
                ).to(self.model.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                decoded = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )

                # ---- store outputs ----
                for idx, text in zip(uncached_indices, decoded):
                    batch_outputs[idx] = text
                    if use_cache:
                        self._set_cache(batch_prompts[idx], text)

            results.extend(batch_outputs)

        return results

    # --------------------------------------------------
    # Convenience QA API
    # --------------------------------------------------
    def answer(
        self,
        questions: Union[str, List[str]],
        contexts: Optional[Union[str, List[str]]] = None,
        **gen_kwargs
    ) -> List[str]:

        if isinstance(questions, str):
            questions = [questions]

        if contexts is None:
            contexts = [None] * len(questions)
        elif isinstance(contexts, str):
            contexts = [contexts]

        prompts = [
            self.format_prompt(q, c)
            for q, c in zip(questions, contexts)
        ]

        return self.generate(prompts, **gen_kwargs)

    # --------------------------------------------------
    # Debug / monitoring
    # --------------------------------------------------
    def gpu_memory(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0