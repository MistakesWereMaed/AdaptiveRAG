#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

VALID_LABELS = {"A", "B", "C"}


class AdaptiveRouterDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: AutoTokenizer,
        max_input_length: int = 256,
        max_target_length: int = 4,
        prompt_template: str = "Question: {question} Complexity:",
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prompt_template = prompt_template
        self.records = self._load_records(self.path)

    @staticmethod
    def _load_records(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in f if line.strip()]
            else:
                payload = json.load(f)
                if not isinstance(payload, list):
                    raise ValueError(f"Expected a list of examples in {path}")
                records = payload
        cleaned = []
        skipped = 0
        for row in records:
            question = str(row.get("question", "")).strip()
            label = str(row.get("answer", "")).strip()
            if not question or label not in VALID_LABELS:
                skipped += 1
                continue
            cleaned.append(row)
        if not cleaned:
            raise ValueError(f"No valid A/B/C examples found in {path}")
        if skipped:
            print(f"[AdaptiveRouterDataset] Skipped {skipped} invalid rows from {path}")
        return cleaned

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        question = " ".join(str(row["question"]).split())
        label = str(row["answer"]).strip()
        source_text = self.prompt_template.format(question=question)
        target_text = label
        source = self.tokenizer(
            source_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "id": str(row.get("id", idx)),
            "dataset_name": str(row.get("dataset_name", "")),
            "question": question,
            "target_text": target_text,
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }


class AdaptiveRouterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        validation_file: str,
        model_name_or_path: str = "t5-large",
        batch_size: int = 4,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 2,
        max_input_length: int = 256,
        max_target_length: int = 4,
        prompt_template: str = "Question: {question} Complexity:",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=256,
        )
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = AdaptiveRouterDataset(
                self.hparams.train_file,
                self.tokenizer,
                self.hparams.max_input_length,
                self.hparams.max_target_length,
                self.hparams.prompt_template,
            )
            self.val_dataset = AdaptiveRouterDataset(
                self.hparams.validation_file,
                self.tokenizer,
                self.hparams.max_input_length,
                self.hparams.max_target_length,
                self.hparams.prompt_template,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batch_size or self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
