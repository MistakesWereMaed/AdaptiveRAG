#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

LABEL2ID = {"A": 0, "B": 1, "C": 2}
ID2LABEL = {0: "A", 1: "B", 2: "C"}


class AdaptiveRouterClassificationDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        text_template: str = "{question}",
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_template = text_template
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
                    raise ValueError(f"Expected list JSON in {path}")
                records = payload

        cleaned = []
        skipped = 0
        for row in records:
            question = str(row.get("question", "")).strip()
            label = str(row.get("answer", "")).strip()
            if not question or label not in LABEL2ID:
                skipped += 1
                continue
            cleaned.append(row)

        if not cleaned:
            raise ValueError(f"No valid A/B/C examples found in {path}")
        if skipped:
            print(f"[Dataset] Skipped {skipped} invalid rows from {path}")
        return cleaned

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        question = " ".join(str(row["question"]).split())
        label_text = str(row["answer"]).strip()
        text = self.text_template.format(question=question)

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "id": str(row.get("id", idx)),
            "dataset_name": str(row.get("dataset_name", "")),
            "question": question,
            "label_text": label_text,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(LABEL2ID[label_text], dtype=torch.long),
        }


class AdaptiveRouterClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        validation_file: str,
        model_name_or_path: str = "microsoft/deberta-v3-large",
        batch_size: int = 4,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 4,
        max_length: int = 256,
        text_template: str = "{question}",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=max_length,
            use_fast=True,
        )
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = AdaptiveRouterClassificationDataset(
                self.hparams.train_file,
                self.tokenizer,
                self.hparams.max_length,
                self.hparams.text_template,
            )
            self.val_dataset = AdaptiveRouterClassificationDataset(
                self.hparams.validation_file,
                self.tokenizer,
                self.hparams.max_length,
                self.hparams.text_template,
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
