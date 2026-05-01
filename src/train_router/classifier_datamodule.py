from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from src.file_loader import load_raw_records


class RouterDataset(Dataset):
    """Question-only dataset for strategy-router training."""

    def __init__(self, records: Sequence[Any]):
        self.records: List[Dict[str, Any]] = []

        for record in records:
            if isinstance(record, dict):
                question = record.get("question")
                label = record.get("label")
            else:
                question = getattr(record, "question", None)
                label = getattr(record, "label", None)

            if isinstance(question, str) and label is not None:
                self.records.append({"question": question, "label": int(label)})

        if not self.records:
            raise ValueError("RouterDataset received no labeled records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


class RouterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: str | Path | Sequence[Any],
        model_name: str,
        val_data: str | Path | Sequence[Any] | None = None,
        batch_size: int = 16,
        max_length: int = 128,
        num_workers: int = 0,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.num_workers = int(num_workers)
        self.val_split = float(val_split)
        self.seed = int(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    @staticmethod
    def _load(source: str | Path | Sequence[Any]) -> List[Any]:
        if isinstance(source, (str, Path)):
            return list(load_raw_records(source))
        return list(source)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        full_train = RouterDataset(self._load(self.train_data))

        if self.val_data is not None:
            self.train_dataset = full_train
            self.val_dataset = RouterDataset(self._load(self.val_data))
            return

        if len(full_train) < 2:
            self.train_dataset = full_train
            self.val_dataset = full_train
            return

        val_size = max(1, int(round(len(full_train) * self.val_split)))
        train_size = len(full_train) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["question"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return encoded

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=torch.cuda.is_available(),
        )
