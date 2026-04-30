from typing import Any, Dict, List, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from myproject.src.file_loader import load_records
from myproject.src.schemas import QAItem


class RouterDataset(Dataset):
	def __init__(self, records: Sequence[Union[Dict[str, object], QAItem]]):
		print(f"[RouterDataset] Building dataset from {len(records)} records", flush=True)
		cleaned = []
		for record in records:
			if isinstance(record, QAItem):
				if record is not None:
					continue
			elif isinstance(record, dict):
				if isinstance(record.get("question"), str) and record.get("label") is not None:
					cleaned.append(record)
		self.records = cleaned

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, index: int) -> Dict[str, object]:
		record = self.records[index]
		return {
			"question": str(record["question"]),
			"label": int(record["label"]),
		}


class RouterDataModule(pl.LightningDataModule):
	def __init__(self, train_data, model_name: str, val_data=None, batch_size: int = 8, max_length: int = 256, num_workers: int = 0):
		super().__init__()
		print(f"[RouterDataModule] Initializing model={model_name} batch_size={batch_size}", flush=True)
		self.train_data = train_data
		self.val_data = val_data
		self.model_name = model_name
		self.batch_size = batch_size
		self.max_length = max_length
		self.num_workers = num_workers
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.train_dataset = None
		self.val_dataset = None

	def _load_records(self, source):
		print(f"[RouterDataModule] Loading records from {source}", flush=True)
		if isinstance(source, (str, Path)):
			records = load_records(source)
		else:
			records = list(source)
		return records

	def setup(self, stage: Optional[str] = None):
		print(f"[RouterDataModule] Setup stage={stage}", flush=True)
		if self.train_dataset is None:
			train_records = self._load_records(self.train_data)
			dataset = RouterDataset(train_records)

			if len(dataset) == 0:
				raise ValueError("Training data does not contain any labeled examples")

			if self.val_data is None:
				if len(dataset) == 1:
					self.train_dataset = dataset
					self.val_dataset = dataset
				else:
					train_size = max(1, int(0.9 * len(dataset)))
					val_size = len(dataset) - train_size
					if val_size == 0:
						val_size = 1
						train_size = len(dataset) - val_size
					self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
			else:
				self.train_dataset = dataset
				self.val_dataset = RouterDataset(self._load_records(self.val_data))

	def _collate(self, batch: List[Dict[str, object]]):
		print(f"[RouterDataModule] Collating batch of {len(batch)} examples", flush=True)
		questions = [item["question"] for item in batch]
		labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
		encoded = self.tokenizer(
			questions,
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		encoded["labels"] = labels
		return encoded

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			collate_fn=self._collate,
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			collate_fn=self._collate,
		)
