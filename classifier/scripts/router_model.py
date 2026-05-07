#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["A", "B", "C"]


class T5RouterModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "t5-large",
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        max_gen_length: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=256,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.validation_predictions: List[str] = []
        self.validation_targets: List[str] = []
        self.validation_datasets: List[str] = []

    @staticmethod
    def clean_label(text: str) -> str:
        text = str(text).strip().upper()
        return text[0] if text and text[0] in LABELS else text

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def decode_labels(self, labels: torch.Tensor) -> List[str]:
        labels = labels.detach().clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        decoded = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return [self.clean_label(x) for x in decoded]

    def generate_predictions(self, batch: Dict[str, Any]) -> List[str]:
        generated = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.hparams.max_gen_length,
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [self.clean_label(x) for x in decoded]

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_step=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        preds = self.generate_predictions(batch)
        targets = self.decode_labels(batch["labels"])
        self.validation_predictions.extend(preds)
        self.validation_targets.extend(targets)
        self.validation_datasets.extend([str(x) for x in batch.get("dataset_name", [""] * len(preds))])
        self.log("val/loss", outputs.loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return outputs.loss

    def on_validation_epoch_end(self) -> None:
        preds = self.validation_predictions
        targets = self.validation_targets
        datasets = self.validation_datasets
        if not preds:
            return
        correct = [p == t for p, t in zip(preds, targets)]
        acc = sum(correct) / len(correct)
        self.log("val/accuracy", acc, prog_bar=True, sync_dist=True)
        pred_counts = Counter(preds)
        target_counts = Counter(targets)
        for label in LABELS:
            label_total = sum(1 for t in targets if t == label)
            label_correct = sum(1 for p, t in zip(preds, targets) if t == label and p == t)
            if label_total:
                self.log(f"val/acc_{label}", label_correct / label_total, sync_dist=True)
            self.log(f"val/pred_count_{label}", float(pred_counts.get(label, 0)), sync_dist=True)
            self.log(f"val/target_count_{label}", float(target_counts.get(label, 0)), sync_dist=True)
        for dataset in sorted(set(d for d in datasets if d)):
            idxs = [i for i, d in enumerate(datasets) if d == dataset]
            if idxs:
                ds_acc = sum(correct[i] for i in idxs) / len(idxs)
                self.log(f"val_by_dataset/{dataset}_accuracy", ds_acc, sync_dist=True)
        self.validation_predictions.clear()
        self.validation_targets.clear()
        self.validation_datasets.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {
                "params": [p for n, p in self.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(grouped_params, lr=self.hparams.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        if total_steps <= 0:
            return optimizer
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.hparams.warmup_ratio),
            num_training_steps=total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
