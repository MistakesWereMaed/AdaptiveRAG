#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABEL2ID = {"A": 0, "B": 1, "C": 2}
ID2LABEL = {0: "A", 1: "B", 2: "C"}


class DebertaRouterModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str = "microsoft/deberta-v3-large",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        if class_weights is not None:
            self.register_buffer("class_weights_tensor", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights_tensor = None

        self.val_preds: List[int] = []
        self.val_targets: List[int] = []
        self.val_datasets: List[str] = []

    def forward(self, input_ids, attention_mask, labels=None):
        if self.class_weights_tensor is None or labels is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights_tensor.to(outputs.logits.device),
        )
        outputs.loss = loss
        return outputs

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train/loss", outputs.loss, prog_bar=True, on_step=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        preds = torch.argmax(outputs.logits, dim=-1)

        self.val_preds.extend(preds.detach().cpu().tolist())
        self.val_targets.extend(batch["labels"].detach().cpu().tolist())
        self.val_datasets.extend([str(x) for x in batch.get("dataset_name", [""] * len(preds))])

        self.log("val/loss", outputs.loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return outputs.loss

    def on_validation_epoch_end(self) -> None:
        preds = self.val_preds
        targets = self.val_targets
        datasets = self.val_datasets

        if not preds:
            return

        correct = [int(p == t) for p, t in zip(preds, targets)]
        acc = sum(correct) / len(correct)
        self.log("val/accuracy", acc, prog_bar=True, sync_dist=True)

        pred_counts = Counter(preds)
        target_counts = Counter(targets)

        macro_f1_values = []
        for idx, label in ID2LABEL.items():
            tp = sum(1 for p, t in zip(preds, targets) if p == idx and t == idx)
            fp = sum(1 for p, t in zip(preds, targets) if p == idx and t != idx)
            fn = sum(1 for p, t in zip(preds, targets) if p != idx and t == idx)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            macro_f1_values.append(f1)

            support = target_counts.get(idx, 0)
            class_acc = tp / support if support else 0.0

            self.log(f"val/acc_{label}", class_acc, sync_dist=True)
            self.log(f"val/precision_{label}", precision, sync_dist=True)
            self.log(f"val/recall_{label}", recall, sync_dist=True)
            self.log(f"val/f1_{label}", f1, sync_dist=True)
            self.log(f"val/pred_count_{label}", float(pred_counts.get(idx, 0)), sync_dist=True)
            self.log(f"val/target_count_{label}", float(support), sync_dist=True)

        macro_f1 = sum(macro_f1_values) / len(macro_f1_values)
        self.log("val/macro_f1", macro_f1, prog_bar=True, sync_dist=True)

        for dataset in sorted(set(d for d in datasets if d)):
            idxs = [i for i, d in enumerate(datasets) if d == dataset]
            if idxs:
                ds_acc = sum(correct[i] for i in idxs) / len(idxs)
                self.log(f"val_by_dataset/{dataset}_accuracy", ds_acc, sync_dist=True)

        self.val_preds.clear()
        self.val_targets.clear()
        self.val_datasets.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        grouped_params = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
