from __future__ import annotations

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.train_router.classifier_model import RouterClassifier


class RouterLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 3,
        learning_rate: float = 2e-5,
        dropout: float = 0.1,
        weight_decay: float = 0.01,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.model = RouterClassifier(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _step(self, batch: Dict[str, torch.Tensor], prefix: str):
        logits = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

        loss = F.cross_entropy(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == batch["labels"]).float().mean()

        self.log(f"{prefix}_loss", loss, prog_bar=(prefix == "val"), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
