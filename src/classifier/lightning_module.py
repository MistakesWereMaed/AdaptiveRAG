from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .model import RouterClassifier


class RouterLightningModule(pl.LightningModule):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 3, learning_rate: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = RouterClassifier(model_name=model_name, num_classes=num_classes)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _step(self, batch: Dict[str, torch.Tensor], prefix: str):
        logits = self.forward(batch["input_ids"], batch.get("attention_mask"))
        loss = F.cross_entropy(logits, batch["labels"])
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == batch["labels"]).float().mean()
        self.log(f"{prefix}_loss", loss, prog_bar=(prefix != "train"), on_step=(prefix == "train"), on_epoch=True)
        self.log(f"{prefix}_accuracy", accuracy, prog_bar=(prefix != "train"), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)