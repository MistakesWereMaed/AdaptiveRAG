from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class RouterClassifier(nn.Module):
    """Encoder + linear classification head for no-rag/single/multi routing."""

    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden_size, int(num_classes))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]

        return self.classifier(self.dropout(pooled))

    @torch.no_grad()
    def predict(self, input_ids, attention_mask=None):
        self.eval()
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return torch.argmax(logits, dim=-1)
