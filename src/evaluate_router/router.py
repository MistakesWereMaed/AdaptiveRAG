from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer

from src.train_router.classifier_lightning import RouterLightningModule


LABEL_TO_STRATEGY = {0: "no-rag", 1: "single", 2: "multi"}
STRATEGY_TO_LABEL = {v: k for k, v in LABEL_TO_STRATEGY.items()}


class RouterPredictor:
    """Inference wrapper for a trained RouterLightningModule checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str,
        device: str | None = None,
        max_length: int = 128,
        num_classes: int = 3,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = RouterLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            model_name=model_name,
            num_classes=num_classes,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, questions: Sequence[str]) -> List[Dict[str, Any]]:
        encoded = self.tokenizer(
            list(questions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
        )

        probs = torch.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=-1)

        rows = []
        for label, prob in zip(labels.cpu().tolist(), probs.cpu().tolist()):
            rows.append(
                {
                    "label": int(label),
                    "strategy": LABEL_TO_STRATEGY[int(label)],
                    "confidence": float(max(prob)),
                    "probabilities": {
                        LABEL_TO_STRATEGY[i]: float(p)
                        for i, p in enumerate(prob)
                    },
                }
            )

        return rows

    def predict(self, questions: Sequence[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for start in range(0, len(questions), batch_size):
            outputs.extend(self.predict_batch(questions[start : start + batch_size]))
        return outputs
