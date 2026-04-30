import torch
import torch.nn as nn
from transformers import AutoModel


class RouterClassifier(nn.Module):
	def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.1):
		super().__init__()
		print(f"[RouterClassifier] Loading encoder={model_name} num_classes={num_classes}", flush=True)
		self.encoder = AutoModel.from_pretrained(model_name)
		self.dropout = nn.Dropout(dropout)
		self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

	def forward(self, input_ids, attention_mask=None):
		print("[RouterClassifier] Forward pass", flush=True)
		outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
		pooled_output = getattr(outputs, "pooler_output", None)
		if pooled_output is None:
			pooled_output = outputs.last_hidden_state[:, 0]
		logits = self.classifier(self.dropout(pooled_output))
		return logits

	def predict(self, input_ids, attention_mask=None):
		print("[RouterClassifier] Predicting", flush=True)
		self.eval()
		with torch.no_grad():
			logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
			return torch.argmax(logits, dim=-1)
