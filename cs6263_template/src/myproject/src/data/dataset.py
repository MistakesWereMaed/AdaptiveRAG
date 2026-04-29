from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from torch.utils.data import Dataset

from .file_loader import load_records
from cs6263_template.src.myproject.src.data.schemas import QAItem


@dataclass
class QAExample:
	question: str
	answer: str
	label: Optional[int] = None
	metadata: Optional[Dict[str, Any]] = None


class QADataset(Dataset):
	def __init__(self, source: Union[str, Path, Sequence[Union[Dict[str, Any], QAItem]]]):
		if isinstance(source, (str, Path)):
			records = load_records(source)
		else:
			records = list(source)

		self.examples: List[QAExample] = []
		for record in records:
			if isinstance(record, QAItem):
				question = record.question
				answer = record.gold
				label = None
				metadata = None
			elif isinstance(record, dict):
				question = record.get("question") or record.get("query")
				answer = record.get("answer")
				label = record.get("label") if isinstance(record.get("label"), int) else None
				metadata = {key: value for key, value in record.items() if key not in {"question", "answer", "label"}}
			else:
				continue

			self.examples.append(QAExample(question=question, answer=answer, label=label, metadata=metadata))

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> QAExample:
		return self.examples[index]


def load_qa_dataset(source: Union[str, Path, Sequence[Dict[str, Any]]]) -> QADataset:
	return QADataset(source)

