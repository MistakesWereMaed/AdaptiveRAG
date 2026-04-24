from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from torch.utils.data import Dataset

from .preprocessing import extract_qa_records, load_records


@dataclass
class QAExample:
	question: str
	answer: str
	label: Optional[int] = None
	metadata: Optional[Dict[str, Any]] = None


class QADataset(Dataset):
	def __init__(self, source: Union[str, Path, Sequence[Dict[str, Any]]]):
		if isinstance(source, (str, Path)):
			records = load_records(source)
		else:
			records = list(source)

		self.examples: List[QAExample] = []
		for record in extract_qa_records(records):
			label = record.get("label")
			self.examples.append(
				QAExample(
					question=record["question"],
					answer=record["answer"],
					label=label if isinstance(label, int) else None,
					metadata={key: value for key, value in record.items() if key not in {"question", "answer", "label"}},
				)
			)

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> QAExample:
		return self.examples[index]


def load_qa_dataset(source: Union[str, Path, Sequence[Dict[str, Any]]]) -> QADataset:
	return QADataset(source)

