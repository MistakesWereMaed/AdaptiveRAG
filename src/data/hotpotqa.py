from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
from tqdm.auto import tqdm

from datasets import load_dataset


def load_hotpotqa_split(split: str = "train", config_name: str = "distractor"):
	return load_dataset("hotpot_qa", config_name, split=split)


def hotpotqa_context_to_documents(context: Any) -> List[str]:
	documents: List[str] = []

	if isinstance(context, str):
		text = context.strip()
		if text:
			for segment in text.split("\n"):
				segment = segment.strip()
				if segment:
					documents.append(segment)
		return documents

	if isinstance(context, dict):
		for key in ("sentences", "paragraphs", "documents", "passages"):
			value = context.get(key)
			if isinstance(value, list):
				for item in value:
					documents.extend(hotpotqa_context_to_documents(item))
		return documents

	if isinstance(context, list):
		for entry in context:
			if isinstance(entry, (list, tuple)) and len(entry) == 2:
				title, sentences = entry
				title_text = title.strip() if isinstance(title, str) else ""
				if isinstance(sentences, list):
					for sentence in sentences:
						if isinstance(sentence, str) and sentence.strip():
							if title_text:
								documents.append(f"{title_text}: {sentence.strip()}")
							else:
								documents.append(sentence.strip())
				elif isinstance(sentences, str) and sentences.strip():
					if title_text:
						documents.append(f"{title_text}: {sentences.strip()}")
					else:
						documents.append(sentences.strip())
				elif title_text:
					documents.append(title_text)
			elif isinstance(entry, str) and entry.strip():
				documents.append(entry.strip())

	return documents


def hotpotqa_record_to_example(record: Dict[str, Any]) -> Dict[str, Any]:
	context_documents = hotpotqa_context_to_documents(record.get("context"))
	return {
		"id": record.get("id"),
		"question": record.get("question", ""),
		"answer": record.get("answer", ""),
		"context": "\n".join(context_documents),
		"context_documents": context_documents,
		"supporting_facts": record.get("supporting_facts", []),
		"type": record.get("type"),
		"level": record.get("level"),
	}


def hotpotqa_dataset_to_records(dataset: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
	return [hotpotqa_record_to_example(record) for record in tqdm(dataset, desc="Converting HotpotQA", unit="example")]


def save_jsonl(records: Iterable[Dict[str, Any]], output_path: str | Path) -> None:
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	with output_file.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")
