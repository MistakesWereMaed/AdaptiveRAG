import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

from tqdm.auto import tqdm


def normalize_answer(text: str) -> str:
	print("[preprocessing] Normalizing answer", flush=True)
	text = text.lower()
	text = re.sub(f"[{string.punctuation}]", "", text)
	return " ".join(text.split())


def load_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
	print(f"[preprocessing] Loading records from {path}", flush=True)
	source_path = Path(path)
	if not source_path.exists():
		raise FileNotFoundError(f"Input file not found: {source_path}")

	suffix = source_path.suffix.lower()
	if suffix == ".jsonl":
		with source_path.open("r", encoding="utf-8") as handle:
			return [json.loads(line) for line in handle if line.strip()]

	with source_path.open("r", encoding="utf-8") as handle:
		payload = json.load(handle)

	if isinstance(payload, list):
		return payload

	if isinstance(payload, dict):
		for key in ("data", "records", "examples"):
			if key in payload and isinstance(payload[key], list):
				return payload[key]

	raise ValueError(f"Unsupported JSON structure in {source_path}")


def extract_text(record: Dict[str, Any]) -> str:
	context_value = record.get("context")
	if isinstance(context_value, str) and context_value.strip():
		return context_value.strip()

	if isinstance(context_value, list):
		flattened_segments = []
		for entry in context_value:
			if isinstance(entry, (list, tuple)) and len(entry) == 2:
				title, sentences = entry
				if isinstance(title, str):
					if isinstance(sentences, list):
						joined_sentences = " ".join(sentence for sentence in sentences if isinstance(sentence, str))
						segment = f"{title.strip()}: {joined_sentences.strip()}".strip()
					elif isinstance(sentences, str):
						segment = f"{title.strip()}: {sentences.strip()}".strip()
					else:
						segment = title.strip()
					if segment:
						flattened_segments.append(segment)
				continue
			if isinstance(entry, str) and entry.strip():
				flattened_segments.append(entry.strip())
		if flattened_segments:
			return "\n".join(flattened_segments)

	for key in ("text", "content", "passage", "document", "context"):
		value = record.get(key)
		if isinstance(value, str) and value.strip():
			return value.strip()
	return ""


def extract_qa_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
	print("[preprocessing] Extracting QA records", flush=True)
	examples: List[Dict[str, Any]] = []
	for record in tqdm(records, desc="Extracting QA records", unit="record"):
		question = record.get("question") or record.get("query")
		answer = record.get("answer") or record.get("answers")

		if isinstance(answer, list):
			answer = answer[0] if answer else ""

		if not isinstance(question, str) or not isinstance(answer, str):
			continue

		examples.append({
			"question": question.strip(),
			"answer": answer.strip(),
			**{key: value for key, value in record.items() if key not in {"question", "query", "answer", "answers"}},
		})

	return examples


def extract_documents(records: Iterable[Dict[str, Any]]) -> List[str]:
	print("[preprocessing] Extracting documents", flush=True)
	documents = []
	for record in tqdm(records, desc="Extracting documents", unit="record"):
		text = extract_text(record)
		if text:
			documents.append(text)
	return documents

