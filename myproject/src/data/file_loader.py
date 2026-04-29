import json
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from tqdm.auto import tqdm

from src.data.schemas import QAItem, PredictionItem, RetrievedDocument, StrategyPredictions

STRATEGIES = ("no-rag", "single", "multi")


def _read_json_or_jsonl(path: Path) -> List[Any]:
	if not path.exists():
		raise FileNotFoundError(f"Missing file: {path}")

	suffix = path.suffix.lower()
	if suffix == ".jsonl":
		with path.open("r", encoding="utf-8") as f:
			return [json.loads(line) for line in f if line.strip()]

	with path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	if isinstance(payload, list):
		return payload

	if isinstance(payload, dict):
		for key in ("data", "records", "examples"):
			if key in payload and isinstance(payload[key], list):
				return payload[key]

	raise ValueError(f"Unsupported JSON structure in {path}")


def load_raw_records(path: Union[str, Path]) -> List[Any]:
	"""Return raw JSON records without Pydantic validation. Use at file boundary only."""
	return _read_json_or_jsonl(Path(path))


def load_records(path: Union[str, Path]) -> List[QAItem]:
	"""Load dataset and validate into List[QAItem].

	- Enforces presence of `question` and `answer` (coerced to `gold`)
	- Coerces/assigns `id` when safe
	- Extracts metadata including supporting_titles
	"""
	print(f"[preprocessing] Loading dataset from {path}", flush=True)
	source_path = Path(path)
	raw = _read_json_or_jsonl(source_path)

	examples: List[QAItem] = []
	for i, record in enumerate(raw):
		# accept dict-like records only at file boundary, then validate
		if isinstance(record, dict):
			q = record.get("question") or record.get("query") or record.get("q")
			a = record.get("answer") or record.get("answers") or record.get("gold")
			# handle list answers
			if isinstance(a, list):
				a = a[0] if a else None

			if not isinstance(q, str) or not isinstance(a, str):
				# skip malformed examples
				continue

			raw_id = record.get("id")
			if raw_id is None:
				raw_id = record.get("_id")
			try:
				item_id = int(raw_id) if raw_id is not None else i
			except Exception:
				item_id = i

			# Extract metadata, particularly supporting_titles
			metadata = {}
			supporting_titles = record.get("supporting_titles")
			if supporting_titles:
				metadata["supporting_titles"] = supporting_titles
			
			examples.append(QAItem(id=item_id, question=q, gold=a, metadata=metadata))

		elif isinstance(record, str) and record.strip():
			examples.append(QAItem(id=i, question=record.strip(), gold=""))

	if not examples:
		raise ValueError(f"No valid QA records found in {source_path}")

	return examples


def _parse_prediction_record(record: Any, strategy: str) -> PredictionItem:
	if isinstance(record, str):
		raise ValueError("Prediction entries must include an `id` when provided as strings are not allowed")

	if not isinstance(record, dict):
		raise ValueError("Unsupported prediction record format")

	raw_id = record.get("id")
	if raw_id is None:
		raw_id = record.get("_id")
	if raw_id is None:
		raise ValueError("Prediction record missing required `id` field")

	try:
		pid = int(raw_id)
	except Exception:
		raise ValueError("Prediction `id` must be coercible to int")

	if "prediction" in record:
		pred_text = record.get("prediction")
	elif "prediction_text" in record:
		pred_text = record.get("prediction_text")
	elif "answer" in record:
		pred_text = record.get("answer")
	elif "text" in record:
		pred_text = record.get("text")
	else:
		raise ValueError(f"Prediction text not found for id={pid}")

	gold = record.get("gold") or record.get("answer")

	return PredictionItem(id=pid, prediction=str(pred_text), gold=gold, strategy=strategy)


def load_single_predictions(path: Union[str, Path], strategy: str = "no-rag") -> List[PredictionItem]:
	path = Path(path)
	raw = _read_json_or_jsonl(path)
	preds: List[PredictionItem] = []
	for r in raw:
		preds.append(_parse_prediction_record(r, strategy))
	return preds


def load_predictions(base_path: Union[str, Path]) -> StrategyPredictions:
	"""Load strategy split predictions and validate into StrategyPredictions.

	Enforces top-level presence of `no-rag`, `single`, `multi` files.
	Each entry must include an `id` field and prediction text.
	"""
	base_path = Path(base_path)

	# directory case
	if base_path.is_dir():
		paths = {s: base_path / f"{s}.json" for s in STRATEGIES}
	else:
		paths = {s: base_path.with_name(f"{base_path.stem}-{s}{base_path.suffix}") for s in STRATEGIES}

	outputs = {}
	for s, p in paths.items():
		print(f"[preprocessing] Loading predictions {s} from {p}", flush=True)
		outputs[s] = [
			_parse_prediction_record(r, s)
			for r in _read_json_or_jsonl(p)
		]

	# validate equal lengths
	lengths = {k: len(v) for k, v in outputs.items()}
	if len(set(lengths.values())) != 1:
		raise ValueError(f"Prediction length mismatch: {lengths}")

	# construct StrategyPredictions using aliases
	data = {
		"no-rag": outputs["no-rag"],
		"single": outputs["single"],
		"multi": outputs["multi"],
	}

	return StrategyPredictions.model_validate(data)


def extract_documents(records: Iterable[Any]) -> List[str]:
	print("[preprocessing] Extracting documents", flush=True)
	documents = []
	for record in tqdm(records, desc="Extracting documents", unit="record"):
		# best-effort extraction from dict-like boundary objects
		if isinstance(record, dict):
			for key in ("context", "text", "content", "passage", "document"):
				v = record.get(key)
				if isinstance(v, str) and v.strip():
					documents.append(v.strip())
					break
		elif isinstance(record, str) and record.strip():
			documents.append(record.strip())
	return documents


def extract_structured_documents(records: Iterable[Any]) -> List[RetrievedDocument]:
	print("[preprocessing] Extracting structured documents", flush=True)
	documents: List[RetrievedDocument] = []
	for index, record in enumerate(tqdm(records, desc="Extracting documents", unit="record")):
		if isinstance(record, RetrievedDocument):
			documents.append(record)
			continue

		if isinstance(record, str):
			text = record.strip()
			if text:
				documents.append(
					RetrievedDocument(
						doc_id=f"doc_{index}",
						title="",
						text=text,
						source="corpus",
					)
				)
			continue

		if isinstance(record, dict):
			raw_text = None
			for key in ("text", "content", "passage", "document", "context"):
				value = record.get(key)
				if isinstance(value, str) and value.strip():
					raw_text = value.strip()
					break

			if raw_text is None:
				continue

			raw_title = record.get("title") or record.get("name") or record.get("doc_title") or ""
			title = str(raw_title).strip() if raw_title is not None else ""
			raw_doc_id = record.get("doc_id") or record.get("id") or record.get("_id") or f"doc_{index}"
			doc_id = str(raw_doc_id).strip() if raw_doc_id is not None else f"doc_{index}"

			metadata = dict(record.get("metadata") or {})
			for key, value in record.items():
				if key not in {"doc_id", "id", "_id", "title", "name", "doc_title", "text", "content", "passage", "document", "context", "metadata", "score", "rank", "source"}:
					metadata.setdefault(key, value)

			documents.append(
				RetrievedDocument(
					doc_id=doc_id,
					title=title,
					text=raw_text,
					source=str(record.get("source") or "corpus"),
					metadata=metadata,
				)
			)

	return documents


def normalize_text(text: str) -> str:
	if text is None:
		return ""

	text = text.lower().strip()
	text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
	text = re.sub(r"\s+", " ", text)
	return text


def load_yaml_config(path: Optional[str | Path], section: Optional[str] = None) -> Dict[str, Any]:
	"""Load a YAML config file from disk."""
	if path is None:
		return {}

	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	import yaml

	with config_path.open("r", encoding="utf-8") as handle:
		payload = yaml.safe_load(handle)

	payload = payload or {}

	if section is not None:
		section_payload = payload.get(section)
		if isinstance(section_payload, dict):
			return section_payload

	return payload