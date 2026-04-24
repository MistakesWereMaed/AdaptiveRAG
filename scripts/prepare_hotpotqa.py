import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from src.data.hotpotqa import hotpotqa_context_to_documents, hotpotqa_dataset_to_records, load_hotpotqa_split


def _write_jsonl(records, output_path: Path):
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		for record in tqdm(records, desc=f"Writing {output_path.name}", unit="record"):
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_corpus(records):
	corpus = []
	seen_texts = set()
	for record in records:
		context_documents = record.get("context_documents")
		if not isinstance(context_documents, list):
			context_documents = hotpotqa_context_to_documents(record.get("context"))

		for passage in context_documents:
			passage = passage.strip()
			if passage and passage not in seen_texts:
				seen_texts.add(passage)
				corpus.append({"text": passage})
	return corpus


def main():
	print("[prepare_hotpotqa] Starting HotpotQA preprocessing", flush=True)
	parser = argparse.ArgumentParser(description="Download and preprocess HotpotQA")
	parser.add_argument("--output-dir", default="data/hotpotqa", help="Directory for processed HotpotQA files")
	parser.add_argument("--config-name", default="distractor", choices=["distractor", "fullwiki"], help="HotpotQA configuration to load")
	parser.add_argument("--build-corpus", action="store_true", help="Also build a lightweight corpus from contexts")
	args = parser.parse_args()

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	for split in tqdm(("train", "validation"), desc="HotpotQA splits", unit="split"):
		print(f"[prepare_hotpotqa] Loading split={split}", flush=True)
		dataset = load_hotpotqa_split(split=split, config_name=args.config_name)
		records = hotpotqa_dataset_to_records(dataset)
		_write_jsonl(records, output_dir / f"{split}.jsonl")

		if args.build_corpus and split == "train":
			print("[prepare_hotpotqa] Building corpus from train split", flush=True)
			corpus = _build_corpus(records)
			if not corpus:
				raise ValueError("HotpotQA corpus extraction produced no passages; check the dataset schema or config name")
			_write_jsonl(corpus, output_dir / "corpus.jsonl")


if __name__ == "__main__":
	main()
