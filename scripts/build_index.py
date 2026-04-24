import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import extract_documents, load_records
from src.retrieval.index import FaissRetrieverIndex
from src.utils.config import load_yaml_config


def main():
	parser = argparse.ArgumentParser(description="Build a FAISS retrieval index from a corpus")
	parser.add_argument("--config", default="configs/retriever.yaml", help="Path to retriever config")
	parser.add_argument("--corpus", default=None, help="Path to a JSON or JSONL corpus")
	parser.add_argument("--output-dir", default=None, help="Directory for the FAISS index")
	parser.add_argument("--encoder", default=None, help="SentenceTransformer encoder name")
	args = parser.parse_args()

	config = load_yaml_config(args.config)
	corpus_path = args.corpus or config.get("corpus")
	output_dir = args.output_dir or config.get("output_dir", "outputs/index")
	encoder_name = args.encoder or config.get("encoder_name", "all-MiniLM-L6-v2")

	if corpus_path is None:
		raise ValueError("A corpus path must be provided via --corpus or the config file")

	records = load_records(corpus_path)
	documents = extract_documents(records)

	index = FaissRetrieverIndex(encoder_name=encoder_name)
	index.build(documents)
	index.save(output_dir)


if __name__ == "__main__":
	main()

