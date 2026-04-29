import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs6263_template.src.myproject.src.data.file_loader import extract_documents, load_raw_records, load_yaml_config
from cs6263_template.src.myproject.src.rag.retriever import FaissIVFRetriever


def main():
    print("[build_index] Starting index build", flush=True)
    parser = argparse.ArgumentParser(description="Build a retrieval index from a corpus")
    parser.add_argument("--config", default="config.yaml", help="Path to retriever config")
    args = parser.parse_args()

    paths = load_yaml_config(args.config, section="paths")
    config = load_yaml_config(args.config, section="retriever")
    corpus_path = str(paths["corpus"])
    output_dir = str(paths["index_dir"])
    encoder_name = str(config["encoder_name"])

    print(f"[build_index] Loading corpus from {corpus_path}", flush=True)
    records = load_raw_records(corpus_path)
    documents = extract_documents(records)
    print(f"[build_index] Building index for {len(documents)} documents", flush=True)

    index = FaissIVFRetriever(encoder_name=encoder_name)
    index.build(documents)
    index.save(output_dir)
    print(f"[build_index] Finished writing index metadata to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
