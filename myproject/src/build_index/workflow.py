from myproject.src.file_loader import extract_structured_documents, load_raw_records, load_yaml_config
from myproject.src.build_index.retriever import FaissIVFRetriever


def run_build_index(config_path: str = "config.yaml") -> None:
    print("[build_index] Starting index build", flush=True)

    paths = load_yaml_config(config_path, section="paths")
    config = load_yaml_config(config_path, section="retriever")
    corpus_path = str(paths["corpus"])
    output_dir = str(paths["index_dir"])
    encoder_name = str(config["encoder_name"])

    print(f"[build_index] Loading corpus from {corpus_path}", flush=True)
    records = load_raw_records(corpus_path)
    documents = extract_structured_documents(records)
    print(f"[build_index] Building index for {len(documents)} documents", flush=True)

    index = FaissIVFRetriever(encoder_name=encoder_name)
    index.build(documents)
    index.save(output_dir)
    print(f"[build_index] Finished writing index metadata to {output_dir}", flush=True)
