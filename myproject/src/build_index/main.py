import argparse

from src.stages.build_index.helpers.workflow import run_build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a retrieval index from a corpus")
    parser.add_argument("--config", default="config.yaml", help="Path to retriever config")
    args = parser.parse_args()
    run_build_index(args.config)


if __name__ == "__main__":
    main()
