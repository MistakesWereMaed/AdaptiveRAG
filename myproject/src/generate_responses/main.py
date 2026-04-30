import argparse

from myproject.src.generate_responses.workflow import run_generate_responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream adaptive RAG predictions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--strategy", default=None, choices=["no-rag", "single", "multi", "all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_generate_responses(
        config_path=args.config,
        split=args.split,
        strategy_override=args.strategy,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
