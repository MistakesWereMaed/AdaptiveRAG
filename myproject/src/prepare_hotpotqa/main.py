import argparse

from src.prepare_hotpotqa.workflow import run_prepare_hotpotqa


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess HotpotQA")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    args = parser.parse_args()
    run_prepare_hotpotqa(args.config)


if __name__ == "__main__":
    main()