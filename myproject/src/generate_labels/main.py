import argparse

from src.stages.generate_labels.helpers.workflow import run_generate_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate router labels")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_generate_labels(args.config)


if __name__ == "__main__":
    main()
