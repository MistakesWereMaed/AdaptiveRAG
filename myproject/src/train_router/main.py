import argparse

from src.train_router.workflow import run_train_router


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the router classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to training config")
    args = parser.parse_args()
    run_train_router(args.config)


if __name__ == "__main__":
    main()
