import argparse
import sys
import os

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.classifier.datamodule import RouterDataModule
from src.classifier.lightning_module import RouterLightningModule
from src.utils.config import load_yaml_config
from src.utils.distributed import get_rank, get_world_size, is_distributed, is_main_process


def main():
    print("[train_classifier] Starting trainer setup", flush=True)
    parser = argparse.ArgumentParser(description="Train the router classifier")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--train-data", default=None, help="Path to labeled training data")
    parser.add_argument("--val-data", default=None, help="Optional validation data")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--strategy", default=None, help="Lightning strategy (e.g. ddp)")
    parser.add_argument("--devices", default=None, help="Lightning devices value")
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    train_data = args.train_data or config.get("train_data")
    model_name = args.model_name or config.get("model_name", "bert-base-uncased")
    batch_size = args.batch_size or int(config.get("batch_size", 8))
    max_epochs = args.max_epochs or int(config.get("max_epochs", 3))
    learning_rate = args.learning_rate or float(config.get("learning_rate", 2e-5))
    use_wandb = args.use_wandb or bool(config.get("use_wandb", False))
    configured_strategy = args.strategy or config.get("strategy", "auto")
    configured_devices = args.devices or config.get("devices", "auto")
    num_nodes = args.num_nodes or int(config.get("num_nodes", 1))
    precision = args.precision or config.get("precision", "32-true")
    accelerator = args.accelerator or config.get("accelerator", "auto")

    torchrun_mode = is_distributed()
    if torchrun_mode:
        accelerator = "gpu"
        configured_strategy = "ddp"
        configured_devices = 1

    if isinstance(configured_devices, str) and configured_devices.isdigit():
        configured_devices = int(configured_devices)

    if train_data is None:
        raise ValueError("A train-data path must be provided via --train-data or the config file")

    print(f"[train_classifier] Loading data from {train_data}", flush=True)
    datamodule = RouterDataModule(
        train_data=train_data,
        val_data=args.val_data,
        model_name=model_name,
        batch_size=batch_size,
    )
    model = RouterLightningModule(model_name=model_name, learning_rate=learning_rate)

    callbacks = [ModelCheckpoint(monitor="val_accuracy", mode="max", save_top_k=1)]
    logger = WandbLogger(project="adaptive-rag", log_model=False) if use_wandb and is_main_process() else False

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=accelerator,
        devices=configured_devices,
        strategy=configured_strategy,
        num_nodes=num_nodes,
        precision=precision,
    )
    print("[train_classifier] Starting training", flush=True)
    trainer.fit(model, datamodule=datamodule)
    print("[train_classifier] Training complete", flush=True)


if __name__ == "__main__":
    main()