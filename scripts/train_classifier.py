import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.classifier.datamodule import RouterDataModule
from src.classifier.lightning_module import RouterLightningModule
from src.data.file_loader import load_yaml_config
from src.utils.distributed import is_distributed, is_main_process


def main():
    print("[train_classifier] Starting trainer setup", flush=True)
    parser = argparse.ArgumentParser(description="Train the router classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to training config")
    args = parser.parse_args()

    config = load_yaml_config(args.config, section="train")
    train_data = str(config["train_data"])
    model_name = str(config["model_name"])
    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])
    learning_rate = float(config["learning_rate"])
    use_wandb = bool(config["use_wandb"])
    configured_strategy = config["strategy"]
    configured_devices = config["devices"]
    num_nodes = int(config["num_nodes"])
    precision = config["precision"]
    accelerator = config["accelerator"]
    val_data = config.get("val_data")

    torchrun_mode = is_distributed()
    if torchrun_mode:
        accelerator = "gpu"
        configured_strategy = "ddp"
        configured_devices = 1

    if isinstance(configured_devices, str) and configured_devices.isdigit():
        configured_devices = int(configured_devices)

    print(f"[train_classifier] Loading data from {train_data}", flush=True)
    datamodule = RouterDataModule(
        train_data=train_data,
        val_data=val_data,
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
