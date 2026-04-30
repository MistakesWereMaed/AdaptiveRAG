from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.file_loader import load_yaml_config
from src.train_router.classifier_datamodule import RouterDataModule
from src.train_router.classifier_lightning import RouterLightningModule


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _existing_optional_path(value):
    if value is None:
        return None
    path = Path(value)
    return str(path) if path.exists() else None


def run_train_router(config_path: str = "config.yaml") -> None:
    paths = load_yaml_config(config_path, section="paths")
    train_cfg = load_yaml_config(config_path, section="train")
    model_cfg = load_yaml_config(config_path, section="model")

    model_name = str(model_cfg.get("model_name", "microsoft/deberta-v3-base"))

    train_data = paths.get("labeled_train")
    val_data = _existing_optional_path(paths.get("labeled_validation"))

    batch_size = int(train_cfg.get("batch_size", model_cfg.get("batch_size", 16)))
    max_length = int(model_cfg.get("max_length", train_cfg.get("max_length", 128)))
    max_epochs = int(train_cfg.get("max_epochs", 3))
    learning_rate = float(model_cfg.get("learning_rate", train_cfg.get("learning_rate", 2e-5)))
    weight_decay = float(model_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.01)))
    dropout = float(model_cfg.get("dropout", 0.1))
    num_workers = int(train_cfg.get("num_workers", 0))
    val_split = float(train_cfg.get("val_split", 0.1))
    seed = int(train_cfg.get("seed", 42))

    accelerator = str(train_cfg.get("accelerator", "auto"))
    devices = train_cfg.get("devices", "auto")
    strategy = train_cfg.get("strategy", "auto")
    precision = train_cfg.get("precision", "16-mixed")
    num_nodes = int(train_cfg.get("num_nodes", 1))
    use_wandb = bool(train_cfg.get("use_wandb", False))

    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    if is_distributed():
        accelerator = "gpu"
        devices = "auto"
        strategy = "ddp"

    pl.seed_everything(seed, workers=True)

    datamodule = RouterDataModule(
        train_data=train_data,
        val_data=val_data,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        val_split=val_split,
        seed=seed,
    )

    model = RouterLightningModule(
        model_name=model_name,
        num_classes=int(model_cfg.get("num_classes", 3)),
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
    )

    checkpoint_dir = Path(train_cfg.get("checkpoints_dir", "checkpoints"))

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="router-{epoch:02d}-{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [checkpoint, LearningRateMonitor(logging_interval="epoch")]

    patience = train_cfg.get("early_stopping_patience")
    if patience is not None:
        callbacks.append(EarlyStopping(monitor="val_accuracy", mode="max", patience=int(patience)))

    logger = False
    if use_wandb and is_main_process():
        logger = WandbLogger(
            project=str(train_cfg.get("wandb_project", "adaptive-rag")),
            name=str(train_cfg.get("wandb_name", "router")),
            log_model=False,
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(checkpoint_dir),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 25)),
        deterministic=bool(train_cfg.get("deterministic", False)),
    )

    trainer.fit(model, datamodule=datamodule)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Adaptive-RAG router classifier")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    run_train_router(args.config)


if __name__ == "__main__":
    main()
