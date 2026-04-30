import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.train_router.classifier_datamodule import RouterDataModule
from src.train_router.classifier_lightning import RouterLightningModule
from myproject.src.file_loader import load_yaml_config


def run_train_router(config_path: str = "config.yaml") -> None:
    print("[train_classifier] Starting trainer setup", flush=True)

    paths = load_yaml_config(config_path, section="paths")
    config = load_yaml_config(config_path, section="train")
    model_cfg = load_yaml_config(config_path, section="model")

    train_data = str(paths["train_data"])
    model_name = str(config.get("model_name", model_cfg.get("model_name", "bert-base-uncased")))
    batch_size = int(config["batch_size"])
    max_epochs = int(config["max_epochs"])
    learning_rate = float(config.get("learning_rate", model_cfg.get("learning_rate", 2e-5)))
    use_wandb = bool(config["use_wandb"])
    configured_strategy = config["strategy"]
    configured_devices = config["devices"]
    num_nodes = int(config["num_nodes"])
    precision = config["precision"]
    accelerator = config["accelerator"]
    val_data = str(paths["validation_data"])

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


def is_distributed() -> bool:
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1


def is_main_process() -> bool:
    import os

    rank = int(os.environ.get("RANK", "0"))
    return rank == 0
