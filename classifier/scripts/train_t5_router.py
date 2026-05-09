#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import torch
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from classifier.scripts.router_data import AdaptiveRouterDataModule
from classifier.scripts.router_model import T5RouterModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--model-name-or-path", default="t5-large")
    parser.add_argument("--output-dir", default="classifier/checkpoints/t5_large_router")
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=4)
    parser.add_argument("--prompt-template", default="Question: {question} Complexity:")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", default="16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--seed", type=int, default=13370)
    parser.add_argument("--wandb-project", default="adaptive-rag-router")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--save-top-k", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    pl.seed_everything(args.seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = AdaptiveRouterDataModule(
        train_file=args.train_file,
        validation_file=args.validation_file,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        prompt_template=args.prompt_template,
    )
    model = T5RouterModule(
        model_name_or_path=args.model_name_or_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    logger = None
    if args.wandb_mode != "disabled":
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir=str(output_dir),
            offline=args.wandb_mode == "offline",
            log_model=False,
        )
        logger.log_hyperparams(vars(args))

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="epoch={epoch:02d}-val_acc={val/accuracy:.4f}",
        monitor="val/accuracy",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=False,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        max_epochs=args.max_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/accuracy", mode="max", patience=args.early_stopping_patience),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=dm)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val accuracy: {checkpoint_cb.best_model_score}")

    torch.cuda.empty_cache()
    if checkpoint_cb.best_model_path:
        best_model = T5RouterModule.load_from_checkpoint(checkpoint_cb.best_model_path)
        hf_dir = output_dir / "hf_best"
        hf_dir.mkdir(parents=True, exist_ok=True)
        best_model.model.save_pretrained(hf_dir)
        best_model.tokenizer.save_pretrained(hf_dir)
        print(f"Saved HF model to: {hf_dir}")

if __name__ == "__main__":
    main()
