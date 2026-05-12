#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from classifier.scripts.deberta_router_data import AdaptiveRouterClassificationDataModule, LABEL2ID
from classifier.scripts.deberta_router_model import DebertaRouterModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--model-name-or-path", default="microsoft/deberta-v3-large")
    parser.add_argument("--output-dir", default="classifier/checkpoints/deberta_v3_large_router")

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-template", default="{question}")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)

    parser.add_argument("--precision", default="16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--strategy", default="auto")

    parser.add_argument("--seed", type=int, default=13370)
    parser.add_argument("--wandb-project", default="adaptive-rag-router")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--monitor", default="val/macro_f1", choices=["val/macro_f1", "val/accuracy", "val/loss"])
    parser.add_argument("--save-top-k", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)

    parser.add_argument("--use-class-weights", action="store_true")
    return parser.parse_args()


def compute_class_weights(train_file: str) -> list[float]:
    with open(train_file, "r", encoding="utf-8") as f:
        if train_file.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    counts = Counter()
    for row in records:
        label = str(row.get("answer", "")).strip()
        if label in LABEL2ID:
            counts[label] += 1

    total = sum(counts.values())
    weights = []
    for label in ["A", "B", "C"]:
        count = counts.get(label, 0)
        weights.append((total / (3 * count)) if count else 1.0)

    print("Class counts:", dict(counts))
    print("Class weights [A,B,C]:", weights)
    return weights


def main() -> None:
    args = parse_args()

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"

    pl.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = AdaptiveRouterClassificationDataModule(
        train_file=args.train_file,
        validation_file=args.validation_file,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        text_template=args.text_template,
    )

    class_weights = compute_class_weights(args.train_file) if args.use_class_weights else None

    model = DebertaRouterModule(
        model_name_or_path=args.model_name_or_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        class_weights=class_weights,
    )

    logger = None
    if args.wandb_mode != "disabled":
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir=str(output_dir / "wandb"),
            offline=args.wandb_mode == "offline",
            log_model=False,
        )
        logger.log_hyperparams(vars(args))

    monitor_mode = "min" if args.monitor == "val/loss" else "max"
    filename_metric = args.monitor.replace("/", "_")
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename=f"epoch={{epoch:02d}}-{filename_metric}={{{filename_metric}:.4f}}",
        monitor=args.monitor,
        mode=monitor_mode,
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=args.monitor,
            mode=monitor_mode,
            patience=args.early_stopping_patience,
            min_delta=0.0,
        ),
    ]

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)

    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best score: {checkpoint_cb.best_model_score}")

    if checkpoint_cb.best_model_path:
        best_model = DebertaRouterModule.load_from_checkpoint(checkpoint_cb.best_model_path)
        hf_dir = output_dir / "hf_best"
        hf_dir.mkdir(parents=True, exist_ok=True)
        best_model.model.save_pretrained(hf_dir)
        best_model.tokenizer.save_pretrained(hf_dir)
        print(f"Saved HF model to: {hf_dir}")


if __name__ == "__main__":
    main()
