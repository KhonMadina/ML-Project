#!/usr/bin/env python3
"""
Transformer baselines (XLM-R / mBERT) fine-tuning for Khmer sentiment (POS/NEG/NEU).

Features:
- Uses HuggingFace Transformers Trainer for text classification.
- Supports either predefined splits (final_train/val/test.csv) or in-script splits.
- Applies the same normalization used by the baseline pipeline (normalization.json saved with the model).
- Logs metrics and saves model, tokenizer, and normalization config in output_dir.

Inputs: CSV with columns id,text,label[,<group_column>]
Labels: POS, NEG, NEU

Examples:
  # Train XLM-R using existing splits
  python modeling/train_transformer.py \
    --input annotation/sample_data/final_dataset.csv \
    --use_splits \
    --output_dir models/xlmr_base \
    --model_name xlm-roberta-base \
    --normalize_all

  # Train mBERT with random stratified split
  python modeling/train_transformer.py \
    --input annotation/sample_data/final_dataset.csv \
    --output_dir models/mbert_base \
    --model_name bert-base-multilingual-cased

Dependencies:
  pip install transformers datasets accelerate evaluate torch
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import sys

# Shared data and evaluation utilities
from .data import (
    Row,
    read_csv_rows,
    stratified_split,
)
from .eval import LABEL_ORDER_DEFAULT

# Experiment utilities for reproducibility and tracking
from .utils.experiment import Config as ExpConfig, prepare_experiment, set_global_seed

# Torch for device selection; guard CUDA access via utils.device
import torch
from .utils.device import get_device

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    # transformers and evaluate are optional runtime dependencies; add type ignores
    # so static analyzers (e.g., Pylance/mypy) don't error when they are missing
    from transformers import (  # type: ignore[import]
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
        set_seed,
    )
    import evaluate  # type: ignore[import]
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install transformers datasets accelerate evaluate torch\n"
        f"Underlying import error: {e}"
    )

# Prefer explicit Trainer.report_to over global WANDB_DISABLED env flags for
# future compatibility with transformers v5+. Keep the existing env defaults as
# a safe fallback but discourage new usage.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

from .text_normalization import (
    build_norm_config_from_args,
    save_norm_config,
    normalize_text,
)

LABELS = LABEL_ORDER_DEFAULT
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def to_dataset(rows: List[Row], tokenizer, max_length: int, norm_cfg: Optional[Dict[str, object]]):
    texts = []
    labels = []
    ids = []
    for r in rows:
        t = r.text
        if norm_cfg:
            t = normalize_text(t, norm_cfg)
        # Validate label to avoid passing invalid targets (e.g., -1) into the model
        if r.label not in LABEL2ID:
            raise ValueError(
                f"Unknown label '{r.label}' for id={r.id}. "
                f"Expected one of {list(LABEL2ID.keys())}."
            )
        texts.append(t)
        labels.append(LABEL2ID[r.label])
        ids.append(r.id)
    enc = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
    data = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "id": ids,
        "text": texts,
    }
    try:
        from datasets import Dataset # type: ignore
        return Dataset.from_dict(data)
    except Exception as e:
        raise SystemExit("datasets package is required: pip install datasets")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a transformer baseline for Khmer sentiment")
    ap.add_argument("--input", required=True, help="Path to finalized dataset CSV (or any of the split CSVs)")
    ap.add_argument("--use_splits", action="store_true", help="Use final_train/val/test.csv next to --input")
    ap.add_argument("--output_dir", required=True, help="Directory to save model artifacts")

    # Model and training params
    ap.add_argument("--model_name", default="xlm-roberta-base", help="HF model name (e.g., xlm-roberta-base, bert-base-multilingual-cased)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")

    # Device control
    # Default to CPU to avoid issues on machines with old/broken CUDA drivers; "auto" still prefers CUDA when safe.
    ap.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device preference: auto picks CUDA if safe, else CPU")
    ap.add_argument("--cuda_device", type=int, default=None, help="CUDA device index when using --device cuda/auto")

    # Splits and reproducibility
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # Reproducibility & tracking config
    ap.add_argument("--config", help="Optional YAML config for experiment settings")
    ap.add_argument("--tracking", choices=["none", "mlflow", "wandb"], default="none", help="Experiment tracking backend")
    ap.add_argument("--experiment_name", default="transformer_baseline", help="Experiment/run name")
    ap.add_argument("--mlflow_tracking_uri", default=None, help="MLflow tracking URI (default: local ./mlruns)")
    ap.add_argument("--mlflow_experiment", default=None, help="MLflow experiment name")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_mode", default=None)

    # Group and normalization
    ap.add_argument("--group_column", default=None)
    # Normalization flags (aligned with text_normalization.build_norm_config_from_args)
    ap.add_argument("--normalize_all", action="store_true", help="Enable a default Khmer/social text normalization pipeline")
    ap.add_argument("--norm_nfc", action="store_true", help="Apply Unicode NFC normalization")
    ap.add_argument("--norm_whitespace", action="store_true", help="Collapse whitespace and trim")
    ap.add_argument("--norm_punct", action="store_true", help="Normalize punctuation variants and compress repeats")
    ap.add_argument("--norm_elongation", action="store_true", help="Compress elongated character runs (>2 -> 2)")
    ap.add_argument("--norm_emoji", choices=["keep", "remove", "map"], default=None, help="Emoji handling mode: keep/remove/map-to-token")
    ap.add_argument("--norm_zero_width", action="store_true", help="Remove zero-width characters (ZWSP, ZWJ, ZWNJ, BOM)")
    ap.add_argument("--norm_khmer_digits", choices=["keep", "map"], default=None, help="Khmer digit handling: keep as-is or map to ASCII digits")
    ap.add_argument("--norm_khmer_punct", action="store_true", help="Normalize Khmer punctuation to ASCII equivalents and compress iteration marks")
    ap.add_argument("--norm_diacritics_reorder", action="store_true", help="Reorder combining diacritics into canonical order after NFC")
    ap.add_argument("--norm_latin_action", choices=["none", "tag", "strip"], default=None, help="How to handle high Latin code-switching: none/tag/strip")
    ap.add_argument("--norm_latin_threshold", type=float, default=None, help="Threshold (0-1) of Latin letters to trigger latin_action when enabled")

    args = ap.parse_args()

    # Resolve device safely (CUDA if healthy, else CPU)
    dev_ctx = get_device(args.device, args.cuda_device)
    print(f"Using device: {dev_ctx.device}")

    # Prepare experiment (seed + tracking)
    exp_cfg = ExpConfig.from_yaml(args.config)
    exp_cfg.merge_overrides({
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "tracking": args.tracking,
        "output_dir": args.output_dir,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "mlflow_experiment": args.mlflow_experiment,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
    })
    tracker = prepare_experiment(exp_cfg)
    set_seed(exp_cfg.seed)

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read rows and splits
    if args.use_splits:
        base = input_path.parent
        train_p = base / "final_train.csv"
        val_p = base / "final_val.csv"
        test_p = base / "final_test.csv"
        if not (train_p.exists() and val_p.exists() and test_p.exists()):
            raise SystemExit(f"--use_splits set but split files not found: {train_p}, {val_p}, {test_p}")
        train_rows = read_csv_rows(train_p, group_col=args.group_column)
        val_rows = read_csv_rows(val_p, group_col=args.group_column)
        test_rows = read_csv_rows(test_p, group_col=args.group_column)
    else:
        rows = read_csv_rows(input_path, group_col=args.group_column)
        train_rows, val_rows, test_rows = stratified_split(rows, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)

    # Normalization config
    norm_cfg = build_norm_config_from_args(args)

    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ds_train = to_dataset(train_rows, tokenizer, args.max_length, norm_cfg)
    ds_val = to_dataset(val_rows, tokenizer, args.max_length, norm_cfg)
    ds_test = to_dataset(test_rows, tokenizer, args.max_length, norm_cfg)

    # Model
    # OSError 1455 ("The paging file is too small for this operation to complete") is a Windows
    # system-level out-of-memory error when allocating tensors for a large checkpoint. We
    # explicitly load on CPU with low_cpu_mem_usage and surface a clearer message with
    # mitigation guidance.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    except OSError as e:
        msg = str(e).lower()
        if "1455" in str(e) or "paging file is too small" in msg or "out of memory" in msg:
            raise SystemExit(
                "Failed to load transformer weights due to insufficient system memory (likely Windows error 1455).\n"
                "The selected model checkpoint is too large for available RAM + page file.\n"
                "Mitigations:\n"
                "  - Use a smaller model (e.g., 'prajjwal1/bert-tiny', 'distilbert-base-multilingual-cased').\n"
                "  - Close other memory-intensive applications and retry.\n"
                "  - Increase the Windows paging file size.\n"
                "  - Reduce sequence length (--max_length) and/or batch size (--batch_size).\n"
                f"Original error: {e}"
            )
        raise

    # Metrics function
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1m = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "f1_macro": f1m}

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Determine precision from device context
    if dev_ctx.is_cuda:
        safe_fp16 = bool(args.fp16)
    else:
        if args.fp16:
            print("Warning: fp16 requested but no usable CUDA device found; disabling fp16.")
        safe_fp16 = False

    # Use a minimal set of TrainingArguments fields compatible with a wide range of transformers versions.
    # More advanced options like evaluation_strategy/save_strategy/load_best_model_at_end can be added
    # if your installed transformers version supports them.
    # Configure Trainer logging/backend behavior explicitly instead of relying on
    # WANDB_DISABLED environment variables. This is the recommended pattern in
    # transformers >= 4.40 and future-proofs for v5.
    report_to: str | list[str] | None
    if exp_cfg.tracking == "wandb":
        report_to = ["wandb"]
    elif exp_cfg.tracking in {"mlflow", "none"}:
        # We log to mlflow separately via our experiment utils; disable HF integrations.
        report_to = "none"
    else:
        report_to = "none"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        fp16=safe_fp16,
        seed=args.seed,
        logging_steps=50,
        max_grad_norm=1.0,
        use_cpu=not dev_ctx.is_cuda,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on val and test
    val_metrics = trainer.evaluate(ds_val)
    test_metrics = trainer.evaluate(ds_test)

    # Log to tracker
    try:
        tracker.log_metrics({
            "val_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
            "val_f1_macro": float(val_metrics.get("eval_f1_macro", val_metrics.get("eval_f1", 0.0))),
            "test_accuracy": float(test_metrics.get("eval_accuracy", 0.0)),
            "test_f1_macro": float(test_metrics.get("eval_f1_macro", test_metrics.get("eval_f1", 0.0))),
        })
    except Exception:
        pass

    # Save model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Save normalization config
    try:
        save_norm_config(out_dir, norm_cfg)
    except Exception as e:
        print(f"Warning: failed to save normalization config: {e}")

    # Save custom metrics.json
    metrics = {
        "label_order": LABELS,
        "val": val_metrics,
        "test": test_metrics,
        "args": {
            "input": str(input_path),
            "use_splits": bool(args.use_splits),
            "output_dir": str(out_dir),
            "model_name": args.model_name,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "max_length": int(args.max_length),
            "grad_accum": int(args.grad_accum),
            "fp16": bool(args.fp16),
            "device": args.device,
            "cuda_device": int(args.cuda_device) if args.cuda_device is not None else None,
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "group_column": args.group_column,
            # Normalization flags and resolved config for reproducibility
            "normalize_all": bool(args.normalize_all),
            "norm_nfc": bool(args.norm_nfc),
            "norm_whitespace": bool(args.norm_whitespace),
            "norm_punct": bool(args.norm_punct),
            "norm_elongation": bool(args.norm_elongation),
            "norm_emoji": args.norm_emoji if args.norm_emoji is not None else None,
            "norm_zero_width": bool(getattr(args, "norm_zero_width", False)),
            "norm_khmer_digits": args.norm_khmer_digits if getattr(args, "norm_khmer_digits", None) is not None else None,
            "norm_khmer_punct": bool(getattr(args, "norm_khmer_punct", False)),
            "norm_diacritics_reorder": bool(getattr(args, "norm_diacritics_reorder", False)),
            "norm_latin_action": args.norm_latin_action if getattr(args, "norm_latin_action", None) is not None else None,
            "norm_latin_threshold": float(args.norm_latin_threshold) if getattr(args, "norm_latin_threshold", None) is not None else None,
        },
        "env": {
            "python": sys.version.replace("\n", " "),
        }
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Track artifacts
    try:
        tracker.log_artifact(out_dir / "metrics.json")
        tracker.log_artifact(out_dir / "pytorch_model.bin", artifact_path="artifacts")
        tracker.log_artifact(out_dir / "config.json", artifact_path="artifacts")
        tracker.log_artifact(out_dir / "tokenizer.json", artifact_path="artifacts")
    except Exception:
        pass

    print(f"Saved transformer model and metrics to {out_dir}")


if __name__ == "__main__":
    main()
