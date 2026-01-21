#!/usr/bin/env python3
"""
Transformer baselines (XLM-R / mBERT) fine-tuning for Khmer/English sentiment (POS/NEG/NEU).

New features in this version:
- Bilingual support via --lang_column and optional stratification by language
- Optional dual-head classification (--dual_head) with language-routed heads
- Optional focal loss (--focal_loss) and class weighting (--class_weighting)
- Per-language metrics and calibration (ECE/Brier) reporting for val/test
- Backwards compatible defaults when no language column/options are provided

Dependencies:
  pip install transformers datasets accelerate evaluate torch
"""
from __future__ import annotations

import argparse
import json
import math
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
import torch.nn as nn
import torch.nn.functional as F
from .utils.device import get_device
# Limit PyTorch CPU thread usage to reduce instability on Windows without MKL/OpenMP issues
try:
    torch.set_num_threads(1)
except Exception:
    pass

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
    from transformers import (  # type: ignore[import]
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModel,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
        set_seed,
    )
    import evaluate  # type: ignore[import]
    try:
        EVAL_METRIC_ACC = evaluate.load("accuracy")
        EVAL_METRIC_F1 = evaluate.load("f1")
    except Exception:
        EVAL_METRIC_ACC = None
        EVAL_METRIC_F1 = None
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install transformers datasets accelerate evaluate torch\n"
        f"Underlying import error: {e}"
    )

# Prefer explicit Trainer.report_to over global WANDB_DISABLED env flags
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
# Disable parallelism in HuggingFace tokenizers on Windows to avoid crashes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .text_normalization import (
    build_norm_config_from_args,
    save_norm_config,
    normalize_text,
)
from .calibration_utils import compute_calibration_summary, probs_from_logits

LABELS = LABEL_ORDER_DEFAULT
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class LangMap:
    lang_to_idx: Dict[str, int]
    idx_to_lang: Dict[int, str]


def build_lang_map(rows: List[Row], explicit_langs: Optional[List[str]] = None) -> Optional[LangMap]:
    langs: List[str] = []
    if explicit_langs:
        langs = [str(x).lower() for x in explicit_langs]
    else:
        seen = set()
        for r in rows:
            if r.lang:
                seen.add(str(r.lang).lower())
        langs = sorted(seen)
    if not langs:
        return None
    l2i = {l: i for i, l in enumerate(langs)}
    i2l = {i: l for l, i in l2i.items()}
    return LangMap(l2i, i2l)


def to_dataset(rows: List[Row], tokenizer, max_length: int, norm_cfg: Optional[Dict[str, object]], lang_map: Optional[LangMap]):
    texts = []
    labels = []
    ids = []
    lang_idx = []
    for r in rows:
        t = r.text
        if norm_cfg:
            t = normalize_text(t, norm_cfg)
        if r.label not in LABEL2ID:
            raise ValueError(
                f"Unknown label '{r.label}' for id={r.id}. Expected one of {list(LABEL2ID.keys())}."
            )
        texts.append(t)
        labels.append(LABEL2ID[r.label])
        ids.append(r.id)
        if lang_map is not None and r.lang is not None:
            lang_idx.append(lang_map.lang_to_idx.get(str(r.lang).lower(), -1))
        elif lang_map is not None:
            lang_idx.append(-1)
    enc = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
    data = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "id": ids,
        "text": texts,
    }
    if lang_map is not None:
        data["lang_idx"] = lang_idx
    try:
        from datasets import Dataset  # type: ignore
        return Dataset.from_dict(data)
    except Exception as e:
        raise SystemExit("datasets package is required: pip install datasets")


class LanguageRoutedSequenceClassifier(nn.Module):
    """A lightweight language-routed classifier on top of a base transformer.

    - Uses base AutoModel to get last_hidden_state and CLS token representation.
    - Applies language-specific Linear heads when lang_idx is provided; else default head.
    - Supports class weighting and focal loss in compute_loss (via external wrapper Trainer).
    """

    def __init__(self, model_name: str, num_labels: int, langs: List[str]):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.base.config, "hidden_size", None)
        if hidden is None:
            raise RuntimeError("Base model config missing hidden_size")
        dropout_p = getattr(self.base.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_p)
        self.num_labels = int(num_labels)
        self.langs = [str(l).lower() for l in langs]
        self.lang_to_idx = {l: i for i, l in enumerate(self.langs)}
        # language-specific heads
        self.heads = nn.ModuleDict({l: nn.Linear(hidden, num_labels) for l in self.langs})
        # default head for unknown/missing language values
        self.default_head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, lang_idx=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # CLS pooling (works for BERT/XLM-R); for models without CLS, mean pooling would be needed
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)

        if lang_idx is None:
            logits = self.default_head(cls)
        else:
            B = cls.shape[0]
            device = cls.device
            logits = torch.zeros((B, self.num_labels), dtype=cls.dtype, device=device)
            # Route by language indices present in the batch
            unique_vals = torch.unique(lang_idx.detach())
            for li in unique_vals.tolist():
                if li < 0:
                    continue
                if 0 <= li < len(self.langs):
                    lkey = self.langs[li]
                    head = self.heads.get(lkey, None)
                    if head is not None:
                        mask = (lang_idx == li)
                        if mask.any():
                            logits[mask] = head(cls[mask])
            # Remaining items route to default head
            default_mask = ~((lang_idx >= 0) & (lang_idx < len(self.langs)))
            if default_mask.any():
                logits[default_mask] = self.default_head(cls[default_mask])

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


class WeightedFocalTrainer(Trainer):
    """Trainer with optional focal loss and class weighting.

    Args
    - class_weights: Optional list/array of shape (num_labels,) mapped to LABELS order
    - focal_loss: bool to enable focal loss
    - focal_gamma: focusing parameter gamma
    """

    def __init__(self, *args, class_weights=None, focal_loss: bool = False, focal_gamma: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            import numpy as _np
            w = _np.asarray(class_weights, dtype=_np.float32)
            if w.ndim != 1:
                raise ValueError("class_weights must be 1D")
            self.class_weights = torch.tensor(w, dtype=torch.float32)
        self.focal_loss = bool(focal_loss)
        self.focal_gamma = float(focal_gamma)

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int | None = None, **kwargs):
        """Custom loss supporting class weights and focal loss.

        Transformers may pass extra keyword arguments (e.g., num_items_in_batch) to compute_loss.
        Accept them for compatibility and ignore if unused.
        """
        labels = inputs.get("labels")
        # Exclude labels when forwarding through the model
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs["logits"]
        # Resolve class weights tensor if provided
        cw = self.class_weights.to(logits.device) if self.class_weights is not None else None
        if self.focal_loss:
            # Focal loss on probabilities; numerical stability by log-softmax
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            nll = F.nll_loss(logp, labels, reduction="none", weight=cw)
            loss = ((1 - p.gather(1, labels.unsqueeze(1)).squeeze(1)) ** self.focal_gamma) * nll
            loss = loss.mean()
        else:
            loss = F.cross_entropy(logits, labels, weight=cw)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(rows: List[Row]) -> List[float]:
    from collections import Counter
    cnt = Counter([r.label for r in rows])
    total = sum(cnt.values())
    weights: List[float] = []
    for l in LABELS:
        c = cnt.get(l, 0)
        if c <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (len(LABELS) * c))
    return weights


def per_language_metrics(preds_logits: np.ndarray, labels: np.ndarray, lang_idx: Optional[np.ndarray], idx_to_lang: Optional[Dict[int, str]]):
    metric_acc = EVAL_METRIC_ACC or evaluate.load("accuracy")
    metric_f1 = EVAL_METRIC_F1 or evaluate.load("f1")
    out: Dict[str, float] = {}
    if lang_idx is None or idx_to_lang is None:
        return out
    for li, lang in idx_to_lang.items():
        mask = (lang_idx == li)
        if mask.sum() <= 0:
            continue
        p = preds_logits[mask].argmax(axis=-1)
        y = labels[mask]
        acc = metric_acc.compute(predictions=p, references=y)["accuracy"]
        f1m = metric_f1.compute(predictions=p, references=y, average="macro")["f1"]
        out[f"accuracy__{lang}"] = float(acc)
        out[f"f1_macro__{lang}"] = float(f1m)
    return out


def add_common_args(ap: argparse.ArgumentParser) -> None:
    # Model and training params
    ap.add_argument("--model_name", default="xlm-roberta-base", help="HF model name (e.g., xlm-roberta-base, bert-base-multilingual-cased)")
    ap.add_argument("--tokenizer_path", default=None, help="Optional path to a custom tokenizer directory (overrides --model_name tokenizer)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; 0 is safest on Windows")
    ap.add_argument("--grad_checkpointing", action="store_true", help="Enable gradient checkpointing to reduce memory usage")

    # Device control
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

    # Group, language and normalization
    ap.add_argument("--group_column", default=None)
    ap.add_argument("--lang_column", default=None, help="Optional language column name (e.g., 'lang')")
    ap.add_argument("--langs", nargs="*", default=None, help="Explicit list of known languages (e.g., km en). If omitted, inferred from data")
    ap.add_argument("--stratify_by_lang", action="store_true", help="Stratify by (label,lang) when creating splits")

    # Dual-head routing and loss shaping
    ap.add_argument("--dual_head", action="store_true", help="Use language-routed dual-head classification")
    ap.add_argument("--class_weighting", choices=["none", "balanced"], default="none", help="Enable class weighting in loss")
    ap.add_argument("--focal_loss", action="store_true", help="Enable focal loss")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")

    # Normalization flags
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a transformer for Khmer/English sentiment with bilingual features")
    ap.add_argument("--input", required=True, help="Path to finalized dataset CSV (or any of the split CSVs)")
    ap.add_argument("--use_splits", action="store_true", help="Use final_train/val/test.csv next to --input")
    ap.add_argument("--output_dir", required=True, help="Directory to save model artifacts")
    add_common_args(ap)
    args = ap.parse_args()

    # Resolve device safely
    dev_ctx = get_device(args.device, args.cuda_device)
    print(f"Using device: {dev_ctx.device}")
    try:
        if dev_ctx.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")  # torch>=2.0
    except Exception:
        pass

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
        train_rows = read_csv_rows(train_p, group_col=args.group_column, lang_col=args.lang_column)
        val_rows = read_csv_rows(val_p, group_col=args.group_column, lang_col=args.lang_column)
        test_rows = read_csv_rows(test_p, group_col=args.group_column, lang_col=args.lang_column)
    else:
        rows = read_csv_rows(input_path, group_col=args.group_column, lang_col=args.lang_column)
        train_rows, val_rows, test_rows = stratified_split(
            rows, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed, stratify_by_lang=bool(args.stratify_by_lang)
        )

    # Build language map if language column present
    lang_map = build_lang_map(train_rows + val_rows + test_rows, explicit_langs=args.langs)

    # Normalization config
    norm_cfg = build_norm_config_from_args(args)

    # Tokenizer and datasets
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ds_train = to_dataset(train_rows, tokenizer, args.max_length, norm_cfg, lang_map)
    ds_val = to_dataset(val_rows, tokenizer, args.max_length, norm_cfg, lang_map)
    ds_test = to_dataset(test_rows, tokenizer, args.max_length, norm_cfg, lang_map)

    # Model selection (single-head vs language-routed dual-head)
    use_cuda = dev_ctx.is_cuda
    if args.dual_head:
        langs = lang_map.lang_to_idx.keys() if lang_map is not None else []
        model = LanguageRoutedSequenceClassifier(args.model_name, num_labels=len(LABELS), langs=list(langs))
    else:
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

    # Enable gradient checkpointing if requested and supported
    try:
        if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Parameter counts
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except Exception:
        total_params = None
        trainable_params = None

    try:
        tracker.log_metrics({
            "model_params_total": float(total_params) if total_params is not None else 0.0,
            "model_params_trainable": float(trainable_params) if trainable_params is not None else 0.0,
        })
    except Exception:
        pass

    # Metrics function (overall only here; per-language computed post-hoc)
    metric_acc = EVAL_METRIC_ACC or evaluate.load("accuracy")
    metric_f1 = EVAL_METRIC_F1 or evaluate.load("f1")

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

    # Reporting backend
    report_to: str | list[str] | None
    if exp_cfg.tracking == "wandb":
        report_to = ["wandb"]
    elif exp_cfg.tracking in {"mlflow", "none"}:
        report_to = "none"
    else:
        report_to = "none"

    try:
        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="linear",
            fp16=safe_fp16,
            seed=args.seed,
            logging_steps=50,
            max_grad_norm=1.0,
            report_to=report_to,
            dataloader_num_workers=args.num_workers,
            dataloader_pin_memory=dev_ctx.is_cuda,
        )
    except TypeError as _e:
        print(f"Warning: falling back to minimal TrainingArguments due to version incompatibility: {_e}")
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
        )

    # Class weighting / focal loss
    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = compute_class_weights(train_rows)

    # Custom Trainer to handle (optional) focal/class-weight losses
    trainer = WeightedFocalTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        focal_loss=bool(args.focal_loss),
        focal_gamma=float(args.focal_gamma),
    )

    # Train
    import time as _time
    _t0 = _time.time()
    trainer.train()
    train_seconds = _time.time() - _t0
    try:
        tracker.log_metrics({"train_seconds": float(train_seconds)})
    except Exception:
        pass

    # Evaluate on val and test (overall)
    val_metrics = trainer.evaluate(ds_val)
    _t1 = _time.time()
    test_metrics = trainer.evaluate(ds_test)
    test_eval_seconds = _time.time() - _t1
    try:
        test_throughput = float(len(ds_test)) / test_eval_seconds if test_eval_seconds > 0 else 0.0
    except Exception:
        test_throughput = 0.0

    # Collect predictions/logits for per-language metrics and calibration
    pred_val = trainer.predict(ds_val)
    pred_test = trainer.predict(ds_test)
    logits_val = pred_val.predictions
    labels_val = pred_val.label_ids
    logits_test = pred_test.predictions
    labels_test = pred_test.label_ids

    # Extract lang_idx arrays if available
    lang_idx_val = None
    lang_idx_test = None
    idx_to_lang = None
    try:
        if lang_map is not None:
            idx_to_lang = lang_map.idx_to_lang
            # datasets map storage -> need to access underlying columns
            lang_idx_val = np.array(ds_val["lang_idx"]) if "lang_idx" in ds_val.column_names else None
            lang_idx_test = np.array(ds_test["lang_idx"]) if "lang_idx" in ds_test.column_names else None
    except Exception:
        pass

    # Per-language metrics
    lang_metrics_val = per_language_metrics(logits_val, labels_val, lang_idx_val, idx_to_lang) if idx_to_lang is not None else {}
    lang_metrics_test = per_language_metrics(logits_test, labels_test, lang_idx_test, idx_to_lang) if idx_to_lang is not None else {}

    # Calibration summaries (overall)
    calib_val = compute_calibration_summary(logits=logits_val, y_idx=labels_val, n_bins=15,
                                            diagram_png=str(out_dir / "val_reliability.png"),
                                            bins_json=str(out_dir / "val_reliability.json"),
                                            title="Val calibration")
    calib_test = compute_calibration_summary(logits=logits_test, y_idx=labels_test, n_bins=15,
                                             diagram_png=str(out_dir / "test_reliability.png"),
                                             bins_json=str(out_dir / "test_reliability.json"),
                                             title="Test calibration")

    # Calibration per-language
    calib_val_lang: Dict[str, Dict[str, float]] = {}
    calib_test_lang: Dict[str, Dict[str, float]] = {}
    if idx_to_lang is not None and lang_idx_val is not None:
        for li, lang in idx_to_lang.items():
            m = (lang_idx_val == li)
            if m.sum() > 0:
                calib_val_lang[lang] = compute_calibration_summary(logits=logits_val[m], y_idx=labels_val[m], n_bins=10)
    if idx_to_lang is not None and lang_idx_test is not None:
        for li, lang in idx_to_lang.items():
            m = (lang_idx_test == li)
            if m.sum() > 0:
                calib_test_lang[lang] = compute_calibration_summary(logits=logits_test[m], y_idx=labels_test[m], n_bins=10)

    # Export standardized test predictions for CI and significance testing
    try:
        import numpy as _np
        import csv as _csv
        preds_test = _np.asarray(logits_test).argmax(axis=-1)
        pred_csv = out_dir / "test_predictions.csv"
        with pred_csv.open("w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["id", "label", "pred_label"])
            ids = ds_test["id"] if "id" in ds_test.column_names else [str(i) for i in range(len(preds_test))]
            for i in range(len(preds_test)):
                gold = int(labels_test[i]) if hasattr(labels_test, "__len__") else labels_test[i]
                gold_lbl = ID2LABEL.get(gold, str(gold))
                pred_lbl = ID2LABEL.get(int(preds_test[i]), str(int(preds_test[i])))
                rid = ids[i]
                w.writerow([rid, str(gold_lbl).upper(), str(pred_lbl).upper()])
    except Exception as _e:
        print(f"Warning: failed to save test_predictions.csv: {_e}")

    # Log to tracker
    try:
        base_metrics = {
            "val_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
            "val_f1_macro": float(val_metrics.get("eval_f1_macro", val_metrics.get("eval_f1", 0.0))),
            "test_accuracy": float(test_metrics.get("eval_accuracy", 0.0)),
            "test_f1_macro": float(test_metrics.get("eval_f1_macro", test_metrics.get("eval_f1", 0.0))),
            "test_eval_seconds": float(test_eval_seconds),
            "test_throughput_examples_per_sec": float(test_throughput),
            # calibration
            "val_ece": float(calib_val.get("ece", 0.0)),
            "val_brier": float(calib_val.get("brier", 0.0)),
            "test_ece": float(calib_test.get("ece", 0.0)),
            "test_brier": float(calib_test.get("brier", 0.0)),
        }
        tracker.log_metrics(base_metrics | {f"val_{k}": v for k, v in lang_metrics_val.items()} | {f"test_{k}": v for k, v in lang_metrics_test.items()})
    except Exception:
        pass

    # Save model and tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    # If a custom tokenizer was used, copy tokenizer_report.json into run dir
    try:
        if args.tokenizer_path:
            from shutil import copyfile
            tok_report_src = Path(args.tokenizer_path) / "tokenizer_report.json"
            tok_report_dst = out_dir / "tokenizer_report.json"
            if tok_report_src.exists():
                copyfile(tok_report_src, tok_report_dst)
                try:
                    tracker.log_artifact(tok_report_dst, artifact_path="artifacts")
                except Exception:
                    pass
    except Exception:
        pass

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
        "per_language": {
            "val": lang_metrics_val,
            "test": lang_metrics_test,
        },
        "calibration": {
            "val": calib_val,
            "test": calib_test,
            "val_by_lang": calib_val_lang,
            "test_by_lang": calib_test_lang,
        },
        "timing": {
            "train_seconds": float(train_seconds) if 'train_seconds' in locals() else None,
            "test_eval_seconds": float(test_eval_seconds) if 'test_eval_seconds' in locals() else None,
            "test_throughput_examples_per_sec": float(test_throughput) if 'test_throughput' in locals() else None,
        },
        "model_params": {
            "total": int(total_params) if total_params is not None else None,
            "trainable": int(trainable_params) if trainable_params is not None else None,
        },
        "args": {
            "input": str(input_path),
            "use_splits": bool(args.use_splits),
            "output_dir": str(out_dir),
            "model_name": args.model_name,
            "tokenizer_path": args.tokenizer_path,
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
            "lang_column": args.lang_column,
            "langs": args.langs,
            "stratify_by_lang": bool(args.stratify_by_lang),
            "dual_head": bool(args.dual_head),
            "class_weighting": args.class_weighting,
            "focal_loss": bool(args.focal_loss),
            "focal_gamma": float(args.focal_gamma),
            # Normalization flags
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
        # Model files: either standard HF or our custom routed model (state_dict)
        if hasattr(model, "save_pretrained"):
            tracker.log_artifact(out_dir / "pytorch_model.bin", artifact_path="artifacts")
            tracker.log_artifact(out_dir / "config.json", artifact_path="artifacts")
        tracker.log_artifact(out_dir / "tokenizer.json", artifact_path="artifacts")
        # Calibration outputs
        tracker.log_artifact(out_dir / "val_reliability.json", artifact_path="artifacts")
        tracker.log_artifact(out_dir / "test_reliability.json", artifact_path="artifacts")
        tracker.log_artifact(out_dir / "val_reliability.png", artifact_path="artifacts")
        tracker.log_artifact(out_dir / "test_reliability.png", artifact_path="artifacts")
    except Exception:
        pass

    print(f"Saved transformer model and metrics to {out_dir}")


if __name__ == "__main__":
    main()
