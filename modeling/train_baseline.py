#!/usr/bin/env python3
"""
Baseline training for Khmer sentiment (POS/NEG/NEU) with enhanced reproducibility and options:
- Character n-gram TF-IDF features (robust for non-segmented scripts).
- Logistic Regression classifier with flexible class weights (balanced/none or JSON mapping).
- Optional group-aware splits to avoid leakage across items/users when a group column is provided.
- Optional probability calibration (Platt/Isotonic) using the validation set.
- Deterministic experiment logging: args, environment versions, seeds, sizes.
- Stratified split (default) or consume existing final_train/val/test CSVs.
- Saves: vectorizer.pkl, model.pkl (possibly calibrated), metrics.json, confusion_matrix.csv.

Input CSV format (UTF-8): id,text,label[,<group_column>]
Labels: POS, NEG, NEU

Examples:
  # Train using a single finalized dataset and random stratified split
  python modeling/train_baseline.py --input annotation/sample_data/final_dataset.csv --output_dir models/baseline_chargram

  # Train using predefined splits produced by finalize_dataset.py --export-splits
  python modeling/train_baseline.py --input annotation/sample_data/final_dataset.csv --use_splits --output_dir models/baseline_chargram

  # Train with group-aware splits (prevent leakage by group id) and calibrate probabilities
  python modeling/train_baseline.py --input data/final_dataset.csv --output_dir models/baseline_chargram \
    --group_column group_id --calibrate platt

Dependencies:
  pip install scikit-learn joblib pandas

If pandas is not installed, the script will fall back to Python's csv module for reading.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Data loading and splitting utilities
from .data import (
    Row,
    read_csv_rows,
    stratified_split,
    group_aware_split,
    group_stratified_split,
    to_xy,
)

# Experiment utilities for reproducibility and tracking
from .utils.experiment import Config as ExpConfig, prepare_experiment

# Local normalization utilities
from .text_normalization import (
    build_norm_config_from_args,
    save_norm_config,
    normalize_corpus,
)
from .calibration_utils import TemperatureScaledModel
from .eval import evaluate_classification, LABEL_ORDER_DEFAULT

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # optional

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit
    try:
        # Available in newer sklearn; optional
        from sklearn.calibration import CalibratedClassifierCV
    except Exception:
        CalibratedClassifierCV = None  # type: ignore
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install scikit-learn joblib pandas\n"
        f"Underlying import error: {e}"
    )

# Optional: imbalanced-learn for resampling
try:
    from imblearn.over_sampling import RandomOverSampler  # type: ignore
    from imblearn.under_sampling import RandomUnderSampler  # type: ignore
except Exception:
    RandomOverSampler = None  # type: ignore
    RandomUnderSampler = None  # type: ignore

try:
    import joblib  # separate package in modern installs
except Exception:
    # sklearn.externals.joblib is deprecated; require joblib
    raise SystemExit("Missing dependency joblib. Install with: pip install joblib")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def class_distribution_list(y: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for lbl in y:
        d[lbl] = d.get(lbl, 0) + 1
    # Normalize order
    out = {k: d.get(k, 0) for k in ["POS", "NEG", "NEU"]}
    # Also include any unexpected labels
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out


def get_env_metadata() -> Dict[str, Optional[str]]:
    meta: Dict[str, Optional[str]] = {
        "python": sys.version.replace("\n", " "),
    }
    try:
        import sklearn  # type: ignore
        meta["sklearn"] = getattr(sklearn, "__version__", None)
    except Exception:
        meta["sklearn"] = None
    try:
        import joblib as _jl  # type: ignore
        meta["joblib"] = getattr(_jl, "__version__", None)
    except Exception:
        meta["joblib"] = None
    try:
        import pandas as _pd  # type: ignore
        meta["pandas"] = getattr(_pd, "__version__", None)
    except Exception:
        meta["pandas"] = None
    try:
        import numpy as _np  # type: ignore
        meta["numpy"] = getattr(_np, "__version__", None)
    except Exception:
        meta["numpy"] = None
    return meta


def parse_class_weight(class_weight: str, class_weight_json: Optional[str]) -> Optional[Dict[str, float] | str]:
    """Return either 'balanced', None, or a dict mapping label->weight from JSON path."""
    if class_weight_json:
        p = Path(class_weight_json)
        if not p.exists():
            raise SystemExit(f"--class_weight_json not found: {p}")
        mapping = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise SystemExit("--class_weight_json must contain an object mapping label -> weight")
        # Ensure values are floats
        out: Dict[str, float] = {}
        for k, v in mapping.items():
            try:
                out[str(k).upper()] = float(v)
            except Exception:
                raise SystemExit(f"Invalid weight for class {k}: {v}")
        return out
    return None if class_weight == "none" else class_weight


def train_and_eval(
    train_rows: List[Row],
    val_rows: List[Row],
    test_rows: List[Row],
    output_dir: Path,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    class_weight_param: Optional[Dict[str, float] | str],
    max_iter: int,
    calibrate: str,
    calibrate_cv_folds: int,
    seed: int,
    args_dict: Dict[str, object],
    resample: str,
    resample_ratio: float,
) -> None:
    ensure_dir(output_dir)

    # Ensure the training split has at least two classes; if not, borrow samples from val/test.
    train_labels = set(r.label for r in train_rows)
    if len(train_labels) < 2:
        all_labels = set(r.label for r in (train_rows + val_rows + test_rows))
        if len(all_labels) < 2:
            raise SystemExit("Dataset contains only one class across all splits; need at least two classes to train a classifier.")
        print("Info: training split has <2 classes; moving examples from val/test to ensure at least two classes.")
        for src in (val_rows, test_rows):
            # iterate on a copy to allow removal
            moved = False
            for i, r in enumerate(list(src)):
                if r.label not in train_labels:
                    train_rows.append(r)
                    del src[i]
                    train_labels.add(r.label)
                    moved = True
                    if len(train_labels) >= 2:
                        break
            if len(train_labels) >= 2:
                break
        if len(train_labels) < 2:
            raise SystemExit("Unable to construct a training split with at least two classes; provide more data or adjust split ratios.")

    X_train, y_train = to_xy(train_rows)
    X_val, y_val = to_xy(val_rows)
    X_test, y_test = to_xy(test_rows)

    # If val or test splits are empty due to tiny dataset or prior adjustments, borrow minimal samples
    # Prefer to keep at least 1 example for val and 1 for test when possible.
    if len(X_val) == 0 and len(X_train) > 1:
        # move one sample from training to validation for evaluation/calibration stability
        r = train_rows.pop()
        val_rows.append(r)
        X_train, y_train = to_xy(train_rows)
        X_val, y_val = to_xy(val_rows)
        print("Info: moved one sample from train to val to avoid empty validation split.")
    if len(X_test) == 0 and len(X_train) > 1:
        r = train_rows.pop()
        test_rows.append(r)
        X_train, y_train = to_xy(train_rows)
        X_test, y_test = to_xy(test_rows)
        print("Info: moved one sample from train to test to avoid empty test split.")

    # Apply normalization if configured (config is passed via args_dict under 'normalization_config')
    norm_config = args_dict.get("normalization_config") if isinstance(args_dict, dict) else None
    if norm_config:
        X_train = normalize_corpus(X_train, norm_config)  # type: ignore
        X_val = normalize_corpus(X_val, norm_config)      # type: ignore
        X_test = normalize_corpus(X_test, norm_config)    # type: ignore

    # Validate dataset size and adapt min_df for tiny training sets to avoid sklearn error:
    # "ValueError: max_df corresponds to < documents than min_df"
    n_docs = len(X_train)
    if n_docs == 0:
        raise SystemExit("Empty training split; provide more data or adjust split ratios.")
    eff_min_df = int(min_df)
    if n_docs < eff_min_df:
        print(f"Info: reducing min_df from {eff_min_df} to {max(1, n_docs)} for {n_docs} training documents.")
        eff_min_df = max(1, n_docs)

    # Adapt n-gram range to the observed text lengths to avoid empty vocabularies
    max_len = max((len(t) for t in X_train), default=0)
    if max_len == 0:
        raise SystemExit("Training texts are empty after preprocessing; cannot build features. Provide non-empty texts or disable aggressive normalization.")
    eff_ng_min = max(1, min(ngram_min, max_len))
    eff_ng_max = max(eff_ng_min, min(ngram_max, max_len))

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(eff_ng_min, eff_ng_max),
        min_df=eff_min_df,
        strip_accents=None,
        lowercase=False,
    )
    try:
        Xtr = vectorizer.fit_transform(X_train)
    except ValueError as e:
        msg = str(e)
        if "After pruning, no terms remain" in msg or "empty vocabulary" in msg:
            print("Info: fallback vectorizer configuration due to tiny corpus; using char n-grams (1,3) with min_df=1.")
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, min(3, max_len)),
                min_df=1,
                strip_accents=None,
                lowercase=False,
            )
            Xtr = vectorizer.fit_transform(X_train)
        else:
            raise

    # Optional resampling on training set only
    resampling_info: Dict[str, object] = {"applied": False}
    if resample != "none":
        if (resample == "oversample" and RandomOverSampler is None) or (resample == "undersample" and RandomUnderSampler is None):
            print("Warning: imbalanced-learn not available; skipping resampling. Install with: pip install imbalanced-learn")
        else:
            y_counts_before = class_distribution_list(y_train)
            if resample == "oversample" and RandomOverSampler is not None:
                # Target each class to max_count * resample_ratio (at least current count)
                maxc = max(y_counts_before.values()) if y_counts_before else 0
                target = {k: max(v, int(round(maxc * max(1.0, resample_ratio)))) for k, v in y_counts_before.items()}
                sampler = RandomOverSampler(sampling_strategy=target, random_state=seed)
                Xtr, y_train = sampler.fit_resample(Xtr, y_train)
            elif resample == "undersample" and RandomUnderSampler is not None:
                # Target each class to min_count * resample_ratio (no more than current count, at least 1)
                minc = min(y_counts_before.values()) if y_counts_before else 0
                tgt = int(round(minc * max(0.0, resample_ratio)))
                if tgt <= 0:
                    tgt = minc
                target = {k: min(v, tgt) for k, v in y_counts_before.items()}
                sampler = RandomUnderSampler(sampling_strategy=target, random_state=seed)
                Xtr, y_train = sampler.fit_resample(Xtr, y_train)
            y_counts_after = class_distribution_list(y_train)
            resampling_info = {
                "applied": True,
                "method": resample,
                "ratio": float(resample_ratio),
                "before": y_counts_before,
                "after": y_counts_after,
            }

    clf = LogisticRegression(
        max_iter=max_iter,
        n_jobs=None,
        class_weight=class_weight_param,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(Xtr, y_train)

    # Evaluate on val and test, with optional calibration
    Xv = vectorizer.transform(X_val) if len(X_val) > 0 else None
    Xt = vectorizer.transform(X_test) if len(X_test) > 0 else None

    model_for_eval = clf
    calibration_info: Dict[str, object] = {"applied": False}
    if calibrate == "temperature":
        try:
            if Xv is None or len(y_val) == 0:
                raise ValueError("No validation data available for temperature scaling")
            ts = TemperatureScaledModel(clf)
            ts.fit(Xv, y_val)
            model_for_eval = ts
            calibration_info = {"applied": True, "method": "temperature", "temperature": float(ts.temperature)}
        except Exception as e:
            print(f"Warning: temperature scaling failed: {e}")
    elif calibrate in ("platt", "isotonic"):
        if CalibratedClassifierCV is None:
            print("Warning: sklearn.calibration.CalibratedClassifierCV not available; skipping calibration.")
        else:
            method = "sigmoid" if calibrate == "platt" else "isotonic"
            try:
                if int(calibrate_cv_folds) and int(calibrate_cv_folds) > 1:
                    calibrator = CalibratedClassifierCV(base_estimator=clf, method=method, cv=int(calibrate_cv_folds))
                    calibrator.fit(Xtr, y_train)
                    model_for_eval = calibrator
                    calibration_info = {"applied": True, "method": calibrate, "cv_folds": int(calibrate_cv_folds)}
                else:
                    if Xv is None or len(y_val) == 0:
                        raise ValueError("No validation data for calibration in prefit mode")
                    calibrator = CalibratedClassifierCV(base_estimator=clf, method=method, cv="prefit")
                    calibrator.fit(Xv, y_val)
                    model_for_eval = calibrator
                    calibration_info = {"applied": True, "method": calibrate, "cv_folds": 0}
            except Exception as e:
                print(f"Warning: calibration failed: {e}")

    # Build metrics with safe handling for possibly empty val/test splits
    if Xv is not None and len(y_val) > 0:
        y_val_pred = model_for_eval.predict(Xv)
        val_split_metrics = evaluate_classification(y_val, y_val_pred, labels=LABEL_ORDER_DEFAULT)
        val_metrics: Dict[str, object] = val_split_metrics.to_dict()
    else:
        val_metrics = {"accuracy": None, "f1_macro": None, "report": {}, "confusion": [[0,0,0],[0,0,0],[0,0,0]]}

    if Xt is not None and len(y_test) > 0:
        y_test_pred = model_for_eval.predict(Xt)
        test_split_metrics = evaluate_classification(y_test, y_test_pred, labels=LABEL_ORDER_DEFAULT)
        test_metrics: Dict[str, object] = test_split_metrics.to_dict()
    else:
        test_metrics = {"accuracy": None, "f1_macro": None, "report": {}, "confusion": [[0,0,0],[0,0,0],[0,0,0]]}

    # Remove non-serializable objects from args (e.g., tracker instance)
    safe_args = dict(args_dict)
    if isinstance(safe_args.get("_tracker", None), object):
        safe_args.pop("_tracker", None)

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "params": {
            "ngram_range": [ngram_min, ngram_max],
            "min_df": min_df,
            "class_weight": class_weight_param,
            "max_iter": max_iter,
            "calibrate": calibrate,
            "calibrate_cv_folds": int(calibrate_cv_folds),
        },
        "sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "label_order": ["POS", "NEG", "NEU"],
        "env": get_env_metadata(),
        "seed": seed,
        "args": safe_args,
        "calibration": calibration_info,
        "resampling": resampling_info,
    }

    # Save artifacts
    joblib.dump(vectorizer, output_dir / "vectorizer.pkl")
    # Save the calibrated model if used, else raw classifier
    joblib.dump(model_for_eval, output_dir / "model.pkl")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Tracking: log metrics and artifacts if enabled via prepare_experiment in main()
    if args_dict.get("_tracker") is not None:
        tr = args_dict.get("_tracker")
        try:
            # flatten a few metrics for logging
            tr.log_metrics({
                "val_accuracy": float(metrics["val"]["accuracy"]),
                "val_f1_macro": float(metrics["val"]["f1_macro"]),
                "test_accuracy": float(metrics["test"]["accuracy"]),
                "test_f1_macro": float(metrics["test"]["f1_macro"]),
            })
            tr.log_artifact(output_dir / "metrics.json")
            tr.log_artifact(output_dir / "confusion_matrix.csv")
            tr.log_artifact(output_dir / "vectorizer.pkl", artifact_path="artifacts")
            tr.log_artifact(output_dir / "model.pkl", artifact_path="artifacts")
        except Exception:
            pass

    # Persist normalization config so inference can reproduce preprocessing
    if isinstance(args_dict, dict) and args_dict.get("normalization_config"):
        try:
            save_norm_config(output_dir, args_dict["normalization_config"])  # type: ignore
        except Exception as e:
            print(f"Warning: failed to save normalization config: {e}")

    # Save test confusion matrix for convenience
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "POS", "NEG", "NEU"])
        for i, row in enumerate(metrics["test"]["confusion"]):
            w.writerow([["POS", "NEG", "NEU"][i]] + row)

    print(f"Saved model and metrics to {output_dir}")
    print(f"Val accuracy={metrics['val']['accuracy']:.4f} F1_macro={metrics['val']['f1_macro']:.4f}")
    print(f"Test accuracy={metrics['test']['accuracy']:.4f} F1_macro={metrics['test']['f1_macro']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a char n-gram TF-IDF + Logistic Regression baseline for Khmer sentiment")
    # Reproducibility & tracking config
    ap.add_argument("--config", help="Optional YAML config for experiment settings")
    ap.add_argument("--tracking", choices=["none", "mlflow", "wandb"], default="mlflow", help="Experiment tracking backend")
    ap.add_argument("--experiment_name", default="baseline_chargram", help="Experiment/run name")
    ap.add_argument("--mlflow_tracking_uri", default=None, help="MLflow tracking URI (default: local ./mlruns)")
    ap.add_argument("--mlflow_experiment", default=None, help="MLflow experiment name")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_mode", default=None)
    ap.add_argument("--input", required=True, help="Path to finalized dataset CSV (or any of the split CSVs)")
    ap.add_argument("--use_splits", action="store_true", help="Use final_train/val/test.csv sitting next to --input; otherwise perform a random stratified split")
    ap.add_argument("--output_dir", required=True, help="Directory to save model artifacts")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ngram_min", type=int, default=3)
    ap.add_argument("--ngram_max", type=int, default=5)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="balanced")
    ap.add_argument("--class_weight_json", help="Optional path to JSON mapping of class label -> weight (overrides --class_weight)")
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--group_column", default=None, help="Optional column name in CSV to perform group-aware splits (prevents leakage)")
    ap.add_argument("--group_stratified", action="store_true", help="Use group-aware stratified split (assign groups to splits while approximating label distribution)")
    ap.add_argument("--calibrate", choices=["none", "platt", "isotonic", "temperature"], default="none", help="Calibrate probabilities (platt/isotonic using val or CV; temperature uses val)")
    ap.add_argument("--calibrate_cv_folds", type=int, default=0, help="For platt/isotonic: if >1, perform cross-validated calibration on training set with given folds; if 0, calibrate on validation set (prefit)")
    ap.add_argument("--resample", choices=["none", "undersample", "oversample"], default="none", help="Optionally apply class resampling on the training set")
    ap.add_argument("--resample_ratio", type=float, default=1.0, help="Resampling ratio; for oversample targets ~max_count*ratio; for undersample targets ~min_count*ratio")

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

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    class_weight_param = parse_class_weight(args.class_weight, args.class_weight_json)

    if args.use_splits:
        base = input_path.parent
        train_p = base / "final_train.csv"
        val_p = base / "final_val.csv"
        test_p = base / "final_test.csv"
        if not (train_p.exists() and val_p.exists() and test_p.exists()):
            raise SystemExit(f"--use_splits set, but split files not found next to {input_path}: {train_p}, {val_p}, {test_p}")
        train_rows = read_csv_rows(train_p, group_col=args.group_column)
        val_rows = read_csv_rows(val_p, group_col=args.group_column)
        test_rows = read_csv_rows(test_p, group_col=args.group_column)
    else:
        rows = read_csv_rows(input_path, group_col=args.group_column)
        if args.group_column:
            if args.group_stratified:
                train_rows, val_rows, test_rows = group_stratified_split(rows, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
            else:
                train_rows, val_rows, test_rows = group_aware_split(rows, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)
        else:
            train_rows, val_rows, test_rows = stratified_split(rows, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)

    # Build normalization config from args (stored in metrics and used during training)
    from .text_normalization import build_norm_config_from_args as _build_norm
    norm_config = _build_norm(args)

    # Log args as dict for metrics
    args_dict = {
        "input": str(input_path),
        "use_splits": bool(args.use_splits),
        "output_dir": str(output_dir),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "ngram_min": int(args.ngram_min),
        "ngram_max": int(args.ngram_max),
        "min_df": int(args.min_df),
        "class_weight": args.class_weight,
        "class_weight_json": str(args.class_weight_json) if args.class_weight_json else None,
        "max_iter": int(args.max_iter),
        "group_column": args.group_column,
        "calibrate": args.calibrate,
        "calibrate_cv_folds": int(args.calibrate_cv_folds),
        "group_stratified": bool(args.group_stratified),
        "resample": args.resample,
        "resample_ratio": float(args.resample_ratio),
        # Normalization: store config and the raw flags for transparency
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
        "normalization_config": norm_config,
        "_tracker": tracker,
    }

    # Log parameters to tracker
    try:
        tracker.log_params(args_dict)
    except Exception:
        pass

    train_and_eval(
        train_rows,
        val_rows,
        test_rows,
        output_dir,
        args.ngram_min,
        args.ngram_max,
        args.min_df,
        class_weight_param,
        args.max_iter,
        args.calibrate,
        int(args.calibrate_cv_folds),
        args.seed,
        args_dict,
        args.resample,
        float(args.resample_ratio),
    )


if __name__ == "__main__":
    main()
