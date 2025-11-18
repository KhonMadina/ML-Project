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

# Experiment utilities for reproducibility and tracking
from .utils.experiment import Config as ExpConfig, prepare_experiment

# Local normalization utilities
from .text_normalization import (
    build_norm_config_from_args,
    save_norm_config,
    normalize_corpus,
)
from .calibration_utils import TemperatureScaledModel

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


@dataclass
class Row:
    id: str
    text: str
    label: str
    group: Optional[str] = None


def read_csv_rows(path: Path, group_col: Optional[str] = None) -> List[Row]:
    """Read rows from CSV, optionally extracting a group column for group-aware splits."""
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        required = {"id", "text", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns {required}. Got {list(df.columns)}")
        rows: List[Row] = []
        for _, r in df.iterrows():
            group_val = None
            if group_col and group_col in df.columns:
                gv = r[group_col]
                group_val = None if (gv is None or (isinstance(gv, float) and pd.isna(gv))) else str(gv)
            rows.append(Row(str(r["id"]), str(r["text"]), str(r["label"]).upper(), group_val))
        return rows
    # Fallback to csv module
    rows2: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "text", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns {required}. Got {reader.fieldnames}")
        for r in reader:
            group_val = None
            if group_col and group_col in (reader.fieldnames or []):
                gv = r.get(group_col)
                group_val = str(gv) if gv is not None and str(gv).strip() != "" else None
            rows2.append(Row(str(r["id"]), str(r["text"]), str(r["label"]).upper(), group_val))
    return rows2


def stratified_split(rows: List[Row], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[List[Row], List[Row], List[Row]]:
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    random.seed(seed)
    by_label: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_label[r.label].append(r)
    train: List[Row] = []
    val: List[Row] = []
    test: List[Row] = []
    for lbl, items in by_label.items():
        items = items.copy()
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return train, val, test


def group_aware_split(rows: List[Row], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[List[Row], List[Row], List[Row]]:
    """Split by groups using GroupShuffleSplit to avoid leakage. Not strictly stratified.
    Uses a two-stage split: train vs temp, then temp -> val/test.
    """
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if np is None:
        raise SystemExit("NumPy is required for group-aware split. Install with: pip install numpy")

    X = np.arange(len(rows))
    y = np.array([r.label for r in rows])  # not used by GroupShuffleSplit, but kept for clarity
    groups = np.array([r.group if r.group is not None else f"__nogroup_{i}" for i, r in enumerate(rows)])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    # Compute proportion of test within temp
    temp_size = val_ratio + test_ratio
    test_within_temp = (test_ratio / temp_size) if temp_size > 0 else 0.0

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_within_temp, random_state=seed + 1)
    val_idx, test_idx = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))

    # Map indices back
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]
    return train_rows, val_rows, test_rows


def group_stratified_split(rows: List[Row], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[List[Row], List[Row], List[Row]]:
    """Approximate group-aware stratified split.
    Assign entire groups to splits to avoid leakage while approximating label distribution targets.
    Heuristic: greedy assignment minimizing L1 distance to target label counts.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Collect groups and per-group label counts
    groups: Dict[str, List[Row]] = {}
    for i, r in enumerate(rows):
        g = r.group if r.group is not None else f"__nogroup_{i}"
        groups.setdefault(g, []).append(r)

    total_label_counts: Dict[str, int] = {}
    for r in rows:
        total_label_counts[r.label] = total_label_counts.get(r.label, 0) + 1

    def target_counts(ratio: float) -> Dict[str, float]:
        return {lbl: total_label_counts.get(lbl, 0) * ratio for lbl in total_label_counts}

    targets = {
        "train": target_counts(train_ratio),
        "val": target_counts(val_ratio),
        "test": target_counts(test_ratio),
    }

    # Current counts per split
    cur = {"train": {lbl: 0 for lbl in total_label_counts},
           "val": {lbl: 0 for lbl in total_label_counts},
           "test": {lbl: 0 for lbl in total_label_counts}}

    # Deterministic order: by group size desc, then name asc, with seeded shuffle for ties
    rng = random.Random(seed)
    items = list(groups.items())
    items.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    assign: Dict[str, str] = {}

    def label_counts_for_group(grows: List[Row]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for r in grows:
            d[r.label] = d.get(r.label, 0) + 1
        return d

    for g, grows in items:
        gc = label_counts_for_group(grows)
        # Evaluate cost for assigning to each split
        best_split = None
        best_cost = None
        for sp in ["train", "val", "test"]:
            cost = 0.0
            for lbl, tgt in targets[sp].items():
                after = cur[sp].get(lbl, 0) + gc.get(lbl, 0)
                cost += abs(after - tgt)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = sp
        assign[g] = best_split or "train"
        # Update counts
        for lbl, c in gc.items():
            cur[assign[g]][lbl] = cur[assign[g]].get(lbl, 0) + c

    train_rows = []
    val_rows = []
    test_rows = []
    for g, grows in groups.items():
        sp = assign[g]
        if sp == "train":
            train_rows.extend(grows)
        elif sp == "val":
            val_rows.extend(grows)
        else:
            test_rows.extend(grows)

    return train_rows, val_rows, test_rows


def to_xy(rows: List[Row]) -> Tuple[List[str], List[str]]:
    X = [r.text for r in rows]
    y = [r.label for r in rows]
    return X, y


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

    X_train, y_train = to_xy(train_rows)
    X_val, y_val = to_xy(val_rows)
    X_test, y_test = to_xy(test_rows)

    # Apply normalization if configured (config is passed via args_dict under 'normalization_config')
    norm_config = args_dict.get("normalization_config") if isinstance(args_dict, dict) else None
    if norm_config:
        X_train = normalize_corpus(X_train, norm_config)  # type: ignore
        X_val = normalize_corpus(X_val, norm_config)      # type: ignore
        X_test = normalize_corpus(X_test, norm_config)    # type: ignore

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        strip_accents=None,
        lowercase=False,
    )
    Xtr = vectorizer.fit_transform(X_train)

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
        multi_class="auto",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(Xtr, y_train)

    # Evaluate on val and test, with optional calibration
    Xv = vectorizer.transform(X_val)
    Xt = vectorizer.transform(X_test)

    model_for_eval = clf
    calibration_info: Dict[str, object] = {"applied": False}
    if calibrate == "temperature":
        try:
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
                    calibrator = CalibratedClassifierCV(base_estimator=clf, method=method, cv="prefit")
                    calibrator.fit(Xv, y_val)
                    model_for_eval = calibrator
                    calibration_info = {"applied": True, "method": calibrate, "cv_folds": 0}
            except Exception as e:
                print(f"Warning: calibration failed: {e}")

    def eval_split(X, y, split_name: str) -> Dict[str, object]:
        yp = model_for_eval.predict(X)
        acc = accuracy_score(y, yp)
        f1m = f1_score(y, yp, average="macro")
        report = classification_report(y, yp, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, yp, labels=["POS", "NEG", "NEU"]).tolist()
        return {"accuracy": acc, "f1_macro": f1m, "report": report, "confusion": cm}

    metrics = {
        "val": eval_split(Xv, y_val, "val"),
        "test": eval_split(Xt, y_test, "test"),
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
        "args": args_dict,
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

    # Normalization flags
    ap.add_argument("--normalize_all", action="store_true", help="Enable a default Khmer/social text normalization pipeline")
    ap.add_argument("--norm_nfc", action="store_true", help="Apply Unicode NFC normalization")
    ap.add_argument("--norm_whitespace", action="store_true", help="Collapse whitespace and trim")
    ap.add_argument("--norm_punct", action="store_true", help="Normalize punctuation variants and compress repeats")
    ap.add_argument("--norm_elongation", action="store_true", help="Compress elongated character runs (>2 -> 2)")
    ap.add_argument("--norm_emoji", choices=["keep", "remove", "map"], default=None, help="Emoji handling mode: keep/remove/map-to-token")

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
