#!/usr/bin/env python3
"""
Paired bootstrap confidence intervals for classification comparisons.

Given two prediction CSVs with columns id,label,pred_label for the SAME examples,
compute macro-F1 and accuracy for each model and paired bootstrap CIs for the delta
(model_b - model_a).

Outputs in --output:
- bootstrap_summary.json: per-model metrics, CIs, and delta CIs

Example usage:
  python tools/paired_bootstrap_ci.py \
    --predictions_a runs/baseline_best/pred_test.csv \
    --predictions_b runs/xlmr_best/pred_test.csv \
    --output runs/paired_bootstrap_baseline_vs_xlmr \
    --n_bootstrap 2000 --alpha 0.05 --seed 123

Input format:
  CSV with columns: id,label,pred_label (case-insensitive). Additional columns are ignored.

Requirements:
  pip install numpy pandas scikit-learn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np  # type: ignore
except Exception as e:
    raise SystemExit("Install NumPy: pip install numpy\n" + str(e))

try:
    import pandas as pd  # type: ignore
except Exception as e:
    raise SystemExit("Install pandas: pip install pandas\n" + str(e))

try:
    from sklearn.metrics import f1_score, accuracy_score  # type: ignore
except Exception as e:
    raise SystemExit("Install scikit-learn: pip install scikit-learn\n" + str(e))


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    cols = {c.lower(): c for c in df.columns}
    required = ["id", "label", "pred_label"]
    for r in required:
        if r not in cols:
            raise SystemExit(f"{path}: missing required column '{r}' (case-insensitive)")
    # Normalize columns
    df = df.rename(columns={cols["id"]: "id", cols["label"]: "label", cols["pred_label"]: "pred_label"})
    df["label"] = df["label"].astype(str).str.upper()
    df["pred_label"] = df["pred_label"].astype(str).str.upper()
    return df[["id", "label", "pred_label"]]


def align_by_id(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Inner join on id to ensure same order and set
    merged = df_a.merge(df_b, on="id", suffixes=("_a", "_b"))
    if len(merged) == 0:
        raise SystemExit("No overlapping ids between the two prediction files")
    # Rebuild two aligned dataframes from merged
    a = merged[["id", "label_a", "pred_label_a"]].rename(columns={"label_a": "label", "pred_label_a": "pred_label"})
    b = merged[["id", "label_b", "pred_label_b"]].rename(columns={"label_b": "label", "pred_label_b": "pred_label"})
    return a.reset_index(drop=True), b.reset_index(drop=True)


def macro_f1_and_acc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    return float(f1_score(y_true, y_pred, average="macro")), float(accuracy_score(y_true, y_pred))


def percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_bootstrap(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, n_bootstrap: int, alpha: float, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)

    # Base metrics on full data
    f1_a, acc_a = macro_f1_and_acc(y_true, y_pred_a)
    f1_b, acc_b = macro_f1_and_acc(y_true, y_pred_b)

    f1_a_s = np.empty(n_bootstrap, dtype=float)
    f1_b_s = np.empty(n_bootstrap, dtype=float)
    acc_a_s = np.empty(n_bootstrap, dtype=float)
    acc_b_s = np.empty(n_bootstrap, dtype=float)

    n = len(y_true)
    idx = np.arange(n)
    for i in range(n_bootstrap):
        bs_idx = rng.integers(0, n, size=n)
        yt = y_true[bs_idx]
        ya = y_pred_a[bs_idx]
        yb = y_pred_b[bs_idx]
        f1_a_s[i], acc_a_s[i] = macro_f1_and_acc(yt, ya)
        f1_b_s[i], acc_b_s[i] = macro_f1_and_acc(yt, yb)

    # Deltas (B - A)
    f1_d = f1_b_s - f1_a_s
    acc_d = acc_b_s - acc_a_s

    f1_a_ci = percentile_ci(f1_a_s, alpha)
    f1_b_ci = percentile_ci(f1_b_s, alpha)
    acc_a_ci = percentile_ci(acc_a_s, alpha)
    acc_b_ci = percentile_ci(acc_b_s, alpha)
    f1_d_ci = percentile_ci(f1_d, alpha)
    acc_d_ci = percentile_ci(acc_d, alpha)

    return {
        "n": int(n),
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
        "seed": int(seed),
        "metrics": {
            "macro_f1": {
                "model_a": {"point": f1_a, "ci": {"low": f1_a_ci[0], "high": f1_a_ci[1]}},
                "model_b": {"point": f1_b, "ci": {"low": f1_b_ci[0], "high": f1_b_ci[1]}},
                "delta_b_minus_a": {"point": float(f1_b - f1_a), "ci": {"low": f1_d_ci[0], "high": f1_d_ci[1]}},
            },
            "accuracy": {
                "model_a": {"point": acc_a, "ci": {"low": acc_a_ci[0], "high": acc_a_ci[1]}},
                "model_b": {"point": acc_b, "ci": {"low": acc_b_ci[0], "high": acc_b_ci[1]}},
                "delta_b_minus_a": {"point": float(acc_b - acc_a), "ci": {"low": acc_d_ci[0], "high": acc_d_ci[1]}},
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap CIs for two prediction files")
    ap.add_argument("--predictions_a", required=True, help="CSV with id,label,pred_label for model A")
    ap.add_argument("--predictions_b", required=True, help="CSV with id,label,pred_label for model B")
    ap.add_argument("--output", required=True, help="Directory to write bootstrap_summary.json")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df_a = load_predictions(Path(args.predictions_a))
    df_b = load_predictions(Path(args.predictions_b))
    a, b = align_by_id(df_a, df_b)

    # Verify labels match between A and B rows
    if not (a["label"].values == b["label"].values).all():
        raise SystemExit("Mismatched labels across the two files for some ids. Ensure both files use the same gold labels.")

    # Convert to arrays
    y_true = a["label"].values
    y_pred_a = a["pred_label"].values
    y_pred_b = b["pred_label"].values

    summary = paired_bootstrap(y_true, y_pred_a, y_pred_b, int(args.n_bootstrap), float(args.alpha), int(args.seed))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "bootstrap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir / 'bootstrap_summary.json'}")


if __name__ == "__main__":
    main()
