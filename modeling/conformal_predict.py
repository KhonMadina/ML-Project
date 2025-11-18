#!/usr/bin/env python3
"""
Conformal prediction for multi-class sentiment models (set-valued predictions).

Implements Thresholded Probability Sets (TPS):
- Calibrate a per-class probability threshold tau on a calibration set to achieve target coverage (1 - alpha).
- On test inputs, include labels whose predicted probability >= tau.

Usage:
  # Using an explicit calibration split file
  python modeling/conformal_predict.py \
    --model_dir models/baseline_chargram_norm \
    --calib_csv annotation/sample_data/final_val.csv \
    --input_csv annotation/sample_data/final_test.csv \
    --output_csv reports/conformal_test_preds.csv \
    --target_coverage 0.9

  # If no calib_csv is provided, split the input_csv into calib/test by ratio
  python modeling/conformal_predict.py \
    --model_dir models/baseline_chargram_norm \
    --input_csv annotation/sample_data/final_test.csv \
    --output_csv reports/conformal_preds.csv \
    --target_coverage 0.9 \
    --split_for_calib 0.3 --seed 42

Outputs:
- Predictions CSV with columns: id,text[,label],pred_set,size,proba_POS,proba_NEG,proba_NEU
- conformal_info.json capturing tau and empirical coverage on calibration and test sets

Requirements:
  pip install scikit-learn joblib pandas numpy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import numpy as np  # type: ignore
    import joblib
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install scikit-learn joblib pandas numpy\n"
        f"Underlying import error: {e}"
    )

from .text_normalization import load_norm_config, normalize_text

LABELS = ["POS", "NEG", "NEU"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_model(model_dir: Path):
    vec_p = model_dir / "vectorizer.pkl"
    mdl_p = model_dir / "model.pkl"
    if not vec_p.exists() or not mdl_p.exists():
        raise SystemExit(f"vectorizer.pkl or model.pkl not found in {model_dir}")
    vectorizer = joblib.load(vec_p)
    model = joblib.load(mdl_p)
    norm_cfg = load_norm_config(model_dir)
    if not hasattr(model, "predict_proba"):
        raise SystemExit("Model does not expose predict_proba; conformal prediction requires class probabilities.")
    return vectorizer, model, norm_cfg


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if not {"id", "text"}.issubset(df.columns):
            raise ValueError(f"CSV must have at least id,text columns. Found: {list(df.columns)}")
        # Normalize label column if present
        if "label" in df.columns:
            df["label"] = df["label"].astype(str).str.upper()
        return df.to_dict(orient="records")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "text"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must have at least id,text columns. Found: {reader.fieldnames}")
        for r in reader:
            if "label" in r and r["label"]:
                r["label"] = str(r["label"]).upper()
            rows.append(r)
    return rows


def write_predictions(path: Path, rows: List[Dict[str, str | float]]):
    ensure_dir(path.parent)
    # Determine fields
    fields = ["id", "text"]
    if rows and "label" in rows[0]:
        fields.append("label")
    fields += ["pred_set", "size"] + [f"proba_{l}" for l in LABELS]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def predict_proba(vectorizer, model, texts: List[str]) -> np.ndarray:
    X = vectorizer.transform(texts)
    proba = model.predict_proba(X)
    return np.asarray(proba, dtype=float)


def calibrate_tau(calib_gold: List[str], calib_proba: np.ndarray, target_coverage: float) -> float:
    """TPS quantile threshold: tau = quantile of (1 - p_true) at level ceil((n+1)*(1-alpha))/n.
    Coverage = 1 - alpha.
    """
    if len(calib_gold) != calib_proba.shape[0]:
        raise ValueError("Calibration labels and probabilities length mismatch")
    # Map gold labels to indices
    classes = LABELS
    c2i = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([c2i.get(g, -1) for g in calib_gold], dtype=int)
    if np.any(y_idx < 0):
        raise ValueError("Unknown labels encountered in calibration set")
    p_true = calib_proba[np.arange(len(y_idx)), y_idx]
    scores = 1.0 - p_true
    alpha = 1.0 - float(target_coverage)
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    tau = float(np.quantile(scores, min(max(q, 0.0), 1.0), interpolation="higher"))
    return tau


def build_pred_set_row(row: Dict[str, str], probs: np.ndarray, tau: float) -> Dict[str, str | float]:
    pred_set_labels = [LABELS[i] for i, p in enumerate(probs) if (1.0 - p) <= tau]
    rec: Dict[str, str | float] = {
        "id": row.get("id", ""),
        "text": row.get("text", ""),
        "pred_set": ",".join(pred_set_labels),
        "size": float(len(pred_set_labels)),
    }
    if "label" in row and row["label"]:
        rec["label"] = row["label"]
    for i, l in enumerate(LABELS):
        rec[f"proba_{l}"] = float(probs[i])
    return rec


def empirical_coverage(rows: List[Dict[str, str | float]]) -> float:
    has_label = [r for r in rows if r.get("label", "")]
    if not has_label:
        return 0.0
    ok = 0
    for r in has_label:
        pred_set = str(r.get("pred_set", "")).split(",") if r.get("pred_set", "") else []
        if str(r.get("label", "")) in pred_set:
            ok += 1
    return float(ok / len(has_label))


def main() -> None:
    ap = argparse.ArgumentParser(description="Conformal prediction (TPS) for multi-class Khmer sentiment models")
    ap.add_argument("--model_dir", required=True, help="Directory with vectorizer.pkl and model.pkl")
    ap.add_argument("--input_csv", required=True, help="CSV with id,text[,label] for prediction/evaluation")
    ap.add_argument("--output_csv", required=True, help="Where to save predictions with prediction sets")
    ap.add_argument("--calib_csv", help="Optional CSV with id,text,label to use for calibration")
    ap.add_argument("--target_coverage", type=float, default=0.9, help="Desired coverage (e.g., 0.9)")
    ap.add_argument("--split_for_calib", type=float, default=0.0, help="If no calib_csv, split input as calib ratio (0..0.9)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_csv = Path(args.output_csv)
    ensure_dir(out_csv.parent)

    vectorizer, model, norm_cfg = load_model(model_dir)

    # Load input and optional calib
    rows_input = read_csv_rows(Path(args.input_csv))
    if args.calib_csv:
        rows_calib = read_csv_rows(Path(args.calib_csv))
    else:
        # Split input for calibration if requested
        if float(args.split_for_calib) > 0.0:
            import random
            rnd = random.Random(args.seed)
            rows = rows_input.copy()
            rnd.shuffle(rows)
            n_cal = int(len(rows) * float(args.split_for_calib))
            rows_calib = rows[:n_cal]
            rows_input = rows[n_cal:]
        else:
            rows_calib = []

    # Normalize texts
    def norm_rows(rows: List[Dict[str, str]]):
        if not norm_cfg:
            return rows
        for r in rows:
            if "text" in r and r["text"] is not None:
                r["text"] = normalize_text(str(r["text"]), norm_cfg)
        return rows

    rows_input = norm_rows(rows_input)
    rows_calib = norm_rows(rows_calib)

    # Calibration
    if rows_calib:
        if not all(r.get("label", "") for r in rows_calib):
            raise SystemExit("Calibration CSV must include gold labels.")
        calib_texts = [r.get("text", "") for r in rows_calib]
        calib_proba = predict_proba(vectorizer, model, calib_texts)
        calib_gold = [str(r.get("label", "")).upper() for r in rows_calib]
        tau = calibrate_tau(calib_gold, calib_proba, float(args.target_coverage))
    else:
        # Fallback tau from input (only if labels present); else use 0.5 as default
        if all(r.get("label", "") for r in rows_input):
            proba = predict_proba(vectorizer, model, [r.get("text", "") for r in rows_input])
            tau = calibrate_tau([str(r.get("label", "")).upper() for r in rows_input], proba, float(args.target_coverage))
        else:
            tau = 0.5
            print("Warning: no calibration set provided; using tau=0.5 default.")

    # Predictions with sets on input
    texts = [r.get("text", "") for r in rows_input]
    proba_input = predict_proba(vectorizer, model, texts)
    pred_rows: List[Dict[str, str | float]] = []
    for i, r in enumerate(rows_input):
        pred_rows.append(build_pred_set_row(r, proba_input[i], tau))

    write_predictions(out_csv, pred_rows)

    # Save conformal info
    info = {
        "target_coverage": float(args.target_coverage),
        "tau": float(tau),
        "labels": LABELS,
        "counts": {"input": len(rows_input), "calib": len(rows_calib)},
        "coverage": {
            "input": empirical_coverage(pred_rows),
        },
        "args": {
            "model_dir": str(model_dir),
            "input_csv": str(Path(args.input_csv)),
            "output_csv": str(out_csv),
            "calib_csv": str(Path(args.calib_csv)) if args.calib_csv else None,
            "split_for_calib": float(args.split_for_calib),
            "seed": int(args.seed),
        }
    }
    info_path = out_csv.parent / "conformal_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Wrote conformal predictions to {out_csv}")
    print(f"Wrote conformal info to {info_path}")


if __name__ == "__main__":
    main()
