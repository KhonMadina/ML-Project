#!/usr/bin/env python3
"""
Stress test evaluation for Khmer sentiment models by category.

Given a CSV with columns: id,text,label,category
- Loads vectorizer.pkl and model.pkl from a model directory.
- Applies saved normalization (normalization.json) if available.
- Computes overall metrics and per-category metrics (accuracy, macro-F1, confusion, counts).
- Outputs JSON and human-readable TXT summary; optional per-category misclassification CSV.

Examples:
  python modeling/stress_eval.py \
    --model_dir models/baseline_chargram_norm \
    --input_csv annotation/sample_data/stress_test.csv \
    --output_dir reports/stress_eval_baseline

Dependencies:
  pip install scikit-learn joblib pandas
  Optional: matplotlib (future plots)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import joblib
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    import numpy as np  # type: ignore
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install scikit-learn joblib pandas\n"
        f"Underlying import error: {e}"
    )

from .text_normalization import load_norm_config, normalize_corpus

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
    return vectorizer, model, norm_cfg


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if not {"id", "text", "label", "category"}.issubset(df.columns):
            raise ValueError(f"CSV must have id,text,label,category columns. Found: {list(df.columns)}")
        # Normalize label to uppercase
        df["label"] = df["label"].astype(str).str.upper()
        return df.to_dict(orient="records")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "text", "label", "category"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must have id,text,label,category columns. Found: {reader.fieldnames}")
        for r in reader:
            r["label"] = str(r.get("label", "")).upper()
            rows.append(r)
    return rows


def compute_metrics(gold: List[str], pred: List[str]) -> Tuple[float, float, Dict, List[List[int]]]:
    acc = float(accuracy_score(gold, pred))
    f1m = float(f1_score(gold, pred, average="macro"))
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=LABELS).tolist()
    return acc, f1m, report, cm


def write_csv(path: Path, rows: List[Dict[str, str | float]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text", "label", "category", "pred_label"])  # minimal header
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress test evaluation by category for Khmer sentiment models")
    ap.add_argument("--model_dir", required=True, help="Directory containing vectorizer.pkl and model.pkl")
    ap.add_argument("--input_csv", required=True, help="CSV with columns: id,text,label,category")
    ap.add_argument("--output_dir", required=True, help="Where to write reports")
    ap.add_argument("--save_misclassified", action="store_true", help="Save per-category misclassified rows CSV")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    vec, model, norm_cfg = load_model(model_dir)
    rows = read_csv_rows(input_csv)

    texts = [str(r["text"]) for r in rows]
    if norm_cfg:
        texts = normalize_corpus(texts, norm_cfg)
    gold = [str(r["label"]).upper() for r in rows]
    cats = [str(r["category"]) for r in rows]

    X = vec.transform(texts)
    preds = model.predict(X)

    # Overall metrics
    overall_acc, overall_f1, overall_report, overall_cm = compute_metrics(gold, preds.tolist())

    # Per-category metrics
    per_cat: Dict[str, Dict] = {}
    mis_by_cat: Dict[str, List[Dict[str, str | float]]] = {}

    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        g = [gold[i] for i in idx]
        p = [preds[i] for i in idx]
        acc, f1m, rpt, cm = compute_metrics(g, p)
        per_cat[cat] = {
            "count": float(len(idx)),
            "accuracy": acc,
            "f1_macro": f1m,
            "report": rpt,
            "confusion": cm,
        }
        if args.save_misclassified:
            mis = []
            for i in idx:
                if gold[i] != preds[i]:
                    rec = {
                        "id": rows[i].get("id", ""),
                        "text": rows[i].get("text", ""),
                        "label": gold[i],
                        "category": cats[i],
                        "pred_label": preds[i],
                    }
                    mis.append(rec)
            mis_by_cat[cat] = mis

    # Save metrics JSON
    metrics = {
        "label_order": LABELS,
        "overall": {
            "count": len(rows),
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "report": overall_report,
            "confusion": overall_cm,
        },
        "per_category": per_cat,
        "env": {
            "python": sys.version.replace("\n", " "),
        },
        "args": {
            "model_dir": str(model_dir),
            "input_csv": str(input_csv),
            "output_dir": str(out_dir),
            "save_misclassified": bool(args.save_misclassified),
        },
    }
    with (out_dir / "stress_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save summary TXT
    lines = []
    lines.append(f"Total: {len(rows)}")
    lines.append(f"Overall: acc={overall_acc:.4f} f1_macro={overall_f1:.4f}")
    lines.append("")
    lines.append("Per-category metrics:")
    for cat, m in per_cat.items():
        lines.append(f"  {cat}: n={int(m['count'])} acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")
    with (out_dir / "stress_summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Optionally write misclassified per category
    if args.save_misclassified:
        mis_dir = out_dir / "misclassified"
        for cat, rows_mis in mis_by_cat.items():
            safe = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in cat])
            write_csv(mis_dir / f"mis_{safe}.csv", rows_mis)

    print(f"Wrote stress metrics: {out_dir / 'stress_metrics.json'}")
    print(f"Wrote stress summary: {out_dir / 'stress_summary.txt'}")
    if args.save_misclassified:
        print(f"Wrote per-category misclassified CSVs under: {out_dir / 'misclassified'}")


if __name__ == "__main__":
    main()
