#!/usr/bin/env python3
"""
Inference and evaluation for Khmer sentiment baseline (POS/NEG/NEU).

Features:
- Load vectorizer.pkl and model.pkl from a model directory.
- Predict a single text (CLI) or batch from CSV with columns: id,text[,label]
- If label column is present, compute accuracy and macro-F1, and save a classification report.
- Outputs predictions CSV with probabilities per class.

Examples:
  # Single text prediction
  python modeling/predict.py --model_dir models/baseline_chargram --text "អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄"

  # Batch prediction (no labels) -> writes predictions CSV
  python modeling/predict.py --model_dir models/baseline_chargram \
    --input_csv annotation/sample_data/final_dataset.csv --output_csv annotation/sample_data/predictions.csv

  # Batch evaluation (labels present)
  python modeling/predict.py --model_dir models/baseline_chargram \
    --input_csv annotation/sample_data/final_test.csv --output_csv annotation/sample_data/pred_test.csv

Dependencies:
  pip install scikit-learn joblib pandas
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, classification_report
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install scikit-learn joblib pandas\n"
        f"Underlying import error: {e}"
    )

LABELS = ["POS", "NEG", "NEU"]


def load_model(model_dir: Path):
    vec_p = model_dir / "vectorizer.pkl"
    mdl_p = model_dir / "model.pkl"
    if not vec_p.exists() or not mdl_p.exists():
        raise SystemExit(f"vectorizer.pkl or model.pkl not found in {model_dir}")
    vectorizer = joblib.load(vec_p)
    model = joblib.load(mdl_p)
    return vectorizer, model


def predict_text(vectorizer, model, text: str) -> Dict[str, float]:
    X = vectorizer.transform([text])
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    if probs is None:
        return {"label": pred}
    return {"label": pred, **{f"proba_{LABELS[i]}": float(probs[i]) for i in range(len(LABELS))}}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if not {"id", "text"}.issubset(df.columns):
            raise ValueError(f"CSV must have at least id,text columns. Found: {list(df.columns)}")
        return df.to_dict(orient="records")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "text"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must have at least id,text columns. Found: {reader.fieldnames}")
        for r in reader:
            rows.append(r)
    return rows


def write_predictions(path: Path, rows: List[Dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Determine fieldnames
    fields = ["id", "text", "pred_label"] + [f"proba_{l}" for l in LABELS]
    if rows and "label" in rows[0]:
        fields.insert(2, "label")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def batch_predict(vectorizer, model, rows: List[Dict[str, str]]) -> List[Dict[str, str | float]]:
    texts = [r.get("text", "") for r in rows]
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    out: List[Dict[str, str | float]] = []
    for i, r in enumerate(rows):
        rec: Dict[str, str | float] = {
            "id": r.get("id", ""),
            "text": r.get("text", ""),
            "pred_label": preds[i],
        }
        if "label" in r and r["label"]:
            rec["label"] = str(r["label"]).upper()
        if probs is not None:
            for j, lbl in enumerate(LABELS):
                rec[f"proba_{lbl}"] = float(probs[i][j])
        out.append(rec)
    return out


def maybe_evaluate(rows_with_preds: List[Dict[str, str | float]]) -> Dict:
    # Only evaluate if gold labels are present
    gold = []
    pred = []
    for r in rows_with_preds:
        if "label" in r and r["label"]:
            gold.append(r["label"])  # type: ignore
            pred.append(r["pred_label"])  # type: ignore
    if not gold:
        return {}
    acc = accuracy_score(gold, pred)
    f1m = f1_score(gold, pred, average="macro")
    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    return {"accuracy": acc, "f1_macro": f1m, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict or evaluate using a trained Khmer sentiment baseline")
    ap.add_argument("--model_dir", required=True, help="Directory containing vectorizer.pkl and model.pkl")
    ap.add_argument("--text", help="Single input text to classify")
    ap.add_argument("--input_csv", help="CSV with columns: id,text[,label]")
    ap.add_argument("--output_csv", help="Where to write predictions CSV for batch mode")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    vectorizer, model = load_model(model_dir)

    if args.text:
        res = predict_text(vectorizer, model, args.text)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.input_csv:
        rows = read_csv_rows(Path(args.input_csv))
        pred_rows = batch_predict(vectorizer, model, rows)
        metrics = maybe_evaluate(pred_rows)
        if args.output_csv:
            write_predictions(Path(args.output_csv), pred_rows)
            print(f"Wrote predictions to {args.output_csv}")
        if metrics:
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        else:
            print("Predictions completed (no gold labels found for evaluation).")
        return

    raise SystemExit("Provide either --text for single prediction or --input_csv for batch prediction.")


if __name__ == "__main__":
    main()
