#!/usr/bin/env python3
"""
Predict/evaluate using a fine-tuned transformer model (HuggingFace) for Khmer sentiment.

Features:
- Loads model and tokenizer from --model_dir
- Applies saved normalization (normalization.json) if present
- Single-text or batch CSV mode (id,text[,label])
- Outputs predictions CSV with probabilities per class
- If labels provided, prints accuracy and macro-F1

Examples:
  # Single text
  python modeling/predict_transformer.py --model_dir models/xlmr_base --text "អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄"

  # Batch prediction
  python modeling/predict_transformer.py --model_dir models/xlmr_base \
    --input_csv annotation/sample_data/final_test.csv \
    --output_csv annotation/sample_data/pred_test_xlmr.csv

Requirements:
  pip install transformers torch pandas scikit-learn
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import sys

try:
    import torch # type: ignore
    from transformers import AutoTokenizer, AutoModelForSequenceClassification # type: ignore
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install transformers torch\n"
        f"Underlying import error: {e}"
    )

from .utils.device import get_device

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
except Exception:
    accuracy_score = None
    f1_score = None
    classification_report = None

from .text_normalization import load_norm_config, normalize_text


def load_model(model_dir: Path, prefer_device: str = "auto", cuda_device: int | None = None):
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    except Exception as e:
        raise SystemExit(f"Failed to load model/tokenizer from {model_dir}: {e}")
    norm_cfg = load_norm_config(model_dir)
    dev_ctx = get_device(prefer_device, cuda_device)
    device = dev_ctx.device
    model.to(device)
    model.eval()
    # Label mapping
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        # Keys may be str indices in safetensors; convert to int ordering
        try:
            id2label = {int(k): v for k, v in id2label.items()}
        except Exception:
            pass
        labels = [id2label[i] for i in sorted(id2label.keys())]
    else:
        labels = ["POS", "NEG", "NEU"]
    return tokenizer, model, norm_cfg, labels, device


def predict_text(tokenizer, model, device, labels: List[str], text: str) -> Dict[str, float]:
    enc = tokenizer([text], truncation=True, padding=False, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        pred_idx = int(probs.argmax())
    res = {"label": labels[pred_idx]}
    for i, l in enumerate(labels):
        res[f"proba_{l}"] = float(probs[i])
    return res


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if not {"id", "text"}.issubset(df.columns):
            raise ValueError(f"CSV must have at least id,text columns. Found: {list(df.columns)}")
        # normalize label
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


def write_predictions(path: Path, rows: List[Dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text", "label", "pred_label"])  # header
        return
    # Determine fieldnames from labels in first row
    fields = ["id", "text", "pred_label"]
    if rows and "label" in rows[0]:
        fields.insert(2, "label")
    # Find probability columns
    proba_cols = [c for c in rows[0].keys() if c.startswith("proba_")]
    fields.extend(sorted(proba_cols))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def batch_predict(tokenizer, model, device, labels: List[str], rows: List[Dict[str, str]]) -> List[Dict[str, str | float]]:
    texts = [r.get("text", "") for r in rows]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out_rows: List[Dict[str, str | float]] = []
    with torch.no_grad():
        for i in range(0, enc["input_ids"].shape[0], 64):
            batch = {k: v[i:i+64] for k, v in enc.items()}
            out = model(**batch)
            probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
            for j in range(probs.shape[0]):
                idx = i + j
                rec: Dict[str, str | float] = {
                    "id": rows[idx].get("id", ""),
                    "text": rows[idx].get("text", ""),
                    "pred_label": labels[int(probs[j].argmax())],
                }
                if "label" in rows[idx] and rows[idx]["label"]:
                    rec["label"] = str(rows[idx]["label"]).upper()
                for k, l in enumerate(labels):
                    rec[f"proba_{l}"] = float(probs[j][k])
                out_rows.append(rec)
    return out_rows


def maybe_evaluate(rows_with_preds: List[Dict[str, str | float]]) -> Dict:
    if accuracy_score is None or f1_score is None or classification_report is None:
        return {}
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
    ap = argparse.ArgumentParser(description="Predict/evaluate using a transformer Khmer sentiment model")
    ap.add_argument("--model_dir", required=True, help="Directory containing HF model and tokenizer")
    ap.add_argument("--text", help="Single input text to classify")
    ap.add_argument("--input_csv", help="CSV with columns: id,text[,label]")
    ap.add_argument("--output_csv", help="Where to write predictions CSV for batch mode")
    # Device args
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device preference: auto picks CUDA if safe, else CPU")
    ap.add_argument("--cuda_device", type=int, default=None, help="CUDA device index when using --device cuda/auto")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer, model, norm_cfg, labels, device = load_model(model_dir, args.device, args.cuda_device)
    print(f"Using device: {device}")

    if args.text:
        t = normalize_text(args.text, norm_cfg)
        res = predict_text(tokenizer, model, device, labels, t)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.input_csv:
        rows = read_csv_rows(Path(args.input_csv))
        if norm_cfg:
            for r in rows:
                if "text" in r and r["text"] is not None:
                    r["text"] = normalize_text(str(r["text"]), norm_cfg)
        pred_rows = batch_predict(tokenizer, model, device, labels, rows)
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
