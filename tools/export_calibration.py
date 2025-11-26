#!/usr/bin/env python3
"""
Export calibration metrics and artifacts (ECE, Brier score, reliability diagram) for a trained model.

Currently supports transformer models trained with modeling/train_transformer.py.

Outputs in --output_dir:
- calibration_summary.json: { ece, brier, n_samples, n_classes, n_bins }
- reliability_bins.json and reliability_bins.csv
- reliability_diagram.png

Examples:
  # Use test split next to finalized CSV (when --use_splits was used for training)
  python tools/export_calibration.py \
    --model_dir runs/xlmr_base \
    --data_csv annotation/sample_data/final_test.csv \
    --output_dir runs/xlmr_base_calib

  # Or specify an arbitrary CSV with id,text,label columns
  python tools/export_calibration.py \
    --model_dir runs/xlmr_base \
    --data_csv data/my_eval.csv \
    --output_dir runs/xlmr_base_calib

Requirements:
  pip install transformers torch pandas numpy matplotlib
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np  # type: ignore
except Exception as e:
    raise SystemExit("Install NumPy: pip install numpy\n" + str(e))

try:
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
except Exception as e:
    raise SystemExit("Install transformers+torch: pip install transformers torch\n" + str(e))

# Local imports (repository)
from modeling.text_normalization import load_norm_config, normalize_text
from modeling.utils.device import get_device
from modeling.calibration_utils import compute_calibration_summary


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(path, encoding="utf-8")
        if not {"id", "text", "label"}.issubset(df.columns):
            raise ValueError(f"CSV must have id,text,label columns. Found: {list(df.columns)}")
        df["label"] = df["label"].astype(str).str.upper()
        return df.to_dict(orient="records")
    except Exception:
        rows: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not {"id", "text", "label"}.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"CSV must have id,text,label columns. Found: {reader.fieldnames}")
            for r in reader:
                r["label"] = str(r.get("label", "")).upper()
                rows.append(r)
        return rows


def load_transformer(model_dir: Path, device_pref: str = "auto", cuda_device: int | None = None):
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    dev = get_device(device_pref, cuda_device).device
    mdl.to(dev)
    mdl.eval()
    # Build label mapping from model config if available
    id2label = getattr(mdl.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        try:
            id2label = {int(k): v for k, v in id2label.items()}
        except Exception:
            pass
        labels = [id2label[i] for i in sorted(id2label.keys())]
    else:
        labels = ["POS", "NEG", "NEU"]
    label2id = {l: i for i, l in enumerate(labels)}
    return tok, mdl, dev, labels, label2id


def collect_logits_labels(tok, mdl, device, rows: List[Dict[str, str]], norm_cfg: Dict | None, label2id: Dict[str, int], max_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    texts = []
    y_idx = []
    for r in rows:
        t = r.get("text", "")
        if norm_cfg:
            t = normalize_text(t, norm_cfg)
        y = str(r.get("label", "")).upper()
        if y not in label2id:
            # skip unknown labels
            continue
        texts.append(t)
        y_idx.append(label2id[y])
    if not texts:
        raise SystemExit("No valid rows found with known labels")

    # Batch tokenize and forward pass in chunks
    batch_size = 64
    logits_list: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            logits = out.logits.detach().cpu().numpy()
            logits_list.append(logits)
    logits_all = np.concatenate(logits_list, axis=0)
    y_all = np.asarray(y_idx, dtype=int)
    if logits_all.shape[0] != y_all.shape[0]:
        raise RuntimeError("Mismatch between logits and labels lengths")
    return logits_all, y_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Export calibration metrics/plots for a trained transformer model")
    ap.add_argument("--model_dir", required=True, help="Path to trained model directory (HF format)")
    ap.add_argument("--data_csv", required=True, help="CSV with id,text,label columns (e.g., final_test.csv)")
    ap.add_argument("--output_dir", required=True, help="Directory to write calibration artifacts")
    ap.add_argument("--n_bins", type=int, default=15, help="Number of confidence bins for ECE/diagram")
    ap.add_argument("--max_length", type=int, default=256, help="Max sequence length for tokenization")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device preference")
    ap.add_argument("--cuda_device", type=int, default=None, help="CUDA device index when using GPU")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    data_csv = Path(args.data_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model/tokenizer and normalization config
    tok, mdl, device, labels, label2id = load_transformer(model_dir, args.device, args.cuda_device)
    norm_cfg = load_norm_config(model_dir)

    rows = read_csv_rows(data_csv)
    logits, y_idx = collect_logits_labels(tok, mdl, device, rows, norm_cfg, label2id, max_length=int(args.max_length))

    # Compute calibration summary and save artifacts
    diagram_png = str(out_dir / "reliability_diagram.png")
    bins_json = str(out_dir / "reliability_bins.json")
    bins_csv = str(out_dir / "reliability_bins.csv")

    summary = compute_calibration_summary(
        logits=logits,
        y_idx=y_idx,
        n_bins=int(args.n_bins),
        diagram_png=diagram_png,
        bins_json=bins_json,
        bins_csv=bins_csv,
        title="Reliability Diagram",
    )

    with (out_dir / "calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {diagram_png}, {bins_json}, {bins_csv}")


if __name__ == "__main__":
    main()
