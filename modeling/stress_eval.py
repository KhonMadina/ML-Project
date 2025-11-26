#!/usr/bin/env python3
"""
Stress test evaluation for sentiment models with perturbation presets and per-category reports.

Supports two evaluation modes:
- Baseline models (scikit-learn): load vectorizer.pkl and model.pkl
- Transformer checkpoints (HF): specify --transformer_model_dir; uses AutoTokenizer+AutoModelForSequenceClassification

Input formats:
- Category CSV: columns id,text,label,category  (evaluates given categories)
- Or plain CSV without category: columns id,text,label (evaluates overall only)

Perturbation presets (apply to input text before inference):
- diacritics: random removal/duplication/reordering of Khmer combining marks
- zero_width: random insertion of ZWSP/ZWJ/ZWNJ
- emoji_burst: inject bursts of emojis
- elongation: extend repeated characters
- mixed_script: insert lookalike Latin/Cyrillic characters in place of Latin segments
- code_switch: inject short phrases from the other language
- latinized_khmer: approximate transliteration by mapping Khmer digits/punct to ASCII and stripping marks

Each preset accepts a --severity in [0,1] and a --prob in [0,1] controlling intensity and application probability.

Examples (baseline):
  python modeling/stress_eval.py \
    --model_dir models/baseline_chargram \
    --input_csv data/stress_tests.csv \
    --output_dir reports/stress_baseline --preset diacritics --severity 0.7 --prob 0.8 --save_misclassified

Examples (transformer):
  python modeling/stress_eval.py \
    --transformer_model_dir runs/xlmr_base \
    --input_csv data/stress_tests.csv \
    --output_dir reports/stress_xlmr --preset code_switch --severity 0.5 --prob 0.5

Dependencies:
  pip install scikit-learn joblib pandas
  pip install transformers datasets torch
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
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

# Optional transformer deps
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    torch = None  # type: ignore

from .text_normalization import load_norm_config, normalize_corpus

LABELS = ["POS", "NEG", "NEU"]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Model loading
# -----------------------------

def load_baseline(model_dir: Path):
    vec_p = model_dir / "vectorizer.pkl"
    mdl_p = model_dir / "model.pkl"
    if not vec_p.exists() or not mdl_p.exists():
        raise SystemExit(f"vectorizer.pkl or model.pkl not found in {model_dir}")
    vectorizer = joblib.load(vec_p)
    model = joblib.load(mdl_p)
    norm_cfg = load_norm_config(model_dir)
    return vectorizer, model, norm_cfg


def load_transformer(model_dir: Path):
    if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
        raise SystemExit("Transformers not available. Install with: pip install transformers datasets torch")
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    norm_cfg = load_norm_config(model_dir)
    return tok, mdl, norm_cfg


# -----------------------------
# Data IO
# -----------------------------

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        cols = set(df.columns)
        if not {"id", "text", "label"}.issubset(cols):
            raise ValueError(f"CSV must have id,text,label columns. Found: {list(df.columns)}")
        if "category" not in cols:
            df["category"] = "overall"
        # Normalize label to uppercase
        df["label"] = df["label"].astype(str).str.upper()
        return df.to_dict(orient="records")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        flds = set(reader.fieldnames or [])
        if not {"id", "text", "label"}.issubset(flds):
            raise ValueError(f"CSV must have id,text,label columns. Found: {reader.fieldnames}")
        has_cat = "category" in flds
        for r in reader:
            r["label"] = str(r.get("label", "")).upper()
            if not has_cat:
                r["category"] = "overall"
            rows.append(r)
    return rows


# -----------------------------
# Preset perturbations
# -----------------------------

_ZW = ["\u200b", "\u200c", "\u200d", "\ufeff"]
_KH_DIAC = re.compile(r"[\u17B4-\u17D3]")
_LATIN = re.compile(r"[A-Za-z]")
_KHMER = re.compile(r"[\u1780-\u17FF]")
_EMOJIS = ["😀","😁","😂","🤣","😅","😊","😍","😡","😭","👍","👎","🔥","✨","💯","❗"]
_LOOKALIKES = {"A":"А","B":"Β","C":"С","E":"Е","H":"Н","K":"Κ","M":"Μ","O":"О","P":"Р","T":"Т","X":"Χ","Y":"Υ","a":"а","e":"е","o":"о","p":"р","c":"с","x":"х","y":"у"}
_CODE_SWITCH_EN = ["ok","wow","haha","lol","great","bad","love","hate"]
_CODE_SWITCH_KM = ["ល្អ","អាក្រក់","សុខចិត្ត","ច្រណែន","ពិតៗ","អូហូ"]


def _apply_prob(p: float) -> bool:
    return random.random() < max(0.0, min(1.0, p))


def p_diacritics(s: str, severity: float, prob: float) -> str:
    if not _apply_prob(prob):
        return s
    sev = max(0.0, min(1.0, severity))
    out = []
    for ch in s:
        if _KH_DIAC.match(ch) and random.random() < sev:
            # drop or duplicate
            if random.random() < 0.5:
                continue
            else:
                out.append(ch)
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def p_zero_width(s: str, severity: float, prob: float) -> str:
    if not _apply_prob(prob):
        return s
    sev = max(0.0, min(1.0, severity))
    out = []
    for ch in s:
        out.append(ch)
        if random.random() < sev * 0.2:
            out.append(random.choice(_ZW))
    return "".join(out)


def p_emoji_burst(s: str, severity: float, prob: float) -> str:
    if not _apply_prob(prob):
        return s
    burst = max(1, int(1 + severity * 5))
    return s + " " + "".join(random.choices(_EMOJIS, k=burst))


def p_elongation(s: str, severity: float, prob: float) -> str:
    if not _apply_prob(prob):
        return s
    sev = max(0.0, min(1.0, severity))
    out = []
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        run = s[i:j]
        if len(run) == 1 and random.random() < sev * 0.6:
            out.append(run * (2 + int(sev * 3)))
        else:
            out.append(run)
        i = j
    return "".join(out)


def p_mixed_script(s: str, severity: float, prob: float) -> str:
    if not _apply_prob(prob):
        return s
    out = []
    for ch in s:
        if ch in _LOOKALIKES and random.random() < severity:
            out.append(_LOOKALIKES[ch])
        else:
            out.append(ch)
    return "".join(out)


def p_code_switch(s: str, severity: float, prob: float) -> str:
    if not _apply_prob(prob):
        return s
    # Guess dominant script and insert a short phrase from the other language
    has_km = bool(_KHMER.search(s))
    phrase = random.choice(_CODE_SWITCH_EN if has_km else _CODE_SWITCH_KM)
    return f"{s} {phrase}" if random.random() < (0.5 + 0.5 * severity) else f"{phrase} {s}"


def p_latinized_khmer(s: str, severity: float, prob: float) -> str:
    # Heuristic: strip diacritics and zero-width; map Khmer digits/punct to ASCII; keep Latin letters
    if not _apply_prob(prob):
        return s
    s2 = p_diacritics(s, severity, 1.0)
    s2 = p_zero_width(s2, 1.0, 1.0)
    # Map Khmer digits to ASCII
    trans = {ord("០"): ord("0"), ord("១"): ord("1"), ord("២"): ord("2"), ord("៣"): ord("3"), ord("៤"): ord("4"),
             ord("៥"): ord("5"), ord("៦"): ord("6"), ord("៧"): ord("7"), ord("៨"): ord("8"), ord("៩"): ord("9")}
    return s2.translate(trans)


PRESETS: Dict[str, Callable[[str, float, float], str]] = {
    "diacritics": p_diacritics,
    "zero_width": p_zero_width,
    "emoji_burst": p_emoji_burst,
    "elongation": p_elongation,
    "mixed_script": p_mixed_script,
    "code_switch": p_code_switch,
    "latinized_khmer": p_latinized_khmer,
}


# -----------------------------
# Metrics and reporting
# -----------------------------

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


# -----------------------------
# Inference helpers
# -----------------------------

def predict_baseline(vectorizer, model, texts: List[str]) -> List[str]:
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    return [str(p) for p in preds]


def predict_transformer(tokenizer, model, texts: List[str], batch_size: int = 32, device: Optional[str] = None) -> List[str]:
    if torch is None:
        raise SystemExit("Transformers not available. Install with: pip install transformers datasets torch")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    out_labels: List[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
        out_labels.extend([LABELS[p] if 0 <= p < len(LABELS) else "NEU" for p in preds])
    return out_labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress test evaluation for sentiment models with presets")
    # Model selection
    ap.add_argument("--model_dir", help="Directory containing baseline vectorizer.pkl and model.pkl")
    ap.add_argument("--transformer_model_dir", help="Directory containing HF checkpoint to load tokenizer+model")
    # Data
    ap.add_argument("--input_csv", required=True, help="CSV with columns: id,text,label[,category]")
    ap.add_argument("--output_dir", required=True, help="Where to write reports")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for transformer inference")
    # Presets
    ap.add_argument("--preset", choices=list(PRESETS.keys()), help="Perturbation preset to apply to input text")
    ap.add_argument("--severity", type=float, default=0.5, help="Severity in [0,1]")
    ap.add_argument("--prob", type=float, default=0.7, help="Probability of applying perturbation to a given example")
    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_misclassified", action="store_true", help="Save per-category misclassified rows CSV")
    args = ap.parse_args()

    random.seed(args.seed)

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    rows = read_csv_rows(input_csv)

    # Choose model type
    mode = None
    vectorizer = model = tok = mdl = norm_cfg = None
    if args.transformer_model_dir:
        mode = "transformer"
        tok, mdl, norm_cfg = load_transformer(Path(args.transformer_model_dir))
    elif args.model_dir:
        mode = "baseline"
        vectorizer, model, norm_cfg = load_baseline(Path(args.model_dir))
    else:
        raise SystemExit("Specify either --model_dir (baseline) or --transformer_model_dir (HF checkpoint)")

    # Prepare texts
    raw_texts = [str(r["text"]) for r in rows]
    if norm_cfg:
        raw_texts = normalize_corpus(raw_texts, norm_cfg)

    # Apply preset perturbation if requested
    if args.preset:
        fn = PRESETS[args.preset]
        texts = [fn(t, args.severity, args.prob) for t in raw_texts]
    else:
        texts = raw_texts

    gold = [str(r["label"]).upper() for r in rows]
    cats = [str(r.get("category", "overall")) for r in rows]

    # Predict
    if mode == "baseline":
        preds = predict_baseline(vectorizer, model, texts)
    else:
        preds = predict_transformer(tok, mdl, texts, batch_size=args.batch_size)

    # Overall metrics
    overall_acc, overall_f1, overall_report, overall_cm = compute_metrics(gold, preds)

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
            "model_dir": str(args.model_dir) if args.model_dir else None,
            "transformer_model_dir": str(args.transformer_model_dir) if args.transformer_model_dir else None,
            "input_csv": str(input_csv),
            "output_dir": str(out_dir),
            "preset": args.preset,
            "severity": float(args.severity),
            "prob": float(args.prob),
            "seed": int(args.seed),
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
