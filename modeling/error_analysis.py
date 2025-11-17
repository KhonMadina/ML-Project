#!/usr/bin/env python3
"""
Enhanced Error analysis for Khmer sentiment baseline (POS/NEG/NEU).

Generates:
- misclassified.csv: All incorrect predictions with confidence and margin
- error_analysis_metrics.json: Metrics, confusion matrix, calibration (ECE/MCE + bins), and optional per-slice metrics
- error_analysis_summary.txt: Human-readable summary
- reliability_diagram.png: Reliability diagram (if matplotlib is available)
- calibration_bins.csv: Calibration bin table (if probabilities are available)
- top_features_POS.csv, top_features_NEG.csv, top_features_NEU.csv: Top and bottom char n-grams by class weight

Usage:
  python modeling/error_analysis.py --model_dir models/baseline_chargram \
    --input_csv annotation/sample_data/final_test.csv \
    --output_dir reports/baseline_chargram \
    --slice_column group --reliability_bins 15

Dependencies:
  pip install scikit-learn joblib pandas
  Optional for plots: matplotlib
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

try:
    import sklearn  # type: ignore
    SKLEARN_VER = getattr(sklearn, "__version__", None)
except Exception:
    SKLEARN_VER = None

try:
    import joblib as _jl  # type: ignore
    JOBLIB_VER = getattr(_jl, "__version__", None)
except Exception:
    JOBLIB_VER = None

try:
    import pandas as _pd  # type: ignore
    PANDAS_VER = getattr(_pd, "__version__", None)
except Exception:
    PANDAS_VER = None

try:
    import numpy as _np  # type: ignore
    NUMPY_VER = getattr(_np, "__version__", None)
except Exception:
    NUMPY_VER = None

MATPLOTLIB_VER = None
try:
    import matplotlib  # type: ignore
    MATPLOTLIB_VER = getattr(matplotlib, "__version__", None)
except Exception:
    pass

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

try:
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

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
    return vectorizer, model


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if not {"id", "text", "label"}.issubset(df.columns):
            raise ValueError(f"CSV must have id,text,label columns. Found: {list(df.columns)}")
        return df.to_dict(orient="records")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "text", "label"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must have id,text,label columns. Found: {reader.fieldnames}")
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str | float]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        # Write header only
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text", "label", "pred_label", "proba_POS", "proba_NEG", "proba_NEU", "confidence", "margin"])
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_confidence_and_margin(probs: List[float]) -> Tuple[float, float]:
    if not probs:
        return 0.0, 0.0
    ordered = sorted(probs, reverse=True)
    conf = float(ordered[0])
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else conf
    return conf, margin


def top_features(vectorizer, model, k: int = 50) -> Dict[str, Dict[str, float]]:
    # Returns mapping class -> top and bottom features with weights
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        feature_names = vectorizer.get_feature_names()
    if not hasattr(model, "coef_"):
        return {}
    coefs = model.coef_  # shape: (n_classes, n_features)
    classes = list(getattr(model, "classes_", LABELS))
    out: Dict[str, Dict[str, float]] = {}
    for ci, cls in enumerate(classes):
        weights = coefs[ci]
        idx_sorted = weights.argsort()
        bottom = idx_sorted[:k]  # most negative for this class
        top = idx_sorted[-k:][::-1]  # most positive
        d: Dict[str, float] = {}
        for i in top:
            d[str(feature_names[i])] = float(weights[i])
        for i in bottom:
            d[str(feature_names[i])] = float(weights[i])
        out[str(cls)] = d
    return out


def save_top_features(output_dir: Path, vec, model, k: int = 50) -> None:
    ensure_dir(output_dir)
    feats = top_features(vec, model, k=k)
    for cls, weights in feats.items():
        rows = sorted(weights.items(), key=lambda x: -x[1])
        p = output_dir / f"top_features_{cls}.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "weight"])
            for feat, wgt in rows:
                w.writerow([feat, f"{wgt:.6f}"])


def compute_calibration_bins(y_true: List[str], y_pred: List[str], proba: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    """Compute calibration bins using max predicted probability as confidence and correctness indicator.
    Returns a list of bins with lower/upper edges, count, accuracy (empirical), and average confidence.
    """
    if proba is None or len(proba) == 0:
        return []
    conf = np.max(proba, axis=1)
    correct = (np.array(y_pred) == np.array(y_true)).astype(float)

    bins = []
    # Bin edges from 0.0 to 1.0
    edges = np.linspace(0.0, 1.0, num=n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(np.sum(mask))
        if count == 0:
            bins.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": 0, "accuracy": 0.0, "confidence": float((lo + hi) / 2.0)})
            continue
        acc = float(np.mean(correct[mask]))
        avg_conf = float(np.mean(conf[mask]))
        bins.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": count, "accuracy": acc, "confidence": avg_conf})
    return bins


def expected_calibration_error(bins: List[Dict[str, float]], total: int) -> float:
    if total <= 0 or not bins:
        return 0.0
    ece = 0.0
    for b in bins:
        w = b["count"] / total
        ece += w * abs(b["accuracy"] - b["confidence"])
    return float(ece)


def maximum_calibration_error(bins: List[Dict[str, float]]) -> float:
    if not bins:
        return 0.0
    return float(max(abs(b["accuracy"] - b["confidence"]) for b in bins))


def save_calibration_bins(path: Path, bins: List[Dict[str, float]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin_lower", "bin_upper", "count", "accuracy", "confidence"])
        w.writeheader()
        for b in bins:
            w.writerow(b)


def plot_reliability_diagram(path: Path, bins: List[Dict[str, float]]) -> None:
    if not MATPLOTLIB_AVAILABLE or not bins:
        return
    ensure_dir(path.parent)
    confidences = [b["confidence"] for b in bins]
    accuracies = [b["accuracy"] for b in bins]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(confidences, accuracies, marker="o", label="Empirical")
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def per_slice_metrics(rows: List[Dict[str, str | float]], slice_column: str) -> Dict[str, Dict[str, float]]:
    """Compute per-slice accuracy and macro-F1 if the slice column is present and populated.
    Returns mapping slice_value -> metrics dict.
    """
    # Build gold and predictions indexed by slice
    slices: Dict[str, Dict[str, List[str]]] = {}
    for r in rows:
        sv = str(r.get(slice_column, "")).strip()
        if sv == "":
            sv = "__EMPTY__"
        gold = str(r.get("label", ""))
        pred = str(r.get("pred_label", ""))
        if gold == "" or pred == "":
            continue
        d = slices.setdefault(sv, {"gold": [], "pred": []})
        d["gold"].append(gold)
        d["pred"].append(pred)

    out: Dict[str, Dict[str, float]] = {}
    for sv, d in slices.items():
        if not d["gold"]:
            continue
        acc = accuracy_score(d["gold"], d["pred"])
        f1m = f1_score(d["gold"], d["pred"], average="macro")
        cm = confusion_matrix(d["gold"], d["pred"], labels=LABELS).tolist()
        out[sv] = {
            "count": float(len(d["gold"])),
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "confusion": cm,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Error analysis for Khmer sentiment baseline (enhanced)")
    ap.add_argument("--model_dir", required=True, help="Directory containing vectorizer.pkl and model.pkl")
    ap.add_argument("--input_csv", required=True, help="CSV with id,text,label (e.g., final_test.csv)")
    ap.add_argument("--output_dir", required=True, help="Directory to save reports")
    ap.add_argument("--top_k", type=int, default=50, help="Top K n-grams per class to export")
    ap.add_argument("--slice_column", default=None, help="Optional column name for per-slice metrics (e.g., group or domain)")
    ap.add_argument("--reliability_bins", type=int, default=15, help="Number of bins for reliability diagram and ECE/MCE")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    vec, model = load_model(model_dir)
    rows = read_csv_rows(input_csv)

    texts = [str(r["text"]) for r in rows]
    gold = [str(r["label"]).upper() for r in rows]

    X = vec.transform(texts)
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None

    # Build misclassified rows with extra info
    mis_rows: List[Dict[str, str | float]] = []
    pred_rows: List[Dict[str, str | float]] = []
    for i, r in enumerate(rows):
        proba_pos = float(probs[i][0]) if probs is not None else 0.0
        proba_neg = float(probs[i][1]) if probs is not None else 0.0
        proba_neu = float(probs[i][2]) if probs is not None else 0.0
        conf, margin = compute_confidence_and_margin([proba_pos, proba_neg, proba_neu]) if probs is not None else (0.0, 0.0)
        pred_rec: Dict[str, str | float] = {
            "id": r.get("id", ""),
            "text": r.get("text", ""),
            "label": gold[i],
            "pred_label": preds[i],
            "proba_POS": proba_pos,
            "proba_NEG": proba_neg,
            "proba_NEU": proba_neu,
            "confidence": conf,
            "margin": margin,
        }
        # Preserve any extra columns to enable slicing downstream
        if isinstance(r, dict):
            for k, v in r.items():
                if k not in pred_rec:
                    pred_rec[k] = v
        pred_rows.append(pred_rec)
        if gold[i] != preds[i]:
            mis_rows.append(pred_rec)

    # Sort misclassifications by margin ascending (most uncertain first)
    if mis_rows:
        mis_rows.sort(key=lambda x: (x.get("margin", 0.0), x.get("confidence", 0.0)))

    write_csv(out_dir / "misclassified.csv", mis_rows)

    # Metrics
    acc = accuracy_score(gold, preds)
    f1m = f1_score(gold, preds, average="macro")
    report = classification_report(gold, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=LABELS).tolist()

    # Calibration metrics
    calibration = {"available": False}
    if probs is not None and len(probs) == len(gold):
        bins = compute_calibration_bins(gold, list(preds), probs, n_bins=int(args.reliability_bins))
        ece = expected_calibration_error(bins, total=len(gold))
        mce = maximum_calibration_error(bins)
        calibration = {
            "available": True,
            "ECE": float(ece),
            "MCE": float(mce),
            "bins": bins,
            "n_bins": int(args.reliability_bins),
        }
        save_calibration_bins(out_dir / "calibration_bins.csv", bins)
        if MATPLOTLIB_AVAILABLE:
            plot_reliability_diagram(out_dir / "reliability_diagram.png", bins)
        else:
            print("matplotlib not available; skipping reliability diagram plot.")

    # Per-slice metrics
    slices: Dict[str, Dict[str, float]] = {}
    if args.slice_column is not None and len(rows) > 0 and (args.slice_column in rows[0]):
        slices = per_slice_metrics(pred_rows, args.slice_column)

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "report": report,
        "confusion": cm,
        "label_order": LABELS,
        "counts": {
            "total": len(rows),
            "misclassified": len(mis_rows)
        },
        "calibration": calibration,
        "slices": {"column": args.slice_column, "metrics": slices} if slices else {},
        "env": {
            "python": sys.version.replace("\n", " "),
            "sklearn": SKLEARN_VER,
            "joblib": JOBLIB_VER,
            "pandas": PANDAS_VER,
            "numpy": NUMPY_VER,
            "matplotlib": MATPLOTLIB_VER,
        },
        "args": {
            "model_dir": str(model_dir),
            "input_csv": str(input_csv),
            "output_dir": str(out_dir),
            "top_k": int(args.top_k),
            "slice_column": args.slice_column,
            "reliability_bins": int(args.reliability_bins),
        },
    }

    with (out_dir / "error_analysis_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Human-readable summary
    summary_lines = []
    summary_lines.append(f"Total: {len(rows)}")
    summary_lines.append(f"Misclassified: {len(mis_rows)}")
    summary_lines.append(f"Accuracy: {acc:.4f}")
    summary_lines.append(f"F1_macro: {f1m:.4f}")
    if calibration.get("available"):
        summary_lines.append("")
        summary_lines.append(f"Calibration (ECE/MCE) with {calibration.get('n_bins')} bins:")
        summary_lines.append(f"  ECE: {calibration.get('ECE'):.4f}")
        summary_lines.append(f"  MCE: {calibration.get('MCE'):.4f}")
    summary_lines.append("")
    summary_lines.append("Per-class F1 and support:")
    for lbl in LABELS:
        f1_lbl = report.get(lbl, {}).get("f1-score", 0.0)
        sup_lbl = report.get(lbl, {}).get("support", 0)
        summary_lines.append(f"  {lbl}: f1={f1_lbl:.4f} support={sup_lbl}")
    summary_lines.append("")
    summary_lines.append("Confusion matrix (rows=true, cols=pred; order=POS,NEG,NEU):")
    for i, row in enumerate(cm):
        summary_lines.append(f"  {LABELS[i]}:\t{row}")
    if slices:
        summary_lines.append("")
        summary_lines.append(f"Per-slice metrics by '{args.slice_column}':")
        for sv, m in slices.items():
            summary_lines.append(f"  {sv}: n={int(m['count'])} acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")

    with (out_dir / "error_analysis_summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Top features
    save_top_features(out_dir, vec, model, k=args.top_k)

    print(f"Wrote misclassifications: {out_dir / 'misclassified.csv'}")
    print(f"Wrote metrics: {out_dir / 'error_analysis_metrics.json'}")
    print(f"Wrote summary: {out_dir / 'error_analysis_summary.txt'}")
    if calibration.get("available"):
        print(f"Wrote calibration bins: {out_dir / 'calibration_bins.csv'}")
        if MATPLOTLIB_AVAILABLE:
            print(f"Wrote reliability diagram: {out_dir / 'reliability_diagram.png'}")
    print(f"Wrote top features per class in: {out_dir}")


if __name__ == "__main__":
    main()
