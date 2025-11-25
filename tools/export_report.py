#!/usr/bin/env python3
"""
Aggregate experiment results into CSV/Markdown tables and plots.

Features
- Collect metrics from multiple model directories (expecting metrics.json)
- Export:
  - overall.csv: overall val/test metrics, calibration, efficiency, params
  - per_language.csv: per-language accuracy/F1 for val/test if available
  - calibration.csv: val/test calibration metrics
  - summary.md: top-k models summary with key metrics
  - plots/: bar charts for test F1 and test ECE (if matplotlib installed)
- Optional paired bootstrap CI for prediction CSV pairs
- Optional robustness deltas by providing stress metrics per model

Usage
  # Aggregate models via glob and write to reports/final
  python tools/export_report.py --glob "models/*" --out_dir reports/final

  # Include paired bootstrap comparisons
  python tools/export_report.py --glob "models/*" --out_dir reports/final \
    --pred_pair runs/baseline/pred_test.csv:runs/xlmr/pred_test.csv:xlmr_vs_baseline

  # Include robustness deltas (map model->stress metrics JSON)
  python tools/export_report.py --models models/xlmr_base models/baseline_chargram \
    --stress models/xlmr_base:reports/stress_xlmr/stress_metrics.json \
    --stress models/baseline_chargram:reports/stress_baseline/stress_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

# For optional CI computation import helper functions from sibling tool
try:
    from .paired_bootstrap_ci import load_predictions as load_preds_ci  # type: ignore
    from .paired_bootstrap_ci import align_by_id as align_preds_ci  # type: ignore
    from .paired_bootstrap_ci import paired_bootstrap as paired_bs  # type: ignore
except Exception:
    load_preds_ci = None  # type: ignore
    align_preds_ci = None  # type: ignore
    paired_bs = None  # type: ignore


@dataclass
class ModelRecord:
    dir: Path
    metrics: Dict


def load_metrics(model_dir: Path) -> Dict:
    p = model_dir / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_models(models: List[str], glob_pattern: Optional[str]) -> List[Path]:
    out: List[Path] = []
    if glob_pattern:
        import glob
        for g in glob.glob(glob_pattern):
            p = Path(g)
            if p.is_dir():
                out.append(p)
    for m in (models or []):
        p = Path(m)
        if p.is_dir():
            out.append(p)
    # Unique preserve order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        s = str(p.resolve())
        if s not in seen:
            uniq.append(p)
            seen.add(s)
    return uniq


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def aggregate_overall(recs: List[ModelRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in recs:
        m = r.metrics
        val = m.get("val", {})
        test = m.get("test", {})
        timing = m.get("timing", {})
        params = m.get("model_params", {})
        row = {
            "model": str(r.dir),
            # core metrics
            "val_accuracy": _num(val.get("eval_accuracy", val.get("accuracy"))),
            "val_f1_macro": _num(val.get("eval_f1_macro", val.get("f1_macro"))),
            "test_accuracy": _num(test.get("eval_accuracy", test.get("accuracy"))),
            "test_f1_macro": _num(test.get("eval_f1_macro", test.get("f1_macro"))),
            # calibration (if present)
            "val_ece": _num(m.get("calibration", {}).get("val", {}).get("ece", m.get("val_ece"))),
            "val_brier": _num(m.get("calibration", {}).get("val", {}).get("brier", m.get("val_brier"))),
            "test_ece": _num(m.get("calibration", {}).get("test", {}).get("ece", m.get("test_ece"))),
            "test_brier": _num(m.get("calibration", {}).get("test", {}).get("brier", m.get("test_brier"))),
            # efficiency
            "train_seconds": _num(timing.get("train_seconds")),
            "test_eval_seconds": _num(timing.get("test_eval_seconds")),
            "test_throughput_examples_per_sec": _num(timing.get("test_throughput_examples_per_sec")),
            # params
            "params_total": _num(params.get("total")),
            "params_trainable": _num(params.get("trainable")),
        }
        rows.append(row)
    return rows


def _num(x):
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip() != "":
            return float(x)
    except Exception:
        pass
    return ""


def aggregate_per_language(recs: List[ModelRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in recs:
        m = r.metrics
        per_l = m.get("per_language", {})
        for split in ("val", "test"):
            d = per_l.get(split, {})
            for k, v in d.items():
                # k format: accuracy__km or f1_macro__en
                if k.startswith("accuracy__"):
                    lang = k.split("__", 1)[1]
                    rows.append({"model": str(r.dir), "split": split, "lang": lang, "metric": "accuracy", "value": _num(v)})
                elif k.startswith("f1_macro__"):
                    lang = k.split("__", 1)[1]
                    rows.append({"model": str(r.dir), "split": split, "lang": lang, "metric": "f1_macro", "value": _num(v)})
    return rows


def aggregate_calibration(recs: List[ModelRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in recs:
        m = r.metrics
        cal = m.get("calibration", {})
        for split in ("val", "test"):
            d = cal.get(split, {})
            if d:
                rows.append({
                    "model": str(r.dir),
                    "split": split,
                    "ece": _num(d.get("ece")),
                    "brier": _num(d.get("brier")),
                    "n_samples": d.get("n_samples", ""),
                    "n_classes": d.get("n_classes", ""),
                })
    return rows


def plot_bar(values: List[Tuple[str, float]], title: str, ylabel: str, out_png: Path) -> bool:
    if plt is None or not values:
        return False
    try:
        labels = [Path(m).name for m, _ in values]
        nums = [v for _, v in values]
        fig = plt.figure(figsize=(max(6, len(values) * 1.2), 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.bar(range(len(values)), nums)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png)
        plt.close(fig)
        return True
    except Exception:
        return False


def export_summary_md(out_dir: Path, overall_rows: List[Dict[str, object]], top_k: int = 10) -> None:
    # Rank by test_f1_macro desc
    def _getf(r, k):
        v = r.get(k, None)
        try:
            return float(v)
        except Exception:
            return float("nan")
    ranked = sorted(overall_rows, key=lambda r: (_getf(r, "test_f1_macro") if not _isnan(_getf(r, "test_f1_macro")) else -1), reverse=True)
    top = ranked[:top_k]
    lines = []
    lines.append("# Experiment Summary\n")
    lines.append("## Top models by test macro-F1\n")
    lines.append("| Model | Val Acc | Val F1 | Test Acc | Test F1 | Val ECE | Test ECE | Params (M) | Train s | Throughput |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in top:
        params_m = "" if r.get("params_total", "") == "" else f"{float(r['params_total'])/1e6:.2f}"
        lines.append(
            f"| {Path(str(r['model'])).name} | {r.get('val_accuracy','')} | {r.get('val_f1_macro','')} | "
            f"{r.get('test_accuracy','')} | {r.get('test_f1_macro','')} | {r.get('val_ece','')} | {r.get('test_ece','')} | "
            f"{params_m} | {r.get('train_seconds','')} | {r.get('test_throughput_examples_per_sec','')} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _isnan(x: float) -> bool:
    try:
        import math
        return math.isnan(x)
    except Exception:
        return False


def run_paired_bootstrap(pairs: List[str], out_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    if not pairs:
        return results
    if load_preds_ci is None or align_preds_ci is None or paired_bs is None:
        print("Warning: paired bootstrap not available (import failed)")
        return results
    for spec in pairs:
        # spec format: a.csv:b.csv[:name]
        parts = spec.split(":")
        if len(parts) < 2:
            print(f"Skipping invalid --pred_pair spec: {spec}")
            continue
        a = Path(parts[0])
        b = Path(parts[1])
        name = parts[2] if len(parts) > 2 else f"{a.name}_VS_{b.name}"
        try:
            df_a = load_preds_ci(a)
            df_b = load_preds_ci(b)
            A, B = align_preds_ci(df_a, df_b)
            y_true = A["label"].values
            y_pred_a = A["pred_label"].values
            y_pred_b = B["pred_label"].values
            summary = paired_bs(y_true, y_pred_a, y_pred_b, n_bootstrap=2000, alpha=0.05, seed=123)
            out_p = out_dir / "bootstrap" / f"{name}.json"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append({"name": name, "path": str(out_p)})
        except Exception as e:
            print(f"Failed CI for {spec}: {e}")
    return results


def parse_stress_args(items: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for it in items or []:
        # format: model_dir:stress_metrics.json
        try:
            m, s = it.split(":", 1)
            mapping[Path(m).resolve().as_posix()] = Path(s)
        except Exception:
            print(f"Skipping invalid --stress mapping: {it}")
    return mapping


def aggregate_stress_deltas(overall: List[Dict[str, object]], stress_map: Dict[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    # Build lookup of model->row
    by_model = {Path(r["model"]).resolve().as_posix(): r for r in overall}
    for m_key, stress_json in stress_map.items():
        if m_key not in by_model:
            print(f"Warning: --stress model not found in aggregated list: {m_key}")
            continue
        base = by_model[m_key]
        try:
            sdata = json.loads(Path(stress_json).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read stress metrics {stress_json}: {e}")
            continue
        ov = sdata.get("overall", {})
        s_acc = ov.get("accuracy", None)
        s_f1 = ov.get("f1_macro", None)
        rows.append({
            "model": m_key,
            "stress_path": str(stress_json),
            "delta_test_accuracy": _num(base.get("test_accuracy", "")) - _num(s_acc),
            "delta_test_f1_macro": _num(base.get("test_f1_macro", "")) - _num(s_f1),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Export aggregated experiment report (CSV/plots/markdown)")
    ap.add_argument("--models", nargs="*", help="Model directories containing metrics.json")
    ap.add_argument("--glob", help="Glob pattern for model directories (e.g., 'models/*' or 'runs/*')")
    ap.add_argument("--out_dir", default="reports/final", help="Output directory for aggregated reports")
    ap.add_argument("--pred_pair", action="append", help="Prediction pair for paired bootstrap 'a.csv:b.csv[:name]' (repeatable)")
    ap.add_argument("--stress", action="append", help="Map model_dir to stress metrics: 'model_dir:reports/.../stress_metrics.json' (repeatable)")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and load
    model_dirs = collect_models(args.models or [], args.glob)
    if not model_dirs:
        raise SystemExit("No model directories provided (use --models or --glob)")

    records: List[ModelRecord] = [ModelRecord(d, load_metrics(d)) for d in model_dirs]

    # Aggregate tables
    overall_rows = aggregate_overall(records)
    per_lang_rows = aggregate_per_language(records)
    calib_rows = aggregate_calibration(records)

    # Write CSVs
    write_csv(out_dir / "overall.csv", overall_rows, [
        "model", "val_accuracy", "val_f1_macro", "test_accuracy", "test_f1_macro",
        "val_ece", "val_brier", "test_ece", "test_brier",
        "train_seconds", "test_eval_seconds", "test_throughput_examples_per_sec",
        "params_total", "params_trainable",
    ])
    write_csv(out_dir / "per_language.csv", per_lang_rows, ["model", "split", "lang", "metric", "value"])
    write_csv(out_dir / "calibration.csv", calib_rows, ["model", "split", "ece", "brier", "n_samples", "n_classes"])

    # Plots
    try:
        # Test macro-F1 bar chart
        f1_vals = [(r["model"], float(r["test_f1_macro"])) for r in overall_rows if isinstance(r.get("test_f1_macro", None), float)]
        plot_bar(f1_vals, "Test Macro-F1 by Model", "Macro-F1", out_dir / "plots" / "bar_test_f1.png")
        # Test ECE bar chart
        ece_vals = [(r["model"], float(r["test_ece"])) for r in overall_rows if isinstance(r.get("test_ece", None), float)]
        plot_bar(ece_vals, "Test ECE by Model", "ECE", out_dir / "plots" / "bar_test_ece.png")
    except Exception:
        pass

    # Markdown summary
    export_summary_md(out_dir, overall_rows, top_k=int(args.top_k))

    # Paired bootstrap CIs
    bs_results = run_paired_bootstrap(args.pred_pair or [], out_dir)
    if bs_results:
        (out_dir / "bootstrap" / "README.md").write_text(
            "\n".join([f"- {r['name']}: {r['path']}" for r in bs_results]), encoding="utf-8"
        )

    # Stress deltas (optional)
    stress_map = parse_stress_args(args.stress or [])
    if stress_map:
        stress_rows = aggregate_stress_deltas(overall_rows, stress_map)
        write_csv(out_dir / "stress_deltas.csv", stress_rows, ["model", "stress_path", "delta_test_accuracy", "delta_test_f1_macro"])

    print(f"Exported aggregated report to {out_dir}")


if __name__ == "__main__":
    main()
