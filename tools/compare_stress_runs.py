#!/usr/bin/env python3
"""
Compare stress test runs against a clean baseline and compute robustness deltas.

Inputs
- --clean_dir: directory containing clean/stress_metrics.json (produced by modeling/stress_eval.py without a preset)
- --stress_dirs: one or more directories, each containing stress_metrics.json (produced by stress_eval.py with a preset)
- --output: directory to write robustness_summary.json and robustness_summary.csv

Outputs
- robustness_summary.json:
  {
    "clean": {overall metrics},
    "runs": [
       {
         "name": <dir name>,
         "preset": <preset>,
         "severity": <float>,
         "prob": <float>,
         "overall": {acc, f1_macro},
         "delta": {acc, f1_macro},
         "per_category": {
            <category>: {acc, f1_macro, delta_acc, delta_f1_macro, count}
         }
       }, ...
    ]
  }
- robustness_summary.csv:
  combo-level flat rows with preset, severity, prob, acc, f1, delta_acc, delta_f1

Usage
  python tools/compare_stress_runs.py \
    --clean_dir runs/xlmr_base_stress/clean \
    --stress_dirs runs/xlmr_base_stress/diacritics_s3 runs/xlmr_base_stress/emoji_s1 \
    --output runs/xlmr_base_stress

Requirements
  pip install pandas
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_metrics(dir_path: Path) -> Dict[str, Any]:
    mpath = dir_path / "stress_metrics.json"
    if not mpath.exists():
        raise SystemExit(f"Missing stress_metrics.json in {dir_path}")
    try:
        return json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to parse {mpath}: {e}")


def overall_from(metrics: Dict[str, Any]) -> Dict[str, float]:
    ov = metrics.get("overall", {})
    return {
        "accuracy": float(ov.get("accuracy", 0.0)),
        "f1_macro": float(ov.get("f1_macro", 0.0)),
    }


def per_cat_from(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    pc = metrics.get("per_category", {})
    if isinstance(pc, dict):
        for cat, m in pc.items():
            try:
                out[str(cat)] = {
                    "count": float(m.get("count", 0.0)),
                    "accuracy": float(m.get("accuracy", 0.0)),
                    "f1_macro": float(m.get("f1_macro", 0.0)),
                }
            except Exception:
                continue
    return out


def meta_from(metrics: Dict[str, Any]) -> Dict[str, Any]:
    a = metrics.get("args", {}) if isinstance(metrics.get("args"), dict) else {}
    return {
        "preset": a.get("preset"),
        "severity": a.get("severity"),
        "prob": a.get("prob"),
    }


def write_csv_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute robustness deltas vs. a clean baseline for stress tests")
    ap.add_argument("--clean_dir", required=True, help="Directory with clean stress_metrics.json (no preset)")
    ap.add_argument("--stress_dirs", nargs="+", required=True, help="One or more directories with stress_metrics.json (with preset)")
    ap.add_argument("--output", required=True, help="Directory to write robustness_summary.json and CSVs")
    args = ap.parse_args()

    clean_dir = Path(args.clean_dir)
    stress_dirs = [Path(p) for p in args.stress_dirs]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_metrics = load_metrics(clean_dir)
    clean_overall = overall_from(clean_metrics)
    clean_per_cat = per_cat_from(clean_metrics)

    runs_out: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    csv_rows_cat: List[Dict[str, Any]] = []

    for sd in stress_dirs:
        m = load_metrics(sd)
        meta = meta_from(m)
        name = sd.name
        ov = overall_from(m)
        pc = per_cat_from(m)
        # deltas
        d_acc = ov["accuracy"] - clean_overall["accuracy"]
        d_f1 = ov["f1_macro"] - clean_overall["f1_macro"]
        per_cat_delta: Dict[str, Dict[str, float]] = {}
        for cat, mm in pc.items():
            cm = clean_per_cat.get(cat, {"accuracy": 0.0, "f1_macro": 0.0, "count": 0.0})
            per_cat_delta[cat] = {
                "count": float(mm.get("count", 0.0)),
                "accuracy": float(mm.get("accuracy", 0.0)),
                "f1_macro": float(mm.get("f1_macro", 0.0)),
                "delta_accuracy": float(mm.get("accuracy", 0.0) - cm.get("accuracy", 0.0)),
                "delta_f1_macro": float(mm.get("f1_macro", 0.0) - cm.get("f1_macro", 0.0)),
            }
            csv_rows_cat.append({
                "run": name,
                "preset": meta.get("preset"),
                "severity": meta.get("severity"),
                "prob": meta.get("prob"),
                "category": cat,
                "count": per_cat_delta[cat]["count"],
                "acc": per_cat_delta[cat]["accuracy"],
                "f1": per_cat_delta[cat]["f1_macro"],
                "delta_acc": per_cat_delta[cat]["delta_accuracy"],
                "delta_f1": per_cat_delta[cat]["delta_f1_macro"],
            })

        runs_out.append({
            "name": name,
            "preset": meta.get("preset"),
            "severity": meta.get("severity"),
            "prob": meta.get("prob"),
            "overall": {"accuracy": ov["accuracy"], "f1_macro": ov["f1_macro"]},
            "delta": {"accuracy": d_acc, "f1_macro": d_f1},
            "per_category": per_cat_delta,
        })
        csv_rows.append({
            "run": name,
            "preset": meta.get("preset"),
            "severity": meta.get("severity"),
            "prob": meta.get("prob"),
            "acc": ov["accuracy"],
            "f1": ov["f1_macro"],
            "delta_acc": d_acc,
            "delta_f1": d_f1,
        })

    summary = {
        "clean": {"overall": clean_overall, "per_category": clean_per_cat},
        "runs": runs_out,
        "args": {
            "clean_dir": str(clean_dir),
            "stress_dirs": [str(p) for p in stress_dirs],
            "output": str(out_dir),
        }
    }

    # Write outputs
    with (out_dir / "robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv_summary(out_dir / "robustness_summary.csv", csv_rows)
    write_csv_summary(out_dir / "robustness_per_category.csv", csv_rows_cat)

    print(f"Wrote: {out_dir / 'robustness_summary.json'}")
    print(f"Wrote: {out_dir / 'robustness_summary.csv'}")
    print(f"Wrote: {out_dir / 'robustness_per_category.csv'}")


if __name__ == "__main__":
    main()
