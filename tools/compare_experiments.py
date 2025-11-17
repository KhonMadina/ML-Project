#!/usr/bin/env python3
"""
Compare experiments by aggregating key metrics from models/*/metrics.json.

Usage:
  python tools/compare_experiments.py --models models/baseline_chargram models/baseline_chargram_cal
  python tools/compare_experiments.py --glob "models/*"

Outputs a CSV to stdout with columns: model, val_accuracy, val_f1_macro, test_accuracy, test_f1_macro, n_train, n_val, n_test, calibrate
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict


def load_metrics(model_dir: Path) -> Dict:
    p = model_dir / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def row_from_metrics(model_dir: Path, m: Dict) -> Dict[str, str]:
    if not m:
        return {}
    params = m.get("params", {})
    sizes = m.get("sizes", {})
    val = m.get("val", {})
    test = m.get("test", {})
    return {
        "model": str(model_dir),
        "val_accuracy": f"{val.get('accuracy', float('nan')):.6f}" if isinstance(val.get('accuracy'), (float, int)) else str(val.get('accuracy', '')),
        "val_f1_macro": f"{val.get('f1_macro', float('nan')):.6f}" if isinstance(val.get('f1_macro'), (float, int)) else str(val.get('f1_macro', '')),
        "test_accuracy": f"{test.get('accuracy', float('nan')):.6f}" if isinstance(test.get('accuracy'), (float, int)) else str(test.get('accuracy', '')),
        "test_f1_macro": f"{test.get('f1_macro', float('nan')):.6f}" if isinstance(test.get('f1_macro'), (float, int)) else str(test.get('f1_macro', '')),
        "n_train": str(sizes.get('train', '')),
        "n_val": str(sizes.get('val', '')),
        "n_test": str(sizes.get('test', '')),
        "calibrate": str(params.get('calibrate', 'none')),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate metrics.json from multiple models")
    ap.add_argument("--models", nargs="*", help="List of model directories")
    ap.add_argument("--glob", help="Glob pattern to expand (e.g., 'models/*')")
    args = ap.parse_args()

    model_dirs: List[Path] = []
    if args.glob:
        import glob
        for g in glob.glob(args.glob):
            p = Path(g)
            if p.is_dir():
                model_dirs.append(p)
    if args.models:
        for m in args.models:
            p = Path(m)
            if p.is_dir():
                model_dirs.append(p)

    if not model_dirs:
        print("No model directories provided.", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, str]] = []
    for md in model_dirs:
        m = load_metrics(md)
        r = row_from_metrics(md, m)
        if r:
            rows.append(r)

    # Write CSV to stdout
    w = csv.DictWriter(sys.stdout, fieldnames=[
        "model", "val_accuracy", "val_f1_macro", "test_accuracy", "test_f1_macro", "n_train", "n_val", "n_test", "calibrate"
    ])
    w.writeheader()
    for r in rows:
        w.writerow(r)


if __name__ == "__main__":
    main()
