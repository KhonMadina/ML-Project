#!/usr/bin/env python3
"""
McNemar's test for paired classification predictions.

Given two prediction CSVs with columns id,label,pred_label (same examples),
compute the 2x2 contingency table of disagreements and McNemar's test statistic
with continuity correction and the corresponding p-value.

Usage:
  python tools/mcnemar_test.py \
    --predictions_a runs/baseline/test_predictions.csv \
    --predictions_b runs/xlmr/test_predictions.csv \
    --output reports/mcnemar_baseline_vs_xlmr

Outputs:
- mcnemar_summary.json: contains b, c, chi2_cc, p_value, and table

Requirements:
  pip install pandas scipy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except Exception as e:
    raise SystemExit("Install pandas: pip install pandas\n" + str(e))

try:
    from scipy.stats import chi2  # type: ignore
except Exception as e:
    raise SystemExit("Install SciPy: pip install scipy\n" + str(e))


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    cols = {c.lower(): c for c in df.columns}
    required = ["id", "label", "pred_label"]
    for r in required:
        if r not in cols:
            raise SystemExit(f"{path}: missing required column '{r}' (case-insensitive)")
    df = df.rename(columns={cols["id"]: "id", cols["label"]: "label", cols["pred_label"]: "pred_label"})
    df["label"] = df["label"].astype(str).str.upper()
    df["pred_label"] = df["pred_label"].astype(str).str.upper()
    return df[["id", "label", "pred_label"]]


def align_on_id(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(merged) == 0:
        raise SystemExit("No overlapping ids between the two prediction files")
    a2 = merged[["id", "label_a", "pred_label_a"]].rename(columns={"label_a": "label", "pred_label_a": "pred_label"})
    b2 = merged[["id", "label_b", "pred_label_b"]].rename(columns={"label_b": "label", "pred_label_b": "pred_label"})
    return a2.reset_index(drop=True), b2.reset_index(drop=True)


def mcnemar(b: int, c: int) -> tuple[float, float]:
    # Continuity-corrected chi-square statistic: (|b-c|-1)^2 / (b+c)
    if b + c == 0:
        return 0.0, 1.0
    chi2_cc = (abs(b - c) - 1) ** 2 / float(b + c)
    p = 1 - chi2.cdf(chi2_cc, df=1)
    return float(chi2_cc), float(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="McNemar's test for paired predictions")
    ap.add_argument("--predictions_a", required=True, help="CSV with id,label,pred_label for model A")
    ap.add_argument("--predictions_b", required=True, help="CSV with id,label,pred_label for model B")
    ap.add_argument("--output", required=True, help="Directory to write mcnemar_summary.json")
    args = ap.parse_args()

    a = load_predictions(Path(args.predictions_a))
    b = load_predictions(Path(args.predictions_b))
    a2, b2 = align_on_id(a, b)

    if not (a2["label"].values == b2["label"].values).all():
        raise SystemExit("Mismatched labels across the two files; ensure same gold set")

    # 2x2 contingency table on disagreements
    # b: A is correct, B is wrong
    # c: A is wrong, B is correct
    gold = a2["label"].values
    a_ok = (a2["pred_label"].values == gold)
    b_ok = (b2["pred_label"].values == gold)
    b_count = int(((a_ok == True) & (b_ok == False)).sum())
    c_count = int(((a_ok == False) & (b_ok == True)).sum())

    chi2_cc, p_value = mcnemar(b_count, c_count)

    table = {
        "a_correct_b_correct": int(((a_ok == True) & (b_ok == True)).sum()),
        "a_correct_b_wrong": b_count,
        "a_wrong_b_correct": c_count,
        "a_wrong_b_wrong": int(((a_ok == False) & (b_ok == False)).sum()),
    }

    out = {
        "b": b_count,
        "c": c_count,
        "chi2_cc": float(chi2_cc),
        "p_value": float(p_value),
        "table": table,
        "args": {
            "predictions_a": args.predictions_a,
            "predictions_b": args.predictions_b,
            "output": args.output,
        }
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "mcnemar_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir / 'mcnemar_summary.json'}")


if __name__ == "__main__":
    main()
