#!/usr/bin/env python3
"""
Compute inter-annotator agreement (IAA) metrics from raw annotation exports.

Supported inputs
- A CSV with columns: id, annotator, label
- Optionally: lang column to filter/report per-language

Outputs
- Prints overall and per-annotator-pair agreement (Cohen's kappa)
- Optionally writes a JSON summary via --output_json

Examples
  python tools/compute_iaa.py --input annotation/annotations_raw.csv
  python tools/compute_iaa.py --input annotation/annotations_raw.csv --lang_column lang --output_json reports/annotation_iaa.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


LABELS = ["NEG", "NEU", "POS"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        return df.to_dict(orient="records")  # type: ignore
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _kappa(a: List[str], b: List[str], labels: List[str]) -> float:
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    # Observed agreement
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    # Expected agreement
    from collections import Counter

    ca = Counter(a)
    cb = Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    denom = (1 - pe)
    if denom == 0:
        return float("nan")
    return (po - pe) / denom


def compute_iaa(rows: List[Dict[str, str]], lang_column: str | None) -> Dict[str, object]:
    # Map: annotator -> {id -> label}
    by_annot: Dict[str, Dict[str, str]] = defaultdict(dict)
    langs_by_id: Dict[str, str] = {}
    for r in rows:
        rid = str(r.get("id", ""))
        ann = str(r.get("annotator", ""))
        lab = str(r.get("label", "")).strip().upper()
        if rid and ann and lab:
            by_annot[ann][rid] = lab
            if lang_column:
                l = str(r.get(lang_column, "")).strip().lower()
                if l:
                    langs_by_id[rid] = l

    annotators = sorted(by_annot.keys())
    pair_scores: Dict[str, float] = {}
    per_lang_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for a, b in combinations(annotators, 2):
        # Common items
        common_ids = sorted(set(by_annot[a].keys()) & set(by_annot[b].keys()))
        la = [by_annot[a][i] for i in common_ids]
        lb = [by_annot[b][i] for i in common_ids]
        k = _kappa(la, lb, LABELS)
        pair_key = f"{a}__vs__{b}"
        pair_scores[pair_key] = k
        # Per-language subsets
        if langs_by_id:
            by_lang: Dict[str, Tuple[List[str], List[str]]] = defaultdict(lambda: ([], []))
            for i, x, y in zip(common_ids, la, lb):
                l = langs_by_id.get(i, "")
                if l:
                    xs, ys = by_lang[l]
                    xs.append(x)
                    ys.append(y)
                    by_lang[l] = (xs, ys)
            for l, (xs, ys) in by_lang.items():
                per_lang_scores[l][pair_key] = _kappa(xs, ys, LABELS)

    overall = {
        "annotators": annotators,
        "pairs": pair_scores,
        "pairs_per_lang": per_lang_scores,
    }
    return overall


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute inter-annotator agreement (Cohen's kappa) from raw annotations")
    ap.add_argument("--input", required=True, help="CSV with columns: id,annotator,label[,<lang_column>]")
    ap.add_argument("--lang_column", default=None, help="Optional language column name (e.g., 'lang')")
    ap.add_argument("--output_json", default=None, help="Optional path to write JSON summary")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    rows = _read_csv(inp)
    res = compute_iaa(rows, args.lang_column)

    print("=== Inter-Annotator Agreement (Cohen's kappa) ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote IAA summary to {out}")


if __name__ == "__main__":
    main()
