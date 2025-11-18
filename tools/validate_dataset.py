#!/usr/bin/env python3
"""
Dataset validator for Khmer Sentiment Analysis Pipeline

Purpose
- Enforce the accepted training data schema and basic quality checks before ingestion.
- Accepted CSV schema: id,text,label[,<group_column>]
  * id: unique string identifier (required)
  * text: UTF-8 text (required, non-empty)
  * label: POS | NEG | NEU (case-insensitive; normalized to uppercase) (required)
  * group_column (optional): user/thread/group id for leakage-safe splits

Checks
- Required columns exist (id,text,label)
- Optional group column present when provided via --group_column
- Unique ids (no duplicates)
- Non-empty text after stripping whitespace
- Label normalization to {POS,NEG,NEU} or allow extended labels with --allow_extra_labels
- Summary: row count, label distribution, % empty text, group coverage
- Optional: write a normalized copy with labels uppercased via --output_clean

Examples
  # Basic validation
  python tools/validate_dataset.py --input annotation/sample_data/final_dataset.csv

  # Validate with group column present and emit a cleaned CSV
  python tools/validate_dataset.py \
    --input annotation/sample_data/new_annotations.csv \
    --group_column group \
    --output_clean annotation/sample_data/new_annotations_clean.csv

  # Allow extra labels (for exploratory or multi-class sources)
  python tools/validate_dataset.py --input data/raw.csv --allow_extra_labels
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

LABELS = {"POS", "NEG", "NEU"}


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV as list of dicts and return rows and header fields.
    Prefers pandas if available; otherwise uses csv module.
    """
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        rows = df.to_dict(orient="records")  # type: ignore
        fields = list(df.columns)
        return rows, fields
    rows2: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for r in reader:
            rows2.append(r)
    return rows2, fields


def label_norm(s: str) -> str:
    return (s or "").strip().upper()


def summarize(rows: List[Dict[str, str]], group_column: Optional[str]) -> Dict[str, object]:
    total = len(rows)
    by_label: Dict[str, int] = {}
    empty_text = 0
    seen_ids = set()
    dup_ids = 0
    group_count = 0

    for r in rows:
        rid = str(r.get("id", ""))
        if rid in seen_ids:
            dup_ids += 1
        else:
            seen_ids.add(rid)
        txt = str(r.get("text", ""))
        if txt.strip() == "":
            empty_text += 1
        lab = label_norm(str(r.get("label", "")))
        by_label[lab] = by_label.get(lab, 0) + 1
        if group_column is not None and str(r.get(group_column, "")).strip() != "":
            group_count += 1

    return {
        "total_rows": total,
        "unique_ids": len(seen_ids),
        "duplicate_ids": dup_ids,
        "empty_text": empty_text,
        "empty_text_pct": (empty_text / total * 100.0) if total else 0.0,
        "label_distribution": by_label,
        "group_nonempty_count": group_count if group_column is not None else None,
    }


def validate(
    rows: List[Dict[str, str]],
    fields: List[str],
    allow_extra_labels: bool,
    group_column: Optional[str],
) -> List[str]:
    msgs: List[str] = []

    required = {"id", "text", "label"}
    missing = [c for c in required if c not in fields]
    if missing:
        msgs.append(f"ERROR: Missing required columns: {missing}. Found: {fields}")
        return msgs

    if group_column and group_column not in fields:
        msgs.append(f"ERROR: --group_column '{group_column}' not found in columns: {fields}")

    seen_ids = set()
    for idx, r in enumerate(rows):
        rid = str(r.get("id", ""))
        if rid == "":
            msgs.append(f"Row {idx}: empty id")
        if rid in seen_ids:
            msgs.append(f"Row {idx}: duplicate id '{rid}'")
        else:
            seen_ids.add(rid)

        txt = str(r.get("text", ""))
        if txt.strip() == "":
            msgs.append(f"Row {idx} id={rid}: empty text")

        lab = label_norm(str(r.get("label", "")))
        if lab == "":
            msgs.append(f"Row {idx} id={rid}: empty label")
        elif not allow_extra_labels and lab not in LABELS:
            msgs.append(f"Row {idx} id={rid}: invalid label '{lab}' (expected one of {sorted(LABELS)})")

    return msgs


def write_clean_copy(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force label to uppercase in output
    out_fields = fields[:]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            r2 = r.copy()
            if "label" in r2 and r2["label"] is not None:
                r2["label"] = label_norm(str(r2["label"]))
            w.writerow({k: r2.get(k, "") for k in out_fields})


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Khmer sentiment dataset CSV format and basic quality")
    ap.add_argument("--input", required=True, help="Path to CSV with columns: id,text,label[,<group_column>]")
    ap.add_argument("--group_column", default=None, help="Optional group column name for leakage-safe splits")
    ap.add_argument("--allow_extra_labels", action="store_true", help="Allow labels outside {POS,NEG,NEU} (for exploratory datasets)")
    ap.add_argument("--output_clean", help="Optional path to write a cleaned CSV with labels uppercased")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    try:
        rows, fields = read_csv_rows(inp)
    except UnicodeDecodeError as e:
        raise SystemExit(f"Failed to read CSV as UTF-8: {e}")

    msgs = validate(rows, fields, allow_extra_labels=bool(args.allow_extra_labels), group_column=args.group_column)
    summary = summarize(rows, group_column=args.group_column)

    print("=== Dataset Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if msgs:
        print("\n=== Issues ===")
        for m in msgs[:500]:  # cap spam
            print("- ", m)
        if len(msgs) > 500:
            print(f"... and {len(msgs)-500} more")
        print("\nValidation FAILED (see issues above).")
        # Still allow writing a clean copy if requested (may help fix casing only)
    else:
        print("\nValidation PASSED.")

    if args.output_clean:
        try:
            write_clean_copy(Path(args.output_clean), rows, fields)
            print(f"Wrote cleaned CSV to {args.output_clean}")
        except Exception as e:
            print(f"Failed to write cleaned CSV: {e}")


if __name__ == "__main__":
    main()
