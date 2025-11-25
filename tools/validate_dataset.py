#!/usr/bin/env python3
"""
Dataset validator for Khmer + English Sentiment Analysis Pipeline

Purpose
- Enforce accepted training data schema and basic quality checks before ingestion.
- Supported CSV schema (minimal): id,text,label
- Optional recommended columns: lang,group,split,source
  * id: unique string identifier (required)
  * text: UTF-8 text (required, non-empty)
  * label: POS | NEG | NEU (case-insensitive; normalized to uppercase) (required)
  * lang (optional): language code per row, default column name 'lang' (e.g., km, en)
  * group (optional): user/thread/group id for leakage-safe splits
  * split (optional): train | val | test
  * source (optional): dataset/source name

Checks
- Required columns exist (id,text,label)
- Optional columns, when specified via flags, must exist
- Unique ids (no duplicates)
- Non-empty text after stripping whitespace
- Label normalization to {POS,NEG,NEU} unless --allow_extra_labels
- Language value in --allowed_langs if lang column present (unless --allow_extra_langs)
- Split value in {train,val,test} if split column present
- No ID leakage across splits when split column present
- If group column present, warn on group leakage across splits
- Script/code-switch summary (Khmer vs Latin mix)
- Summary: row count, label/lang/split distributions, % empty text, group coverage
- Optional: write a normalized copy with labels uppercased via --output_clean

Examples
  # Basic validation
  python tools/validate_dataset.py --input annotation/sample_data/final_dataset.csv

  # With bilingual checks and cleaned output
  python tools/validate_dataset.py \
    --input annotation/sample_data/final_dataset.csv \
    --lang_column lang --allowed_langs km en \
    --output_clean annotation/sample_data/final_dataset_clean.csv

  # Validate with group + split columns
  python tools/validate_dataset.py \
    --input annotation/sample_data/final_dataset.csv \
    --group_column group --split_column split

  # Allow extra labels/languages (for exploratory or multi-class sources)
  python tools/validate_dataset.py --input data/raw.csv --allow_extra_labels --allow_extra_langs
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

LABELS = {"POS", "NEG", "NEU"}
SPLITS = {"train", "val", "test"}
# Khmer: \u1780-\u17FF; common combining/zero-width chars handled implicitly
_KHMER_RE = re.compile(r"[\u1780-\u17FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


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


def _lang_norm(s: str) -> str:
    return (s or "").strip().lower()


def _detect_codeswitch(text: str) -> str:
    has_kh = bool(_KHMER_RE.search(text))
    has_lat = bool(_LATIN_RE.search(text))
    if has_kh and has_lat:
        return "km+en"
    if has_kh:
        return "km"
    if has_lat:
        return "en"
    return "other"


def summarize(
    rows: List[Dict[str, str]],
    group_column: Optional[str],
    lang_column: Optional[str],
    split_column: Optional[str],
    allowed_langs: Optional[List[str]],
) -> Dict[str, object]:
    total = len(rows)
    by_label: Counter[str] = Counter()
    by_lang: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    empty_text = 0
    seen_ids = set()
    dup_ids = 0
    group_count = 0
    codeswitch: Counter[str] = Counter()

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
        by_label[lab] += 1
        if lang_column:
            lv = _lang_norm(str(r.get(lang_column, "")))
            by_lang[lv] += 1
        if split_column:
            sv = str(r.get(split_column, "")).strip().lower()
            by_split[sv] += 1
        if group_column is not None and str(r.get(group_column, "")).strip() != "":
            group_count += 1
        if txt:
            codeswitch[_detect_codeswitch(txt)] += 1

    summary = {
        "total_rows": total,
        "unique_ids": len(seen_ids),
        "duplicate_ids": dup_ids,
        "empty_text": empty_text,
        "empty_text_pct": (empty_text / total * 100.0) if total else 0.0,
        "label_distribution": dict(by_label),
        "group_nonempty_count": group_count if group_column is not None else None,
        "codeswitch_counts": dict(codeswitch),
    }
    if lang_column:
        # If allowed_langs provided, only show those first in ordered dict-like print
        if allowed_langs:
            ordered = {k: by_lang.get(k, 0) for k in allowed_langs}
            for k, v in by_lang.items():
                if k not in ordered:
                    ordered[k] = v
            summary["lang_distribution"] = ordered
        else:
            summary["lang_distribution"] = dict(by_lang)
    if split_column:
        summary["split_distribution"] = dict(by_split)
    return summary


def validate(
    rows: List[Dict[str, str]],
    fields: List[str],
    allow_extra_labels: bool,
    group_column: Optional[str],
    lang_column: Optional[str],
    split_column: Optional[str],
    allowed_langs: Optional[List[str]],
    allow_extra_langs: bool,
) -> List[str]:
    msgs: List[str] = []

    required = {"id", "text", "label"}
    missing = [c for c in required if c not in fields]
    if missing:
        msgs.append(f"ERROR: Missing required columns: {missing}. Found: {fields}")
        return msgs

    if group_column and group_column not in fields:
        msgs.append(f"ERROR: --group_column '{group_column}' not found in columns: {fields}")
    if lang_column and lang_column not in fields:
        msgs.append(f"ERROR: --lang_column '{lang_column}' not found in columns: {fields}")
    if split_column and split_column not in fields:
        msgs.append(f"ERROR: --split_column '{split_column}' not found in columns: {fields}")

    seen_ids = set()
    id_to_split: Dict[str, str] = {}
    group_to_splits: Dict[str, set] = defaultdict(set)
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

        if lang_column:
            lv = _lang_norm(str(r.get(lang_column, "")))
            if lv == "":
                msgs.append(f"Row {idx} id={rid}: empty {lang_column}")
            elif (not allow_extra_langs) and allowed_langs and lv not in set(allowed_langs):
                msgs.append(
                    f"Row {idx} id={rid}: invalid language '{lv}' (expected one of {allowed_langs})"
                )

        if split_column:
            sv = str(r.get(split_column, "")).strip().lower()
            if sv == "":
                msgs.append(f"Row {idx} id={rid}: empty {split_column}")
            elif sv not in SPLITS:
                msgs.append(f"Row {idx} id={rid}: invalid split '{sv}' (expected one of {sorted(SPLITS)})")
            else:
                if rid in id_to_split and id_to_split[rid] != sv:
                    msgs.append(
                        f"Row {idx} id={rid}: appears in multiple splits ('{id_to_split[rid]}' and '{sv}')"
                    )
                id_to_split[rid] = sv

        if group_column:
            gv = str(r.get(group_column, "")).strip()
            if gv != "" and split_column:
                sv2 = str(r.get(split_column, "")).strip().lower()
                if sv2:
                    group_to_splits[gv].add(sv2)

    if group_column and split_column:
        for g, ss in group_to_splits.items():
            if len(ss) > 1:
                msgs.append(
                    f"WARNING: group '{g}' spans multiple splits {sorted(ss)} (potential leakage)."
                )

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
    ap = argparse.ArgumentParser(description="Validate Khmer/English sentiment dataset CSV format and basic quality")
    ap.add_argument("--input", required=True, help="Path to CSV with columns: id,text,label[,lang,group,split,source]")
    ap.add_argument("--group_column", default=None, help="Optional group column name for leakage-safe splits")
    ap.add_argument("--lang_column", default=None, help="Optional language column name (e.g., 'lang')")
    ap.add_argument("--split_column", default=None, help="Optional split column name ('train'|'val'|'test')")
    ap.add_argument("--allowed_langs", nargs="*", default=None, help="List of allowed language codes (e.g., km en)")
    ap.add_argument("--allow_extra_labels", action="store_true", help="Allow labels outside {POS,NEG,NEU} (for exploratory datasets)")
    ap.add_argument("--allow_extra_langs", action="store_true", help="Allow language values outside --allowed_langs")
    ap.add_argument("--output_clean", help="Optional path to write a cleaned CSV with labels uppercased")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input CSV not found: {inp}")

    try:
        rows, fields = read_csv_rows(inp)
    except UnicodeDecodeError as e:
        raise SystemExit(f"Failed to read CSV as UTF-8: {e}")

    msgs = validate(
        rows,
        fields,
        allow_extra_labels=bool(args.allow_extra_labels),
        group_column=args.group_column,
        lang_column=args.lang_column,
        split_column=args.split_column,
        allowed_langs=args.allowed_langs,
        allow_extra_langs=bool(args.allow_extra_langs),
    )
    summary = summarize(
        rows,
        group_column=args.group_column,
        lang_column=args.lang_column,
        split_column=args.split_column,
        allowed_langs=args.allowed_langs,
    )

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
