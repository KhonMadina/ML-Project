#!/usr/bin/env python3
"""
Ingest a new annotated dataset into the Khmer+English Sentiment pipeline.

Steps:
1) Validate the input CSV schema and basic quality (bilingual-aware)
2) Run the finalization pipeline to produce final_dataset.csv
3) Optionally export leakage-safe splits (final_train/val/test.csv)

Expected input CSV columns: id,text,label[,lang,group,split,source]
Labels are normalized to POS/NEG/NEU by the finalization stage.

Examples:
  # Minimal (no groups, export splits)
  python tools/ingest_dataset.py \
    --raw_csv path/to/annotations.csv \
    --output_dir annotation/sample_data \
    --export_splits

  # With group metadata and group-aware splits
  python tools/ingest_dataset.py \
    --raw_csv path/to/annotations.csv \
    --groups_csv annotation/sample_data/groups.csv \
    --group_column group \
    --output_dir annotation/sample_data \
    --export_splits --group_aware_splits

  # With language column and allowed language set
  python tools/ingest_dataset.py \
    --raw_csv path/to/annotations.csv \
    --lang_column lang --allowed_langs km en \
    --output_dir annotation/sample_data
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest new annotated dataset (validate + finalize + export splits)")
    ap.add_argument("--raw_csv", required=True, help="Path to raw annotations CSV with id,text,label[,lang,group,split,source]")
    ap.add_argument("--output_dir", required=True, help="Directory where outputs will be written")
    ap.add_argument("--groups_csv", help="Optional path to CSV mapping id -> group (or id,<group_column>)")
    ap.add_argument("--group_column", default="group", help="Group column name in groups_csv (default: group)")
    ap.add_argument("--lang_column", default=None, help="Optional language column name (e.g., 'lang')")
    ap.add_argument("--allowed_langs", nargs="*", default=None, help="List of allowed language codes (e.g., km en)")
    ap.add_argument("--split_column", default=None, help="Optional split column name ('train'|'val'|'test')")
    ap.add_argument("--export_splits", action="store_true", help="Export final_train/val/test.csv next to final_dataset.csv")
    ap.add_argument("--group_aware_splits", action="store_true", help="Use group-aware splitting if group metadata is provided")
    ap.add_argument("--allow_extra_labels", action="store_true", help="Allow labels outside {POS,NEG,NEU} during validation")
    ap.add_argument("--allow_extra_langs", action="store_true", help="Allow language values outside --allowed_langs during validation")
    args = ap.parse_args()

    raw_csv = Path(args.raw_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Validate dataset (write a cleaned CSV with uppercased labels)
    cleaned_csv = out_dir / (raw_csv.stem + "_clean.csv")
    val_cmd = [
        sys.executable,
        str(Path("tools") / "validate_dataset.py"),
        "--input", str(raw_csv),
        "--output_clean", str(cleaned_csv),
    ]
    if args.allow_extra_labels:
        val_cmd.append("--allow_extra_labels")
    if args.allow_extra_langs:
        val_cmd.append("--allow_extra_langs")
    # Include optional column checks if provided
    if args.groups_csv:
        val_cmd.extend(["--group_column", str(args.group_column)])
    if args.lang_column:
        val_cmd.extend(["--lang_column", str(args.lang_column)])
    if args.allowed_langs:
        val_cmd.extend(["--allowed_langs", *args.allowed_langs])
    if args.split_column:
        val_cmd.extend(["--split_column", str(args.split_column)])

    rc = run(val_cmd)
    if rc != 0:
        print("Warning: validation script returned a non-zero code; continuing.")

    # 2) Finalize dataset
    final_csv = out_dir / "final_dataset.csv"
    log_md = out_dir / "adjudication_log.md"

    fin_cmd = [
        sys.executable,
        str(Path("annotation") / "finalize_dataset.py"),
        "combine",
        "--input", str(cleaned_csv if cleaned_csv.exists() else raw_csv),
        "--log", str(log_md),
        "--output", str(final_csv),
        "--strategy", "majority",
    ]
    if args.groups_csv:
        fin_cmd.extend(["--groups_csv", str(args.groups_csv), "--group_column", str(args.group_column)])
        if args.group_aware_splits and args.export_splits:
            fin_cmd.append("--group_aware_splits")
    if args.export_splits:
        fin_cmd.append("--export-splits")

    rc2 = run(fin_cmd)
    if rc2 != 0:
        print("ERROR: finalize_dataset.py failed. See output above.")
        sys.exit(rc2)

    print(f"Ingestion complete. Final dataset: {final_csv}")
    if args.export_splits:
        base = final_csv.parent
        print(f"Splits (if requested): {base / 'final_train.csv'}, {base / 'final_val.csv'}, {base / 'final_test.csv'}")


if __name__ == "__main__":
    main()
