#!/usr/bin/env python3
"""
Curate a review queue from model misclassifications to close the loop back into annotation.

Inputs:
- misclassified.csv from modeling/error_analysis.py (requires columns: id,text,label,pred_label; optional: proba_POS,proba_NEG,proba_NEU,confidence,margin)

Outputs:
- A review CSV with prioritized items: id,text,gold_label,model_label,confidence,margin
- Optionally, inserts Markdown stubs into annotation/adjudication_log.md under the '## Decisions' section.

Prioritization strategies:
- uncertain: smallest margin first (most ambiguous)
- confident_wrong: highest confidence first (model strongly wrong)
- random: random shuffle

Usage examples:
  python annotation/curate_from_errors.py \
    --errors reports/baseline_chargram/misclassified.csv \
    --output_csv annotation/review_queue.csv \
    --strategy uncertain \
    --limit 50 \
    --insert_log annotation/adjudication_log.md

  python annotation/curate_from_errors.py --errors reports/baseline_chargram/misclassified.csv --strategy confident_wrong --limit 30 --output_csv annotation/review_queue.csv

Notes:
- Duplicate prevention for adjudication log uses 'Item ID: <id>' matching.
- If margin or confidence are missing in input CSV, they will be computed from proba_ columns if available.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class ErrRow:
    id: str
    text: str
    gold: str
    pred: str
    proba_pos: float
    proba_neg: float
    proba_neu: float
    confidence: float
    margin: float


def parse_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def read_errors(path: Path) -> List[ErrRow]:
    rows: List[ErrRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "text", "label", "pred_label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"errors CSV missing required columns {required}. Got {reader.fieldnames}")
        for r in reader:
            proba_pos = parse_float(r.get("proba_POS"), 0.0)
            proba_neg = parse_float(r.get("proba_NEG"), 0.0)
            proba_neu = parse_float(r.get("proba_NEU"), 0.0)
            confidence = parse_float(r.get("confidence"), 0.0)
            margin = parse_float(r.get("margin"), 0.0)
            # compute if missing and probabilities available
            if (confidence == 0.0 or margin == 0.0) and any([proba_pos, proba_neg, proba_neu]):
                probs = sorted([proba_pos, proba_neg, proba_neu], reverse=True)
                if probs:
                    confidence = probs[0]
                    margin = probs[0] - (probs[1] if len(probs) > 1 else 0.0)
            rows.append(ErrRow(
                id=str(r.get("id", "")).strip(),
                text=str(r.get("text", "")).strip(),
                gold=str(r.get("label", "")).strip().upper(),
                pred=str(r.get("pred_label", "")).strip().upper(),
                proba_pos=proba_pos,
                proba_neg=proba_neg,
                proba_neu=proba_neu,
                confidence=confidence,
                margin=margin,
            ))
    return rows


def prioritize(rows: List[ErrRow], strategy: str) -> List[ErrRow]:
    rows = rows.copy()
    if strategy == "uncertain":
        rows.sort(key=lambda x: (x.margin, -x.confidence))  # smallest margin first
    elif strategy == "confident_wrong":
        rows.sort(key=lambda x: (-x.confidence, -x.margin))
    elif strategy == "random":
        random.shuffle(rows)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return rows


def write_review_csv(path: Path, rows: List[ErrRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "gold_label", "model_label", "confidence", "margin"])
        for r in rows:
            w.writerow([r.id, r.text, r.gold, r.pred, f"{r.confidence:.6f}", f"{r.margin:.6f}"])


def format_md_stub(r: ErrRow) -> str:
    return (
        f"- Item ID: {r.id}\n"
        f"- Text (de-identified): {r.text}\n"
        f"- Label A: {r.gold}\n"
        f"- Label B: {r.pred} (model)\n"
        f"- Final Label: <POS/NEG/NEU>\n"
        f"- Rationale: <why chosen / model error analysis>\n"
        f"- Edge Cases: <e.g., sarcasm, negation, code-mix>\n"
        f"- Action Items: <add example/update guidelines/augment data>\n"
        f"- Date: <YYYY-MM-DD>\n\n"
    )


def insert_into_adjudication_log(log_path: Path, rows: List[ErrRow]) -> int:
    content = log_path.read_text(encoding="utf-8")
    if "## Decisions" not in content:
        content = content + "\n\n## Decisions\n\n"
    existing_ids = set(re.findall(r"Item ID:\s*(\S+)", content))
    insertion_point = content.find("## Decisions")
    after_header_pos = content.find("\n", insertion_point)
    if after_header_pos == -1:
        after_header_pos = len(content)
    after_header_pos += 1

    stubs = []
    for r in rows:
        if r.id in existing_ids:
            continue
        stubs.append(format_md_stub(r))
    if not stubs:
        return 0
    updated = content[:after_header_pos] + "".join(stubs) + content[after_header_pos:]
    log_path.write_text(updated, encoding="utf-8")
    return len(stubs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Curate review queue from model misclassifications and optionally insert into adjudication log")
    ap.add_argument("--errors", required=True, help="Path to misclassified.csv from error_analysis.py")
    ap.add_argument("--output_csv", help="Where to write the review queue CSV")
    ap.add_argument("--strategy", choices=["uncertain", "confident_wrong", "random"], default="uncertain")
    ap.add_argument("--limit", type=int, default=50, help="Max number of items to include")
    ap.add_argument("--insert_log", help="Path to annotation/adjudication_log.md to insert stubs under '## Decisions'")
    args = ap.parse_args()

    err_path = Path(args.errors)
    rows = read_errors(err_path)
    if not rows:
        print("No rows found in errors file.")
        return
    prioritized = prioritize(rows, args.strategy)
    if args.limit > 0:
        prioritized = prioritized[: args.limit]

    if args.output_csv:
        write_review_csv(Path(args.output_csv), prioritized)
        print(f"Wrote review queue to {args.output_csv}")

    if args.insert_log:
        lp = Path(args.insert_log)
        if not lp.exists():
            raise FileNotFoundError(f"Log file not found: {lp}")
        inserted = insert_into_adjudication_log(lp, prioritized)
        print(f"Inserted {inserted} review stubs into {lp}")

    print(f"Prepared {len(prioritized)} items for review (strategy={args.strategy}).")


if __name__ == "__main__":
    main()
