#!/usr/bin/env python3
"""
Finalize a single-label dataset for Khmer sentiment (POS/NEG/NEU) by resolving disagreements
using adjudication decisions and configurable fallback strategies. Optionally create
stratified train/val/test splits.

Relies on annotation/adjudicate.py for loading annotations from CSV or Label Studio JSON.

Usage examples:
  # From two annotator CSVs; drop unresolved disagreements
  python annotation/finalize_dataset.py combine \
    --input path/to/annotatorA.csv \
    --input2 path/to/annotatorB.csv \
    --log annotation/adjudication_log.md \
    --output data/processed/final_dataset.csv \
    --strategy drop_unresolved

  # From a single Label Studio JSON; resolve via majority, then split
  python annotation/finalize_dataset.py combine \
    --input path/to/export.json \
    --log annotation/adjudication_log.md \
    --output data/processed/final_dataset.csv \
    --strategy majority \
    --export-splits \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1

  # Use annotator priority (falls back in given order when disagreement unresolved)
  python annotation/finalize_dataset.py combine \
    --input path/to/annotatorA.csv \
    --input2 path/to/annotatorB.csv \
    --log annotation/adjudication_log.md \
    --output data/processed/final_dataset.csv \
    --strategy annotator_priority \
    --annotator-priority "annotator_a,annotator_b"
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional imports for group-aware splitting
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    from sklearn.model_selection import GroupShuffleSplit  # type: ignore
except Exception:
    GroupShuffleSplit = None  # type: ignore

# Import loading utilities from adjudicate.py (same directory)
try:
    from adjudicate import load_annotations, VALID_LABELS, Ann
except Exception as e:
    raise SystemExit(f"Failed to import from adjudicate.py: {e}")


@dataclass
class FinalItem:
    id: str
    text: str
    label: str
    group: Optional[str] = None


def parse_adjudication_log(log_path: Path) -> Dict[str, str]:
    """
    Parse adjudication decisions from the Markdown log more robustly.
    Tolerates different bullet markers (-, *), numeric lists, extra whitespace, and capitalization.
    Returns mapping {item_id: final_label} only for valid labels in {POS, NEG, NEU}.
    Prints warnings for invalid/missing final labels.
    """
    if not log_path.exists():
        return {}

    content = log_path.read_text(encoding="utf-8")
    decisions: Dict[str, str] = {}
    skipped_missing = 0
    skipped_invalid = 0

    # Find blocks starting with a bullet/number and 'Item ID:'; capture until next block or end
    block_re = re.compile(r"(?mis)^[\-\*\d\.]+\s*Item\s*ID:\s*(?P<id>\S+)\s*(?P<body>.*?)(?=^[\-\*\d\.]+\s*Item\s*ID:|\Z)")
    for m in block_re.finditer(content):
        iid = m.group("id").strip()
        body = m.group("body") or ""
        # Find final label line (case-insensitive), allow varied spacing and bullets
        fl = re.search(r"(?im)^\s*[\-\*]?\s*Final\s*Label\s*:\s*([A-Za-z0-9_<>/+\-]+)\s*$", body)
        if not fl:
            skipped_missing += 1
            continue
        raw = fl.group(1).strip()
        lbl = raw.upper()
        # Skip placeholders like <POS/NEG/NEU>
        if '<' in lbl or '>' in lbl:
            skipped_missing += 1
            continue
        # Normalize common variants
        if lbl in {"POSITIVE", "+", "POS-", "+POS"}:
            lbl = "POS"
        elif lbl in {"NEGATIVE", "-", "NEG-", "-NEG"}:
            lbl = "NEG"
        elif lbl in {"NEUTRAL", "0", "NEU-"}:
            lbl = "NEU"
        # Accept only valid labels
        if lbl not in VALID_LABELS:
            skipped_invalid += 1
            continue
        decisions[iid] = lbl

    if skipped_missing or skipped_invalid:
        print((
            f"Adjudication log parsing: applied {len(decisions)} decisions; "
            f"skipped_missing={skipped_missing}, skipped_invalid={skipped_invalid}"
        ))

    return decisions


def choose_text(texts: List[str]) -> str:
    # Pick the longest non-empty text to represent the item
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return ""
    return max(texts, key=lambda x: len(x))


essential_columns = ["id", "text", "label"]


def resolve_labels(
    anns: List[Ann],
    decisions: Dict[str, str],
    strategy: str = "drop_unresolved",
    annotator_priority: Optional[List[str]] = None,
    groups: Optional[Dict[str, str]] = None,
) -> List[FinalItem]:
    grouped: Dict[str, List[Ann]] = defaultdict(list)
    for a in anns:
        grouped[a.id].append(a)

    out: List[FinalItem] = []
    dropped_due_to_unresolved = 0
    for iid, items in grouped.items():
        # gather labels and annotators
        labels = [a.label for a in items if a.label in VALID_LABELS]
        texts = [a.text for a in items]
        text = choose_text(texts)
        grp = (groups or {}).get(iid) if groups else None
        if not labels:
            continue
        unique_labels = set(labels)

        # If decision exists, it overrides
        if iid in decisions:
            out.append(FinalItem(id=iid, text=text, label=decisions[iid], group=grp))
            continue

        if len(unique_labels) == 1:
            out.append(FinalItem(id=iid, text=text, label=labels[0], group=grp))
            continue

        # Disagreement path
        if strategy == "drop_unresolved":
            dropped_due_to_unresolved += 1
            continue
        elif strategy == "majority":
            c = Counter(labels)
            most = c.most_common()
            if len(most) == 1 or (len(most) > 1 and most[0][1] > most[1][1]):
                out.append(FinalItem(id=iid, text=text, label=most[0][0], group=grp))
            else:
                # tie: drop
                dropped_due_to_unresolved += 1
                continue
        elif strategy == "annotator_priority":
            ap = annotator_priority or []
            chosen: Optional[str] = None
            by_annot: Dict[str, str] = {a.annotator: a.label for a in items}
            for name in ap:
                if name in by_annot and by_annot[name] in VALID_LABELS:
                    chosen = by_annot[name]
                    break
            if chosen is not None:
                out.append(FinalItem(id=iid, text=text, label=chosen, group=grp))
            else:
                dropped_due_to_unresolved += 1
                continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if dropped_due_to_unresolved:
        print(f"Dropped {dropped_due_to_unresolved} items due to unresolved disagreements (strategy={strategy}).")
    return out


def write_csv(path: Path, rows: List[FinalItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dynamically include group column if present in any row
    include_group = any(getattr(r, "group", None) for r in rows)
    cols = ["id", "text", "label"] + (["group"] if include_group else [])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            row = [r.id, r.text, r.label]
            if include_group:
                row.append(r.group or "")
            w.writerow(row)


def class_distribution(rows: List[FinalItem]) -> Dict[str, int]:
    c = Counter(r.label for r in rows)
    return {k: c.get(k, 0) for k in ["POS", "NEG", "NEU"]}


def stratified_split(rows: List[FinalItem], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[List[FinalItem], List[FinalItem], List[FinalItem]]:
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    random.seed(seed)
    by_label: Dict[str, List[FinalItem]] = defaultdict(list)
    for r in rows:
        by_label[r.label].append(r)

    train: List[FinalItem] = []
    val: List[FinalItem] = []
    test: List[FinalItem] = []

    for lbl, items in by_label.items():
        items = items.copy()
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def group_aware_split(rows: List[FinalItem], train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42) -> Tuple[List[FinalItem], List[FinalItem], List[FinalItem]]:
    """Perform a two-stage GroupShuffleSplit using FinalItem.group to prevent leakage.
    Falls back to stratified_split if dependencies are missing or groups are absent.
    """
    if np is None or GroupShuffleSplit is None:
        print("Warning: numpy/sklearn missing for group-aware splitting; falling back to stratified split.")
        return stratified_split(rows, train_ratio, val_ratio, test_ratio, seed=seed)

    has_groups = any((r.group or "").strip() != "" for r in rows)
    if not has_groups:
        print("Warning: No group values found; falling back to stratified split.")
        return stratified_split(rows, train_ratio, val_ratio, test_ratio, seed=seed)

    X = np.arange(len(rows))
    y = np.array([r.label for r in rows])  # not used but kept for clarity
    groups = np.array([r.group if (r.group and str(r.group).strip() != "") else f"__nogroup_{i}" for i, r in enumerate(rows)])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    temp_size = val_ratio + test_ratio
    test_within_temp = (test_ratio / temp_size) if temp_size > 0 else 0.0

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_within_temp, random_state=seed + 1)
    val_rel, test_rel = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]
    return train_rows, val_rows, test_rows


def run_combine(args: argparse.Namespace) -> None:
    in1 = Path(args.input)
    anns1 = load_annotations(in1)
    anns2 = None
    if args.input2:
        anns2 = load_annotations(Path(args.input2))
        anns = anns1 + anns2
    else:
        anns = anns1

    decisions = parse_adjudication_log(Path(args.log)) if args.log else {}

    # Optional groups mapping (id -> group)
    groups_map: Optional[Dict[str, str]] = None
    if args.groups_csv:
        groups_map = {}
        gpath = Path(args.groups_csv)
        if not gpath.exists():
            raise SystemExit(f"--groups_csv not found: {gpath}")
        with gpath.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not {"id", args.group_column}.issubset(set(reader.fieldnames or [])):
                raise SystemExit(f"--groups_csv must contain columns: id,{args.group_column}. Found: {reader.fieldnames}")
            for r in reader:
                gid = str(r.get("id", "")).strip()
                gval = str(r.get(args.group_column, "")).strip()
                if gid:
                    groups_map[gid] = gval

    annotator_priority: Optional[List[str]] = None
    if args.annotator_priority:
        annotator_priority = [s.strip() for s in args.annotator_priority.split(',') if s.strip()]

    final_rows = resolve_labels(
        anns,
        decisions,
        strategy=args.strategy,
        annotator_priority=annotator_priority,
        groups=groups_map,
    )

    if not final_rows:
        print("No items to write after resolution. Check inputs, strategy, and adjudication log.")
        return

    out_path = Path(args.output)
    write_csv(out_path, final_rows)

    dist = class_distribution(final_rows)
    print(f"Wrote {len(final_rows)} items to {out_path}")
    print(f"Class distribution: POS={dist['POS']}, NEG={dist['NEG']}, NEU={dist['NEU']}")

    if args.export_splits:
        train_ratio = float(args.train_ratio)
        val_ratio = float(args.val_ratio)
        test_ratio = float(args.test_ratio)
        if getattr(args, "group_aware_splits", False):
            train, val, test = group_aware_split(final_rows, train_ratio, val_ratio, test_ratio, seed=int(args.seed))
        else:
            train, val, test = stratified_split(final_rows, train_ratio, val_ratio, test_ratio, seed=int(args.seed))
        base = out_path.parent
        write_csv(base / "final_train.csv", train)
        write_csv(base / "final_val.csv", val)
        write_csv(base / "final_test.csv", test)
        print(f"Splits written: {base / 'final_train.csv'}, {base / 'final_val.csv'}, {base / 'final_test.csv'}")
        print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finalize dataset by applying adjudication decisions and resolution strategies")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("combine", help="Combine annotations, apply decisions/strategy, write final dataset")
    pc.add_argument("--input", required=True, help="Path to annotations (CSV or Label Studio JSON)")
    pc.add_argument("--input2", help="Optional second annotations file (CSV or JSON)")
    pc.add_argument("--log", help="Path to adjudication log (Markdown) to apply final labels")
    pc.add_argument("--output", required=True, help="Output CSV path for final dataset")
    pc.add_argument("--strategy", choices=["drop_unresolved", "majority", "annotator_priority"], default="drop_unresolved", help="Fallback when disagreements lack decisions")
    pc.add_argument("--annotator-priority", dest="annotator_priority", help="Comma-separated annotator names in priority order (for annotator_priority strategy)")
    pc.add_argument("--groups_csv", help="Optional CSV mapping id -> group to propagate into final dataset (columns: id,<group_column>)")
    pc.add_argument("--group_column", default="group", help="Column name in --groups_csv that contains the group value (default: 'group')")
    pc.add_argument("--export-splits", action="store_true", help="If set, also export train/val/test CSVs")
    pc.add_argument("--group_aware_splits", action="store_true", help="Use group-aware splits when exporting (requires group column in data and numpy/sklearn)")
    pc.add_argument("--train-ratio", default=0.8, type=float)
    pc.add_argument("--val-ratio", default=0.1, type=float)
    pc.add_argument("--test-ratio", default=0.1, type=float)
    pc.add_argument("--seed", default=42, help="Random seed for splitting")
    pc.set_defaults(func=run_combine)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
