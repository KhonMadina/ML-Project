#!/usr/bin/env python3
"""
Adjudication utilities for Khmer sentiment annotation (POS/NEG/NEU).

Features:
- Load annotations from CSV (id,text,annotator,label) or Label Studio JSON export.
- Detect disagreements between two annotators and generate an adjudication queue.
- Optionally insert Markdown stubs into annotation/adjudication_log.md under the `## Decisions` section.
- Compute inter-annotator agreement (Cohen's kappa) and a confusion matrix summary.

CSV expected columns:
  id, text, annotator, label
Label values are normalized to upper-case among {POS, NEG, NEU}.

Label Studio JSON: best-effort parsing. Assumes Choices from_name="label", to_name="text" as in label_config.json.
Annotator identity is taken from `completed_by` (if available) else annotation id.

Usage examples:
  python annotation/adjudicate.py queue --input annotatorA.csv --input2 annotatorB.csv --log annotation/adjudication_log.md --limit 50
  python annotation/adjudicate.py queue --input export.json --log annotation/adjudication_log.md
  python annotation/adjudicate.py iaa --input annotatorA.csv --input2 annotatorB.csv

Note: This script inserts Markdown stubs only for items not already present in the log (matching "Item ID: <id>").
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

VALID_LABELS = {"POS", "NEG", "NEU"}

@dataclass
class Ann:
    id: str
    text: str
    annotator: str
    label: str


def normalize_label(lbl: str) -> str:
    if lbl is None:
        return ""
    lbl = str(lbl).strip().upper()
    # Common variants
    if lbl in {"POSITIVE", "+"}: return "POS"
    if lbl in {"NEGATIVE", "-"}: return "NEG"
    if lbl in {"NEUTRAL", "0"}: return "NEU"
    return lbl


def load_csv(path: Path) -> List[Ann]:
    rows: List[Ann] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "text", "annotator", "label"}
        missing = required - set(h.strip() for h in reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Expected {required}")
        for r in reader:
            label = normalize_label(r["label"])
            if label not in VALID_LABELS:
                # Skip invalid labels but warn
                # print(f"Warning: skipping row with invalid label: {label} (id={r['id']})")
                continue
            rows.append(Ann(id=str(r["id"]).strip(),
                            text=str(r["text"]).strip(),
                            annotator=str(r["annotator"]).strip(),
                            label=label))
    return rows


def _extract_label_from_ls_result(result: dict) -> Optional[str]:
    """Robustly extract a label from a Label Studio result entry.
    Supports variations where choices may be a list or single string and minor schema differences.
    """
    try:
        rtype = result.get("type") or result.get("result_type")
        if rtype and str(rtype).lower() != "choices":
            return None
        val = result.get("value") or {}
        choices = val.get("choices")
        if isinstance(choices, list) and choices:
            return normalize_label(str(choices[0]))
        if isinstance(choices, str) and choices.strip():
            return normalize_label(choices)
        # Some exports might use a direct 'labels' field
        labels = val.get("labels")
        if isinstance(labels, list) and labels:
            return normalize_label(str(labels[0]))
    except Exception:
        pass
    return None


def load_label_studio_json(path: Path) -> List[Ann]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Label Studio export may be either a list of tasks or an object with a 'tasks' key
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError("Unrecognized Label Studio export structure")

    anns: List[Ann] = []
    skipped_no_result = 0
    skipped_no_label = 0

    for t in tasks:
        tid = str(t.get("id") or t.get("task_id") or t.get("pk") or t.get("uuid") or t.get("data_id") or "").strip()
        # text can live under data.text or at top-level
        text = t.get("data", {}).get("text") or t.get("text") or ""

        # annotations list can be under 'annotations' (newer) or 'completions' (older)
        ann_list = t.get("annotations")
        if not ann_list:
            ann_list = t.get("completions")
        if not isinstance(ann_list, list):
            skipped_no_result += 1
            continue

        for a in ann_list:
            # Annotator identity
            annotator = None
            cb = a.get("completed_by")
            if isinstance(cb, dict):
                annotator = cb.get("username") or cb.get("email") or cb.get("id")
            elif cb is not None:
                annotator = cb
            annotator = str(annotator or a.get("updated_by") or a.get("created_username") or a.get("id") or "anon").strip()

            # Results may be under 'result' or 'results'
            results = a.get("result")
            if not results:
                results = a.get("results")
            if not isinstance(results, list):
                skipped_no_result += 1
                continue

            label = None
            for res in results:
                cand = _extract_label_from_ls_result(res)
                if cand in VALID_LABELS:
                    label = cand
                    break
            if label not in VALID_LABELS:
                skipped_no_label += 1
                continue

            anns.append(Ann(id=tid, text=str(text).strip(), annotator=str(annotator), label=label))

    if skipped_no_result or skipped_no_label:
        print((
            f"Label Studio JSON parsing: parsed={len(anns)} skipped_no_result={skipped_no_result} skipped_no_label={skipped_no_label}"
        ))

    return anns


def load_annotations(path: Path) -> List[Ann]:
    ext = path.suffix.lower()
    if ext == ".csv":
        return load_csv(path)
    if ext == ".json":
        return load_label_studio_json(path)
    raise ValueError(f"Unsupported input extension: {ext}")


def group_by_item(anns: List[Ann]) -> Dict[str, Dict[str, Ann]]:
    grouped: Dict[str, Dict[str, Ann]] = defaultdict(dict)
    for a in anns:
        grouped[a.id][a.annotator] = a
    return grouped


def merge_two_sources(anns_a: List[Ann], anns_b: Optional[List[Ann]]) -> Dict[str, Dict[str, Ann]]:
    if anns_b is None:
        # Assume anns_a contains both annotators
        return group_by_item(anns_a)
    grouped: Dict[str, Dict[str, Ann]] = defaultdict(dict)
    for a in anns_a:
        grouped[a.id][a.annotator] = a
    for b in anns_b:
        grouped[b.id][b.annotator] = b
    return grouped


def disagreements(grouped: Dict[str, Dict[str, Ann]]) -> List[Tuple[str, str, str, str, str]]:
    """
    Return list of tuples: (id, text, annotator_a, label_a, annotator_b, label_b)
    If more than two annotators exist for an id, include all unique unordered pairs.
    """
    out: List[Tuple[str, str, str, str, str]] = []
    for iid, anns in grouped.items():
        # All annotators
        ann_list = list(anns.values())
        n = len(ann_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ann_list[i], ann_list[j]
                if a.label != b.label:
                    # Prefer the longer non-empty text if texts differ
                    text = a.text if len(a.text) >= len(b.text) else b.text
                    out.append((iid, text, a.annotator, a.label, b.annotator, b.label))
    return out


def agreements(grouped: Dict[str, Dict[str, Ann]]) -> List[Tuple[str, str, str, str, str]]:
    out: List[Tuple[str, str, str, str, str]] = []
    for iid, anns in grouped.items():
        ann_list = list(anns.values())
        n = len(ann_list)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ann_list[i], ann_list[j]
                if a.label == b.label:
                    text = a.text if len(a.text) >= len(b.text) else b.text
                    out.append((iid, text, a.annotator, a.label, b.annotator, b.label))
    return out


def compute_kappa(pairs: List[Tuple[str, str]]) -> float:
    """
    pairs: list of (label_a, label_b)
    Cohen's kappa for nominal labels.
    """
    if not pairs:
        return float("nan")
    labels = ["POS", "NEG", "NEU"]
    idx = {l: i for i, l in enumerate(labels)}
    # Confusion counts
    C = [[0] * 3 for _ in range(3)]
    for la, lb in pairs:
        if la not in idx or lb not in idx:
            continue
        C[idx[la]][idx[lb]] += 1
    N = sum(sum(row) for row in C)
    if N == 0:
        return float("nan")
    po = sum(C[i][i] for i in range(3)) / N
    row_sums = [sum(C[i]) for i in range(3)]
    col_sums = [sum(C[i][j] for i in range(3)) for j in range(3)]
    pe = sum((row_sums[i] / N) * (col_sums[i] / N) for i in range(3))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def format_markdown_stub(item_id: str, text: str, label_a: str, label_b: str) -> str:
    return (
        f"- Item ID: {item_id}\n"
        f"- Text (de-identified): {text}\n"
        f"- Label A: {label_a}\n"
        f"- Label B: {label_b}\n"
        f"- Final Label: <POS/NEG/NEU>\n"
        f"- Rationale: <why chosen>\n"
        f"- Edge Cases: <e.g., sarcasm, negation>\n"
        f"- Action Items: <update guidelines/add example>\n"
        f"- Date: <YYYY-MM-DD>\n\n"
    )


def insert_into_adjudication_log(log_path: Path, stubs: List[Tuple[str, str, str, str]] ) -> int:
    """
    Insert stubs under the first occurrence of '## Decisions'. Avoid duplicates based on 'Item ID: <id>'.
    Returns number of inserted items.
    """
    content = log_path.read_text(encoding="utf-8")
    # Ensure Decisions section exists
    if "## Decisions" not in content:
        # Prepend a Decisions section
        content = re.sub(r"(---\s*\n)\s*Example Entry", r"\1\n## Decisions\n\nExample Entry", content, count=1)
    existing_ids = set(re.findall(r"Item ID:\s*(\S+)", content))

    insertion_point = content.find("## Decisions")
    if insertion_point == -1:
        insertion_point = 0
    # Find the position just after the Decisions header line
    after_header_pos = content.find("\n", insertion_point)
    if after_header_pos == -1:
        after_header_pos = len(content)
    after_header_pos += 1  # start on a new line

    new_entries = []
    for item_id, text, la, lb in stubs:
        if item_id in existing_ids:
            continue
        new_entries.append(format_markdown_stub(item_id, text, la, lb))

    if not new_entries:
        return 0

    updated = content[:after_header_pos] + "".join(new_entries) + content[after_header_pos:]
    log_path.write_text(updated, encoding="utf-8")
    return len(new_entries)


def run_queue(args: argparse.Namespace) -> None:
    in1 = Path(args.input)
    anns1 = load_annotations(in1)
    anns2 = None
    if args.input2:
        anns2 = load_annotations(Path(args.input2))

    grouped = merge_two_sources(anns1, anns2)
    dis = disagreements(grouped)

    if args.limit and args.limit > 0:
        dis = dis[: args.limit]

    # Write CSV if requested
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "text", "annotator_a", "label_a", "annotator_b", "label_b"])
            for row in dis:
                w.writerow(row)
        print(f"Wrote disagreements: {outp}")

    # Insert Markdown stubs into adjudication log
    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        stubs = [(iid, text, la, lb) for (iid, text, _, la, _, lb) in dis]
        inserted = insert_into_adjudication_log(log_path, stubs)
        print(f"Inserted {inserted} new adjudication stubs into {log_path}")

    print(f"Found {len(dis)} disagreements")


def run_iaa(args: argparse.Namespace) -> None:
    in1 = Path(args.input)
    anns1 = load_annotations(in1)
    anns2 = None
    if args.input2:
        anns2 = load_annotations(Path(args.input2))

    grouped = merge_two_sources(anns1, anns2)
    # Build pairs for exactly two annotations per item (use first two encountered)
    pairs: List[Tuple[str, str]] = []
    label_pairs_counter: Counter = Counter()
    for iid, by_annot in grouped.items():
        if len(by_annot) < 2:
            continue
        items = list(by_annot.values())
        a, b = items[0], items[1]
        pairs.append((a.label, b.label))
        label_pairs_counter[(a.label, b.label)] += 1

    kappa = compute_kappa(pairs)
    total = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    print(f"Items compared: {total}")
    print(f"Agreements: {agree} ({(agree/total*100) if total else 0:.1f}%)")
    print(f"Cohen's kappa: {kappa:.3f}")

    # Confusion-like summary
    labels = ["POS", "NEG", "NEU"]
    print("\nPair counts (A->B):")
    for la in labels:
        row = []
        for lb in labels:
            row.append(str(label_pairs_counter.get((la, lb), 0)))
        print(f"  {la}: \t" + "\t".join(row))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adjudication utilities for Khmer sentiment annotation")
    sub = p.add_subparsers(dest="cmd", required=True)

    pq = sub.add_parser("queue", help="Generate disagreement queue; optionally insert into adjudication log")
    pq.add_argument("--input", required=True, help="Path to annotations (CSV or Label Studio JSON)")
    pq.add_argument("--input2", help="Optional second annotations file (CSV or JSON)")
    pq.add_argument("--output", help="Optional CSV to write disagreements")
    pq.add_argument("--log", help="Path to adjudication log (Markdown) to insert stubs under '## Decisions'")
    pq.add_argument("--limit", type=int, default=0, help="Limit number of disagreements processed/inserted")
    pq.set_defaults(func=run_queue)

    pi = sub.add_parser("iaa", help="Compute inter-annotator agreement (Cohen's kappa)")
    pi.add_argument("--input", required=True, help="Path to annotations (CSV or Label Studio JSON)")
    pi.add_argument("--input2", help="Optional second annotations file (CSV or JSON)")
    pi.set_defaults(func=run_iaa)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
