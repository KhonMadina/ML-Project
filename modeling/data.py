from __future__ import annotations

"""Data loading and splitting utilities for Khmer/English sentiment experiments.

This module centralizes CSV reading and train/val/test split logic so that
training scripts (baseline, transformer) can share a consistent pipeline.

Enhancements over the initial version:
- Optional language-aware parsing (lang column) and split/source metadata
- Optional stratification by (label, lang)
- Group-aware stratified splitting extended to consider (label, lang)
- Helpers to load pre-defined splits from a single CSV with a split column
- Backward-compatible defaults: if you don't pass lang/split columns, behavior
  remains identical to the previous version.
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    pd = None

try:  # optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    np = None

try:
    from sklearn.model_selection import GroupShuffleSplit  # type: ignore
except Exception:  # pragma: no cover - import error will surface when used
    GroupShuffleSplit = None  # type: ignore


@dataclass
class Row:
    """Single labeled example from the sentiment dataset.

    Fields are optional except id/text/label to maintain backwards compat.
    """

    id: str
    text: str
    label: str
    group: Optional[str] = None
    lang: Optional[str] = None
    split: Optional[str] = None
    source: Optional[str] = None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _normalize_label(s: str) -> str:
    return (s or "").strip().upper()


def _normalize_opt(s: object) -> Optional[str]:
    if s is None:
        return None
    try:
        import math

        if isinstance(s, float) and math.isnan(s):  # type: ignore
            return None
    except Exception:
        pass
    s2 = str(s).strip()
    return s2 if s2 != "" else None


def read_csv_rows(
    path: Path,
    group_col: Optional[str] = None,
    *,
    lang_col: Optional[str] = None,
    split_col: Optional[str] = None,
    source_col: Optional[str] = None,
) -> List[Row]:
    """Read rows from a CSV file.

    Backward compatible with the original signature: only 'path' and
    'group_col' are positional. Additional metadata columns are optional.
    """

    required = {"id", "text", "label"}

    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns {required}. Got {list(df.columns)}")
        rows: List[Row] = []
        for _, r in df.iterrows():
            rows.append(
                Row(
                    id=str(r["id"]),
                    text=str(r["text"]),
                    label=_normalize_label(str(r["label"])),
                    group=_normalize_opt(r[group_col]) if (group_col and group_col in df.columns) else None,
                    lang=_normalize_opt(r[lang_col]) if (lang_col and lang_col in df.columns) else None,
                    split=_normalize_opt(r[split_col]) if (split_col and split_col in df.columns) else None,
                    source=_normalize_opt(r[source_col]) if (source_col and source_col in df.columns) else None,
                )
            )
        return rows

    # Fallback to csv module when pandas is unavailable
    rows2: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        flds = set(reader.fieldnames or [])
        if not required.issubset(flds):
            raise ValueError(f"CSV missing required columns {required}. Got {reader.fieldnames}")
        for r in reader:
            rows2.append(
                Row(
                    id=str(r["id"]),
                    text=str(r["text"]),
                    label=_normalize_label(str(r["label"])),
                    group=_normalize_opt(r.get(group_col)) if group_col else None,
                    lang=_normalize_opt(r.get(lang_col)) if lang_col else None,
                    split=_normalize_opt(r.get(split_col)) if split_col else None,
                    source=_normalize_opt(r.get(source_col)) if source_col else None,
                )
            )
    return rows2


def load_with_splits(
    path: Path,
    group_col: Optional[str] = None,
    *,
    split_col: str = "split",
    lang_col: Optional[str] = None,
    source_col: Optional[str] = None,
) -> Tuple[List[Row], List[Row], List[Row]]:
    """Load a single CSV that already contains a split column.

    Returns train, val, test lists. Rows without a valid split are ignored.
    """
    rows = read_csv_rows(path, group_col, lang_col=lang_col, split_col=split_col, source_col=source_col)
    tr: List[Row] = []
    va: List[Row] = []
    te: List[Row] = []
    for r in rows:
        sp = (r.split or "").lower()
        if sp == "train":
            tr.append(r)
        elif sp == "val" or sp == "valid" or sp == "validation":
            va.append(r)
        elif sp == "test":
            te.append(r)
    return tr, va, te


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------


def _by_key(items: Iterable[Row], key_fn) -> Dict[str, List[Row]]:
    d: Dict[str, List[Row]] = defaultdict(list)
    for r in items:
        d[key_fn(r)].append(r)
    return d


def stratified_split(
    rows: List[Row],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    *,
    stratify_by_lang: bool = False,
) -> Tuple[List[Row], List[Row], List[Row]]:
    """Stratified split by label or (label,lang).

    Backward compatible: by default stratifies by label only.
    """

    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    random.seed(seed)

    if stratify_by_lang:
        def key_fn(r: Row) -> str:
            return f"{r.label}||{(r.lang or 'unk').lower()}"
    else:
        def key_fn(r: Row) -> str:
            return r.label

    by_bucket: Dict[str, List[Row]] = _by_key(rows, key_fn)

    train: List[Row] = []
    val: List[Row] = []
    test: List[Row] = []
    for _, items in by_bucket.items():
        items = items.copy()
        random.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # ensure sum to n
        if n_train + n_val > n:
            overflow = (n_train + n_val) - n
            reduce_val = min(overflow, max(0, n_val))
            n_val -= reduce_val
            overflow -= reduce_val
            if overflow > 0:
                n_train -= overflow
        n_test = n - n_train - n_val
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    return train, val, test


def group_aware_split(
    rows: List[Row],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[Row], List[Row], List[Row]]:
    """Group-aware split using GroupShuffleSplit to avoid leakage.

    Matches prior behavior; does not stratify by label/lang. For stratified
    group-aware allocation use group_stratified_split(..., stratify_by_lang=True).
    """

    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if np is None:
        raise SystemExit("NumPy is required for group-aware split. Install with: pip install numpy")
    if GroupShuffleSplit is None:
        raise SystemExit("scikit-learn is required for group-aware split. Install with: pip install scikit-learn")

    X = np.arange(len(rows))
    y = np.array([r.label for r in rows])  # not used by GroupShuffleSplit, kept for clarity
    groups = np.array([r.group if r.group is not None else f"__nogroup_{i}" for i, r in enumerate(rows)])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    # Compute proportion of test within temp
    temp_size = val_ratio + test_ratio
    test_within_temp = (test_ratio / temp_size) if temp_size > 0 else 0.0

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_within_temp, random_state=seed + 1)
    val_idx, test_idx = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))

    # Map indices back
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]
    return train_rows, val_rows, test_rows


def group_stratified_split(
    rows: List[Row],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    *,
    stratify_by_lang: bool = False,
) -> Tuple[List[Row], List[Row], List[Row]]:
    """Approximate group-aware stratified split.

    If stratify_by_lang is True, targets and costs are computed over buckets
    of (label, lang) rather than label only. This preserves both label and
    language composition per split while avoiding group leakage.
    """

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Collect groups and per-group bucket counts
    groups: Dict[str, List[Row]] = {}
    for i, r in enumerate(rows):
        g = r.group if r.group is not None else f"__nogroup_{i}"
        groups.setdefault(g, []).append(r)

    def bucket_key(r: Row) -> str:
        if stratify_by_lang:
            return f"{r.label}||{(r.lang or 'unk').lower()}"
        return r.label

    total_bucket_counts: Dict[str, int] = {}
    for r in rows:
        k = bucket_key(r)
        total_bucket_counts[k] = total_bucket_counts.get(k, 0) + 1

    def target_counts(ratio: float) -> Dict[str, float]:
        return {b: total_bucket_counts.get(b, 0) * ratio for b in total_bucket_counts}

    targets = {
        "train": target_counts(train_ratio),
        "val": target_counts(val_ratio),
        "test": target_counts(test_ratio),
    }

    # Current counts per split
    cur = {
        "train": {b: 0 for b in total_bucket_counts},
        "val": {b: 0 for b in total_bucket_counts},
        "test": {b: 0 for b in total_bucket_counts},
    }

    # Deterministic order: by group size desc, then name asc
    rng = random.Random(seed)
    items = list(groups.items())
    items.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    assign: Dict[str, str] = {}

    def bucket_counts_for_group(grows: List[Row]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for r in grows:
            k = bucket_key(r)
            d[k] = d.get(k, 0) + 1
        return d

    for g, grows in items:
        gc = bucket_counts_for_group(grows)
        # Evaluate cost for assigning to each split
        best_split: Optional[str] = None
        best_cost: Optional[float] = None
        for sp in ["train", "val", "test"]:
            cost = 0.0
            for b, tgt in targets[sp].items():
                after = cur[sp].get(b, 0) + gc.get(b, 0)
                cost += abs(after - tgt)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = sp
        assign[g] = best_split or "train"
        # Update counts
        for b, c in gc.items():
            cur[assign[g]][b] = cur[assign[g]].get(b, 0) + c

    train_rows: List[Row] = []
    val_rows: List[Row] = []
    test_rows: List[Row] = []
    for g, grows in groups.items():
        sp = assign[g]
        if sp == "train":
            train_rows.extend(grows)
        elif sp == "val":
            val_rows.extend(grows)
        else:
            test_rows.extend(grows)

    return train_rows, val_rows, test_rows


# ---------------------------------------------------------------------------
# Helper conversions and summaries
# ---------------------------------------------------------------------------


def to_xy(rows: List[Row]):
    """Convert a list of Rows into (texts, labels) lists.

    Kept for compatibility with existing training code.
    """

    X = [r.text for r in rows]
    y = [r.label for r in rows]
    return X, y


def distribution_by_label(rows: List[Row]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for r in rows:
        d[r.label] = d.get(r.label, 0) + 1
    return d


def distribution_by_label_lang(rows: List[Row]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for r in rows:
        key = f"{r.label}||{(r.lang or 'unk').lower()}"
        d[key] = d.get(key, 0) + 1
    return d
