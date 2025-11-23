from __future__ import annotations

"""Data loading and splitting utilities for Khmer sentiment experiments.

This module centralizes CSV reading and train/val/test split logic so that
training scripts (baseline, transformer) can share a consistent pipeline.

The initial version is a direct extraction of logic from train_baseline.py
with minimal refactoring to avoid behavior changes.
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    Attributes mirror the existing Row in train_baseline.py so current
    scripts can be migrated incrementally.
    """

    id: str
    text: str
    label: str
    group: Optional[str] = None


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def read_csv_rows(path: Path, group_col: Optional[str] = None) -> List[Row]:
    """Read rows from a CSV file.

    This reproduces the behavior from train_baseline.read_csv_rows so that
    downstream behavior is unchanged while giving us a central entry point
    for future extensions (e.g., additional metadata columns).
    """

    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        required = {"id", "text", "label"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns {required}. Got {list(df.columns)}")
        rows: List[Row] = []
        for _, r in df.iterrows():
            group_val = None
            if group_col and group_col in df.columns:
                gv = r[group_col]
                if gv is None:
                    group_val = None
                else:
                    try:
                        import math

                        if isinstance(gv, float) and math.isnan(gv):
                            group_val = None
                        else:
                            group_val = str(gv)
                    except Exception:
                        group_val = str(gv)
            rows.append(Row(str(r["id"]), str(r["text"]), str(r["label"]).upper(), group_val))
        return rows

    # Fallback to csv module when pandas is unavailable
    rows2: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "text", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns {required}. Got {reader.fieldnames}")
        for r in reader:
            group_val = None
            if group_col and group_col in (reader.fieldnames or []):
                gv = r.get(group_col)
                group_val = str(gv) if gv is not None and str(gv).strip() != "" else None
            rows2.append(Row(str(r["id"]), str(r["text"]), str(r["label"]).upper(), group_val))
    return rows2


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------


def stratified_split(
    rows: List[Row],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[Row], List[Row], List[Row]]:
    """Stratified split by label.

    Identical to the implementation formerly in train_baseline.py.
    """

    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    random.seed(seed)
    by_label: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_label[r.label].append(r)
    train: List[Row] = []
    val: List[Row] = []
    test: List[Row] = []
    for _, items in by_label.items():
        items = items.copy()
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # ensure all remaining examples go to test split to keep total fixed
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

    This matches the previous implementation in train_baseline.group_aware_split.
    """

    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 <= test_ratio < 1
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if np is None:
        raise SystemExit("NumPy is required for group-aware split. Install with: pip install numpy")
    if GroupShuffleSplit is None:
        raise SystemExit("scikit-learn is required for group-aware split. Install with: pip install scikit-learn")

    X = np.arange(len(rows))
    y = np.array([r.label for r in rows])  # not used by GroupShuffleSplit, but kept for clarity
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
) -> Tuple[List[Row], List[Row], List[Row]]:
    """Approximate group-aware stratified split.

    Directly ported from train_baseline.group_stratified_split so behavior
    remains identical while making it reusable from other scripts.
    """

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Collect groups and per-group label counts
    groups: Dict[str, List[Row]] = {}
    for i, r in enumerate(rows):
        g = r.group if r.group is not None else f"__nogroup_{i}"
        groups.setdefault(g, []).append(r)

    total_label_counts: Dict[str, int] = {}
    for r in rows:
        total_label_counts[r.label] = total_label_counts.get(r.label, 0) + 1

    def target_counts(ratio: float) -> Dict[str, float]:
        return {lbl: total_label_counts.get(lbl, 0) * ratio for lbl in total_label_counts}

    targets = {
        "train": target_counts(train_ratio),
        "val": target_counts(val_ratio),
        "test": target_counts(test_ratio),
    }

    # Current counts per split
    cur = {
        "train": {lbl: 0 for lbl in total_label_counts},
        "val": {lbl: 0 for lbl in total_label_counts},
        "test": {lbl: 0 for lbl in total_label_counts},
    }

    # Deterministic order: by group size desc, then name asc, with seeded shuffle for ties
    rng = random.Random(seed)
    items = list(groups.items())
    # Note: rng not currently used for tie-breaking, but kept for future extension
    items.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    assign: Dict[str, str] = {}

    def label_counts_for_group(grows: List[Row]) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for r in grows:
            d[r.label] = d.get(r.label, 0) + 1
        return d

    for g, grows in items:
        gc = label_counts_for_group(grows)
        # Evaluate cost for assigning to each split
        best_split: Optional[str] = None
        best_cost: Optional[float] = None
        for sp in ["train", "val", "test"]:
            cost = 0.0
            for lbl, tgt in targets[sp].items():
                after = cur[sp].get(lbl, 0) + gc.get(lbl, 0)
                cost += abs(after - tgt)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = sp
        assign[g] = best_split or "train"
        # Update counts
        for lbl, c in gc.items():
            cur[assign[g]][lbl] = cur[assign[g]].get(lbl, 0) + c

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
# Helper conversion
# ---------------------------------------------------------------------------


def to_xy(rows: List[Row]):
    """Convert a list of Rows into (texts, labels) lists.

    Kept for compatibility with existing training code.
    """

    X = [r.text for r in rows]
    y = [r.label for r in rows]
    return X, y
