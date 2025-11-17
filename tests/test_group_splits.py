import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.finalize_dataset import FinalItem, stratified_split, group_aware_split
from modeling.train_baseline import Row, stratified_split as stratified_split_rows, group_aware_split as group_aware_split_rows


def test_group_aware_split_no_leakage_finalize():
    rows = []
    # create 6 items: groups g1 (3), g2 (2), g3 (1)
    for i in range(3):
        rows.append(FinalItem(id=f"a{i}", text="t", label="POS", group="g1"))
    for i in range(2):
        rows.append(FinalItem(id=f"b{i}", text="t", label="NEG", group="g2"))
    rows.append(FinalItem(id="c0", text="t", label="NEU", group="g3"))

    tr, va, te = group_aware_split(rows, 0.5, 0.25, 0.25, seed=42)
    tr_groups = set(r.group for r in tr)
    va_groups = set(r.group for r in va)
    te_groups = set(r.group for r in te)
    assert tr_groups.isdisjoint(va_groups)
    assert tr_groups.isdisjoint(te_groups)
    assert va_groups.isdisjoint(te_groups)


def test_group_aware_split_rows_no_leakage_modeling():
    rows = []
    for i in range(4):
        rows.append(Row(id=f"x{i}", text="t", label="POS", group="u1"))
    for i in range(4):
        rows.append(Row(id=f"y{i}", text="t", label="NEG", group="u2"))

    tr, va, te = group_aware_split_rows(rows, 0.5, 0.25, 0.25, seed=123)
    tr_groups = set(r.group for r in tr)
    va_groups = set(r.group for r in va)
    te_groups = set(r.group for r in te)
    assert tr_groups.isdisjoint(va_groups)
    assert tr_groups.isdisjoint(te_groups)
    assert va_groups.isdisjoint(te_groups)
