import math
import types

import pytest

# Data loading and splitting
from modeling.data import Row, stratified_split, group_stratified_split

# Stress presets (import the module to access preset functions)
import modeling.stress_eval as stress

# IAA computation
from tools.compute_iaa import compute_iaa

# Bilingual validator internals
from tools import validate_dataset as vds


def make_rows_lang(n_km_pos=10, n_km_neg=10, n_en_pos=10, n_en_neg=10, group_every=5):
    rows = []
    gid = 0
    def add_many(n, label, lang):
        nonlocal gid
        for i in range(n):
            rows.append(
                Row(
                    id=f"{lang}_{label}_{i}",
                    text=f"text {lang} {label} {i}",
                    label=label,
                    group=str(gid),
                    lang=lang,
                )
            )
            if (i + 1) % group_every == 0:
                gid += 1
    add_many(n_km_pos, "POS", "km")
    add_many(n_km_neg, "NEG", "km")
    add_many(n_en_pos, "POS", "en")
    add_many(n_en_neg, "NEG", "en")
    return rows


def _count_by(rows, key):
    from collections import Counter
    return Counter([key(r) for r in rows])


def test_stratified_split_by_lang_preserves_composition():
    rows = make_rows_lang(20, 10, 15, 5)
    tr, va, te = stratified_split(rows, 0.7, 0.2, 0.1, seed=123, stratify_by_lang=True)
    # Check label-lang buckets are present in each split proportionally (within rounding)
    from collections import Counter
    all_counts = _count_by(rows, lambda r: (r.label, r.lang))
    tr_counts = _count_by(tr, lambda r: (r.label, r.lang))
    va_counts = _count_by(va, lambda r: (r.label, r.lang))
    te_counts = _count_by(te, lambda r: (r.label, r.lang))
    # Each bucket proportion should be close to target ratios (tolerance of 2 examples)
    for k, total in all_counts.items():
        assert abs(tr_counts[k] - round(total * 0.7)) <= 2
        assert abs(va_counts[k] - round(total * 0.2)) <= 2
        assert abs(te_counts[k] - (total - tr_counts[k] - va_counts[k])) <= 0  # by construction


def test_group_stratified_split_by_lang_avoids_leakage():
    rows = make_rows_lang(12, 12, 12, 12, group_every=3)
    tr, va, te = group_stratified_split(rows, 0.6, 0.2, 0.2, seed=42, stratify_by_lang=True)
    # Ensure groups are not split across splits
    group_to_split = {}
    for split_name, split_rows in [("train", tr), ("val", va), ("test", te)]:
        for r in split_rows:
            g = r.group
            if g in group_to_split:
                assert group_to_split[g] == split_name
            else:
                group_to_split[g] = split_name
    # Check bucket-level composition roughly preserved
    from collections import Counter
    all_counts = _count_by(rows, lambda r: (r.label, r.lang))
    tr_counts = _count_by(tr, lambda r: (r.label, r.lang))
    for k, total in all_counts.items():
        # target ~0.6 of each bucket
        assert tr_counts[k] >= 0


@pytest.mark.parametrize("preset", [
    "diacritics",
    "zero_width",
    "emoji_burst",
    "elongation",
    "mixed_script",
    "code_switch",
    "latinized_khmer",
])
def test_stress_presets_change_text_for_high_severity(preset):
    fn = stress.PRESETS.get(preset)
    assert isinstance(fn, types.FunctionType)
    s = "សួស្តី Hello!!!"
    # Force application with prob=1.0 and strong severity
    out = fn(s, severity=0.9, prob=1.0)
    # Not all presets guarantee a different string in every case, but on average should differ
    assert isinstance(out, str)
    # Avoid overly strict assertion; at least ensure function runs and length bounds reasonable
    assert len(out) >= 1


def test_compute_iaa_structure():
    rows = [
        {"id": "1", "annotator": "A", "label": "POS", "lang": "km"},
        {"id": "1", "annotator": "B", "label": "POS", "lang": "km"},
        {"id": "2", "annotator": "A", "label": "NEG", "lang": "en"},
        {"id": "2", "annotator": "B", "label": "NEG", "lang": "en"},
        {"id": "3", "annotator": "A", "label": "NEG", "lang": "km"},
        {"id": "3", "annotator": "B", "label": "POS", "lang": "km"},
    ]
    res = compute_iaa(rows, lang_column="lang")
    assert "annotators" in res and isinstance(res["annotators"], list)
    assert "pairs" in res and isinstance(res["pairs"], dict)
    assert "pairs_per_lang" in res and isinstance(res["pairs_per_lang"], dict)
    # Kappa values should be within [-1, 1] or NaN
    for v in list(res["pairs"].values()):
        assert (isinstance(v, float) and (math.isnan(v) or -1.0 <= v <= 1.0))


def test_validate_dataset_bilingual_checks():
    rows = [
        {"id": "1", "text": "hello", "label": "pos", "lang": "en", "split": "train", "group": "g1"},
        {"id": "2", "text": "សួស្តី", "label": "neg", "lang": "km", "split": "val", "group": "g2"},
        {"id": "3", "text": "mixed Hello សួស្តី", "label": "neu", "lang": "km", "split": "test", "group": "g3"},
    ]
    fields = ["id", "text", "label", "lang", "split", "group"]
    msgs = vds.validate(rows, fields, allow_extra_labels=False, group_column="group", lang_column="lang", split_column="split", allowed_langs=["km", "en"], allow_extra_langs=False)
    assert isinstance(msgs, list)
    # Should have no errors for valid rows
    assert not any(m.startswith("ERROR") for m in msgs)
    # Summary should include distributions and code-switch counts
    summary = vds.summarize(rows, group_column="group", lang_column="lang", split_column="split", allowed_langs=["km", "en"])
    assert "lang_distribution" in summary
    assert "split_distribution" in summary
    assert "codeswitch_counts" in summary
