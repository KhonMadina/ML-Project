#!/usr/bin/env python3
"""
Train a Khmer-aware subword tokenizer (Unigram/BPE) using HuggingFace tokenizers.

Upgrades for Step 5:
- Grid training mode across types and vocab sizes with deterministic seeding
- Rich tokenizer report: coverage proxy, unknown-token rate, sequence-length stats (mean/median/p95), char-to-token ratios, per-language breakdown (if available)
- Saves normalization.json alongside tokenizer.json for reproducibility

Examples:
  # Single Unigram tokenizer from final_dataset.csv text column
  python tools/train_tokenizer.py \
    --input_csv annotation/sample_data/final_dataset.csv \
    --text_column text \
    --output_dir tokenizers/unigram_kh_16k \
    --type unigram --vocab_size 16000 --normalize_all --seed 123

  # Grid: types × vocab sizes
  python tools/train_tokenizer.py \
    --input_csv annotation/sample_data/final_dataset.csv \
    --text_column text \
    --output_root tokenizers/kh \
    --grid --types unigram bpe --vocab_sizes 8000 16000 32000 \
    --normalize_all --seed 123

Dependencies:
  pip install tokenizers pandas
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import json
import math
import random

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, Unigram
    from tokenizers.trainers import BpeTrainer, UnigramTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
except Exception as e:
    raise SystemExit(
        "Missing dependency. Install with: pip install tokenizers pandas\n"
        f"Underlying import error: {e}"
    )

# Reuse normalization from modeling
from modeling.text_normalization import build_norm_config_from_args, normalize_text, save_norm_config


def read_texts_from_csv(path: Path, text_column: str) -> Tuple[List[str], Optional[List[str]]]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if text_column not in df.columns:
            raise SystemExit(f"Column '{text_column}' not found in {path} (available: {list(df.columns)})")
        texts = [str(t) for t in df[text_column].astype(str).tolist()]
        langs = None
        if "lang" in df.columns:
            langs = [str(x).lower() if pd.notna(x) else "" for x in df["lang"]]
        return texts, langs
    # Fallback
    rows: List[str] = []
    langs: Optional[List[str]] = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if text_column not in (reader.fieldnames or []):
            raise SystemExit(f"Column '{text_column}' not found in {path} (available: {reader.fieldnames})")
        has_lang = "lang" in (reader.fieldnames or [])
        if has_lang:
            langs = []
        for r in reader:
            rows.append(str(r.get(text_column, "")))
            if has_lang and langs is not None:
                langs.append(str(r.get("lang", "")).lower())
    return rows, langs


def read_texts_from_txt(paths: List[Path]) -> List[str]:
    texts: List[str] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            texts.extend([line.rstrip("\n") for line in f])
    return texts


def write_report(out_dir: Path, report: Dict) -> None:
    (out_dir / "tokenizer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_everything(seed: int) -> None:
    try:
        import numpy as _np  # type: ignore
        _np.random.seed(seed)
    except Exception:
        pass
    random.seed(seed)


def _compute_stats(tok: Tokenizer, texts: List[str], langs: Optional[List[str]] = None) -> Dict[str, object]:
    # Character coverage proxy
    vocab = set([t for t, _ in tok.get_vocab().items()])
    charset = set("".join(texts))
    char_covered = sum(1 for c in charset if any(c in v for v in vocab))
    coverage = char_covered / max(1, len(charset))
    # Sequence stats
    import statistics as _st
    lengths = [len(tok.encode(t).tokens) for t in texts]
    if not lengths:
        lengths = [0]
    mean_len = float(_st.mean(lengths))
    median_len = float(_st.median(lengths))
    p95_len = float(_st.quantiles(lengths, n=100)[94]) if len(lengths) >= 100 else float(max(lengths))
    # Char-to-token ratios
    char_to_tok = [ (len(t) / max(1, l)) for t, l in zip(texts, lengths) ]
    mean_c2t = float(_st.mean(char_to_tok))
    # Unknown token rate (sample on 10% tail)
    n = len(texts)
    n_hold = max(1, int(0.1 * n))
    held = texts[-n_hold:]
    unk_id = tok.token_to_id("<unk>")
    unk_rate = 0.0
    if unk_id is not None:
        total_toks = 0
        total_unk = 0
        for t in held:
            enc = tok.encode(t)
            total_toks += len(enc.ids)
            total_unk += sum(1 for i in enc.ids if i == unk_id)
        if total_toks > 0:
            unk_rate = float(total_unk) / float(total_toks)
    stats: Dict[str, object] = {
        "coverage_proxy": float(coverage),
        "seq_length": {"mean": mean_len, "median": median_len, "p95": p95_len},
        "char_to_token_ratio_mean": mean_c2t,
        "unk_token_rate_heldout": float(unk_rate),
    }
    # Per-language breakdown
    if langs is not None:
        from collections import defaultdict as _dd
        by_lang: Dict[str, List[str]] = {}
        for t, l in zip(texts, langs):
            l2 = l if isinstance(l, str) and l else "unknown"
            by_lang.setdefault(l2, []).append(t)
        per_lang: Dict[str, object] = {}
        for l, ts in by_lang.items():
            per_lang[l] = _compute_stats(tok, ts, None)  # type: ignore
        stats["per_language"] = per_lang
    return stats


def _train_one(texts: List[str], out_dir: Path, ttype: str, vocab_size: int, min_frequency: int, special_tokens: List[str]) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if ttype == "unigram":
        model = Unigram()
        trainer = UnigramTrainer(vocab_size=int(vocab_size), special_tokens=special_tokens)
    else:
        model = BPE(unk_token="<unk>")
        trainer = BpeTrainer(vocab_size=int(vocab_size), min_frequency=int(min_frequency), special_tokens=special_tokens)

    tok = Tokenizer(model)
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 2), ("</s>", 3)],
    )

    # Train from temporary corpus file
    temp_corpus = out_dir / "_tmp_corpus.txt"
    with temp_corpus.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    tok.train(files=[str(temp_corpus)], trainer=trainer)
    temp_corpus.unlink(missing_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    return {
        "tokenizer": tok,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Khmer-aware subword tokenizer (Unigram/BPE)")
    gsrc = ap.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--input_csv", help="CSV file to read text from")
    gsrc.add_argument("--input_txt", nargs="*", help="Plain text files to read from")
    ap.add_argument("--text_column", default="text", help="Column in CSV containing text")

    # Single training mode
    ap.add_argument("--output_dir", help="Directory to save tokenizer artifacts (single mode)")
    ap.add_argument("--type", choices=["unigram", "bpe"], default="unigram")
    ap.add_argument("--vocab_size", type=int, default=16000)

    # Grid mode
    ap.add_argument("--grid", action="store_true", help="Enable grid mode (train multiple tokenizers)")
    ap.add_argument("--types", nargs="*", default=None, help="Tokenizer types for grid (e.g., unigram bpe)")
    ap.add_argument("--vocab_sizes", nargs="*", type=int, default=None, help="Vocab sizes for grid (e.g., 8000 16000 32000)")
    ap.add_argument("--output_root", help="Root directory for grid outputs (subdirs will be created)")

    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--special_tokens", nargs="*", default=["<pad>", "<unk>", "<s>", "</s>"])
    ap.add_argument("--seed", type=int, default=123)

    # Normalization args (subset from modeling)
    ap.add_argument("--normalize_all", action="store_true")
    ap.add_argument("--norm_nfc", action="store_true")
    ap.add_argument("--norm_whitespace", action="store_true")
    ap.add_argument("--norm_punct", action="store_true")
    ap.add_argument("--norm_elongation", action="store_true")
    ap.add_argument("--norm_emoji", choices=["keep", "remove", "map"], default=None)
    ap.add_argument("--norm_zero_width", action="store_true")
    ap.add_argument("--norm_khmer_digits", choices=["keep", "map"], default=None)
    ap.add_argument("--norm_khmer_punct", action="store_true")
    ap.add_argument("--norm_diacritics_reorder", action="store_true")
    ap.add_argument("--norm_latin_action", choices=["none", "tag", "strip"], default=None)
    ap.add_argument("--norm_latin_threshold", type=float, default=None)

    args = ap.parse_args()

    _seed_everything(int(args.seed))

    # Collect texts (+ optional language labels)
    if args.input_csv:
        texts_raw, langs = read_texts_from_csv(Path(args.input_csv), args.text_column)
    else:
        paths = [Path(p) for p in (args.input_txt or [])]
        texts_raw = read_texts_from_txt(paths)
        langs = None

    # Normalize texts deterministically
    norm_cfg = build_norm_config_from_args(args)
    texts = [normalize_text(t, norm_cfg) for t in texts_raw]

    # Helper to finish a trained tokenizer directory
    def finalize_dir(dir_path: Path, tok: Tokenizer, meta: Dict[str, object]) -> None:
        # Save normalization config snapshot
        try:
            save_norm_config(dir_path, norm_cfg)
        except Exception:
            pass
        # Build and save report
        stats = _compute_stats(tok, texts, langs)
        report = {
            "normalization": norm_cfg,
            "seed": int(args.seed),
            **meta,
            **stats,
        }
        write_report(dir_path, report)

    # Single mode
    if not args.grid:
        if not args.output_dir:
            raise SystemExit("--output_dir is required in single mode")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = _train_one(texts, out_dir, args.type, int(args.vocab_size), int(args.min_frequency), list(args.special_tokens))
        finalize_dir(out_dir, res["tokenizer"], {"type": args.type, "vocab_size": int(args.vocab_size), "min_frequency": int(args.min_frequency), "special_tokens": list(args.special_tokens)})
        print(f"Saved tokenizer to {out_dir}/tokenizer.json")
        print((out_dir / "tokenizer_report.json").read_text(encoding="utf-8"))
        return

    # Grid mode
    if not args.output_root:
        raise SystemExit("--output_root is required when --grid is set")
    if not args.types or not args.vocab_sizes:
        raise SystemExit("--types and --vocab_sizes are required when --grid is set")

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    for ttype in args.types:
        if ttype not in ("unigram", "bpe"):
            print(f"Warning: skipping unknown type '{ttype}'")
            continue
        for vs in args.vocab_sizes:
            sub = root / f"{ttype}_{int(vs)}"
            sub.mkdir(parents=True, exist_ok=True)
            print(f"[train_tokenizer] Training {ttype} vocab={vs} -> {sub}")
            res = _train_one(texts, sub, ttype, int(vs), int(args.min_frequency), list(args.special_tokens))
            finalize_dir(sub, res["tokenizer"], {"type": ttype, "vocab_size": int(vs), "min_frequency": int(args.min_frequency), "special_tokens": list(args.special_tokens)})
    print(f"[train_tokenizer] Grid complete under {root}")


if __name__ == "__main__":
    main()
