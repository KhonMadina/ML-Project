#!/usr/bin/env python3
"""
Train a Khmer-aware subword tokenizer (Unigram/BPE) using HuggingFace tokenizers.

- Reads one or more text sources (CSV column or plain text files)
- Applies the same normalization pipeline as modeling/text_normalization.py
- Trains Unigram or BPE with configurable vocab size and special tokens
- Saves tokenizer.json and vocab files under output_dir
- Writes a training report (coverage, OOV proxy via held-out split)

Examples:
  # Train Unigram tokenizer from final_dataset.csv text column
  python tools/train_tokenizer.py \
    --input_csv annotation/sample_data/final_dataset.csv \
    --text_column text \
    --output_dir tokenizers/unigram_kh_16k \
    --type unigram --vocab_size 16000 --normalize_all

  # Train BPE tokenizer from plain text files
  python tools/train_tokenizer.py \
    --input_txt data/corpus1.txt data/corpus2.txt \
    --output_dir tokenizers/bpe_kh_32k \
    --type bpe --vocab_size 32000 --norm_khmer_punct --norm_khmer_digits map

Dependencies:
  pip install tokenizers pandas
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional

import json
import math

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
from modeling.text_normalization import build_norm_config_from_args, normalize_text


def read_texts_from_csv(path: Path, text_column: str) -> List[str]:
    if pd is not None:
        df = pd.read_csv(path, encoding="utf-8")
        if text_column not in df.columns:
            raise SystemExit(f"Column '{text_column}' not found in {path} (available: {list(df.columns)})")
        return [str(t) for t in df[text_column].astype(str).tolist()]
    # Fallback
    rows: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if text_column not in (reader.fieldnames or []):
            raise SystemExit(f"Column '{text_column}' not found in {path} (available: {reader.fieldnames})")
        for r in reader:
            rows.append(str(r.get(text_column, "")))
    return rows


def read_texts_from_txt(paths: List[Path]) -> List[str]:
    texts: List[str] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            texts.extend([line.rstrip("\n") for line in f])
    return texts


def write_report(out_dir: Path, report: Dict) -> None:
    (out_dir / "tokenizer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Khmer-aware subword tokenizer (Unigram/BPE)")
    gsrc = ap.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--input_csv", help="CSV file to read text from")
    gsrc.add_argument("--input_txt", nargs="*", help="Plain text files to read from")
    ap.add_argument("--text_column", default="text", help="Column in CSV containing text")
    ap.add_argument("--output_dir", required=True, help="Directory to save tokenizer artifacts")
    ap.add_argument("--type", choices=["unigram", "bpe"], default="unigram")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--special_tokens", nargs="*", default=["<pad>", "<unk>", "<s>", "</s>"])

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect texts
    if args.input_csv:
        texts = read_texts_from_csv(Path(args.input_csv), args.text_column)
    else:
        paths = [Path(p) for p in (args.input_txt or [])]
        texts = read_texts_from_txt(paths)

    # Normalize
    norm_cfg = build_norm_config_from_args(args)
    texts = [normalize_text(t, norm_cfg) for t in texts]

    # Setup tokenizer model/trainer
    if args.type == "unigram":
        model = Unigram()
        trainer = UnigramTrainer(vocab_size=args.vocab_size, special_tokens=args.special_tokens)
    else:
        model = BPE(unk_token="<unk>")
        trainer = BpeTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=args.special_tokens)

    tok = Tokenizer(model)
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 2), ("</s>", 3)],
    )

    # Train
    # tokenizers lib expects iterator of texts through files; we can train from memory via temporary file
    temp_corpus = out_dir / "_tmp_corpus.txt"
    with temp_corpus.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")
    tok.train(files=[str(temp_corpus)], trainer=trainer)
    temp_corpus.unlink(missing_ok=True)

    # Save
    tok.save(str(out_dir / "tokenizer.json"))

    # Build simple coverage metrics: % of chars present in vocab tokens (proxy), average token length
    # Using naive proxy: map all chars to whether they appear in any token (for character coverage it’s loose)
    vocab = set([t.split("@@")[0] for t, _ in tok.get_vocab().items()])
    charset = set("".join(texts))
    char_covered = sum(1 for c in charset if any(c in v for v in vocab))
    coverage = char_covered / max(1, len(charset))

    # OOV proxy: tokenize held-out 5% and compute average token length vs. training
    n = len(texts)
    n_hold = max(1, int(0.05 * n))
    held = texts[-n_hold:]
    tok_lens = [len(tok.encode(t).tokens) for t in held]
    avg_len = sum(tok_lens) / len(tok_lens)

    report = {
        "type": args.type,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": args.special_tokens,
        "coverage_proxy": coverage,
        "avg_tokens_on_heldout": avg_len,
        "normalization": norm_cfg,
    }
    write_report(out_dir, report)

    print(f"Saved tokenizer to {out_dir}/tokenizer.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
