#!/usr/bin/env python3
"""
Text normalization utilities for Khmer sentiment pipeline.

Khmer-aware, configurable normalization with conservative defaults.

Features (configurable):
- Remove zero-width characters (ZWSP, ZWNJ, ZWJ, BOM)
- Unicode NFC normalization and diacritics reordering by combining class
- Khmer numerals mapping to ASCII digits
- Khmer punctuation normalization (e.g., ។ ៕ ៖ → . . :)
- General punctuation normalization (curly quotes, dashes, ellipsis, repeated punctuation)
- Emoji handling (keep/remove/map-to-token)
- Elongation reduction (compress >2 repeated chars to 2)
- Whitespace normalization (collapse runs, trim)
- Latin code-switch handling (none/tag/strip) with threshold

Config schema (dict):
{
  "nfc": bool,
  "whitespace": bool,
  "punct": bool,
  "elongation": bool,
  "emoji": "keep" | "remove" | "map",
  "remove_zero_width": bool,
  "khmer_digits": "keep" | "map",
  "khmer_punct": bool,
  "diacritics_reorder": bool,
  "latin_action": "none" | "tag" | "strip",
  "latin_threshold": float  # fraction of Latin letters to trigger 'tag' or 'strip'
}

Defaults: conservative (no changes besides NFC/whitespace) unless normalize_all is requested.

This module avoids heavyweight dependencies; normalization is rule-based and unit-testable.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

# ===============================
# Patterns and maps
# ===============================

# Zero-width characters (ZWSP, ZWNJ, ZWJ, BOM)
ZERO_WIDTH_RE = re.compile("[\u200B\u200C\u200D\uFEFF]")

# Basic emoji ranges (not exhaustive but covers most modern emoji)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001F6FF\U0001F900-\U0001FAFF\u2600-\u27BF]")

# Latin letters (for code-switch detection)
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# Khmer numerals mapping (U+17E0..U+17E9) → ASCII '0'..'9'
KHMER_DIGITS_MAP = {chr(0x17E0 + i): str(i) for i in range(10)}

# Khmer punctuation normalization
# - \u17D4 KHMER SIGN KHAN (។) → '.'
# - \u17D5 KHMER SIGN BARIYOOSAN (៕) → '.'
# - \u17D6 KHMER SIGN CAMNUC PII KUUH (៖) → ':'
# - \u17D7 KHMER SIGN LEK TOO (ៗ, iteration mark): keep single; compress repeats
KHMER_PUNCT_MAP = {
    "\u17D4": ".",  # ។ → .
    "\u17D5": ".",  # ៕ → .
    "\u17D6": ":",  # ៖ → :
}
KHMER_ITERATION = "\u17D7"

# General punctuation normalization to ASCII
PUNCT_MAP = {
    "\u2018": "'",   # left single quotation mark
    "\u2019": "'",   # right single quotation mark
    "\u201C": '"',   # left double quotation mark
    "\u201D": '"',   # right double quotation mark
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2026": "...",  # ellipsis
}

# Compress repeated ASCII punctuation
RE_REPEATED_PUNCT = re.compile(r"([!?.,:])\1{1,}")
# Compress char repeats >2 to 2
RE_ELONGATION = re.compile(r"(.)\1{2,}")
# Collapse whitespace
RE_WHITESPACE = re.compile(r"\s+")


# ===============================
# Config utilities
# ===============================

def default_norm_config() -> Dict[str, object]:
    return {
        "nfc": False,
        "whitespace": False,
        "punct": False,
        "elongation": False,
        "emoji": "keep",
        "remove_zero_width": False,
        "khmer_digits": "keep",
        "khmer_punct": False,
        "diacritics_reorder": False,
        "latin_action": "none",
        "latin_threshold": 0.2,
    }


def make_norm_config(
    nfc: bool,
    whitespace: bool,
    punct: bool,
    elongation: bool,
    emoji: str,
    remove_zero_width: bool = False,
    khmer_digits: str = "keep",
    khmer_punct: bool = False,
    diacritics_reorder: bool = False,
    latin_action: str = "none",
    latin_threshold: float = 0.2,
) -> Dict[str, object]:
    cfg = {
        "nfc": bool(nfc),
        "whitespace": bool(whitespace),
        "punct": bool(punct),
        "elongation": bool(elongation),
        "emoji": emoji if emoji in {"keep", "remove", "map"} else "keep",
        "remove_zero_width": bool(remove_zero_width),
        "khmer_digits": khmer_digits if khmer_digits in {"keep", "map"} else "keep",
        "khmer_punct": bool(khmer_punct),
        "diacritics_reorder": bool(diacritics_reorder),
        "latin_action": latin_action if latin_action in {"none", "tag", "strip"} else "none",
        "latin_threshold": float(latin_threshold),
    }
    return cfg


def build_norm_config_from_args(args) -> Dict[str, object]:
    """Build normalization config from argparse args.
    Supported args:
      - normalize_all (bool)
      - norm_nfc, norm_whitespace, norm_punct, norm_elongation (bool)
      - norm_emoji (str: keep/remove/map)
      - norm_zero_width (bool)
      - norm_khmer_digits (str: keep/map)
      - norm_khmer_punct (bool)
      - norm_diacritics_reorder (bool)
      - norm_latin_action (str: none/tag/strip)
      - norm_latin_threshold (float)
    """
    cfg = default_norm_config()

    # Sensible default pipeline for Khmer + social text
    if getattr(args, "normalize_all", False):
        cfg.update({
            "nfc": True,
            "whitespace": True,
            "punct": True,
            "elongation": True,
            "emoji": "map",
            "remove_zero_width": True,
            "khmer_digits": "map",
            "khmer_punct": True,
            "diacritics_reorder": True,
            "latin_action": "tag",
            "latin_threshold": 0.2,
        })

    # Individual overrides
    if getattr(args, "norm_nfc", False):
        cfg["nfc"] = True
    if getattr(args, "norm_whitespace", False):
        cfg["whitespace"] = True
    if getattr(args, "norm_punct", False):
        cfg["punct"] = True
    if getattr(args, "norm_elongation", False):
        cfg["elongation"] = True

    em = getattr(args, "norm_emoji", None)
    if isinstance(em, str) and em in {"keep", "remove", "map"}:
        cfg["emoji"] = em

    if getattr(args, "norm_zero_width", False):
        cfg["remove_zero_width"] = True

    kd = getattr(args, "norm_khmer_digits", None)
    if isinstance(kd, str) and kd in {"keep", "map"}:
        cfg["khmer_digits"] = kd

    if getattr(args, "norm_khmer_punct", False):
        cfg["khmer_punct"] = True

    if getattr(args, "norm_diacritics_reorder", False):
        cfg["diacritics_reorder"] = True

    la = getattr(args, "norm_latin_action", None)
    if isinstance(la, str) and la in {"none", "tag", "strip"}:
        cfg["latin_action"] = la

    lth = getattr(args, "norm_latin_threshold", None)
    if isinstance(lth, (int, float)):
        try:
            cfg["latin_threshold"] = float(lth)
        except Exception:
            pass

    return cfg


def save_norm_config(output_dir: Path, config: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_norm_config(model_dir: Path) -> Optional[Dict[str, object]]:
    p = model_dir / "normalization.json"
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            return None
        base = default_norm_config()
        base.update(cfg)
        return base
    except Exception:
        return None


# ===============================
# Normalization primitives
# ===============================

def _remove_zero_width(text: str) -> str:
    return ZERO_WIDTH_RE.sub("", text)


def _reorder_diacritics(text: str) -> str:
    # Attempt to reorder combining marks by canonical combining class (after NFC)
    # Conservative: reorder only immediate combining sequences
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if unicodedata.combining(ch) == 0:
            # base char: collect following combining marks
            j = i + 1
            combines = []
            while j < n and unicodedata.combining(text[j]) != 0:
                combines.append(text[j])
                j += 1
            if combines:
                combines.sort(key=lambda c: unicodedata.combining(c))
                out.append(ch + "".join(combines))
            else:
                out.append(ch)
            i = j
        else:
            # stray combining mark – keep as-is
            out.append(ch)
            i += 1
    return "".join(out)


def _map_khmer_digits(text: str) -> str:
    return "".join(KHMER_DIGITS_MAP.get(c, c) for c in text)


def _normalize_khmer_punct(text: str) -> str:
    # Map Khmer punct to ASCII
    for src, tgt in KHMER_PUNCT_MAP.items():
        text = text.replace(src, tgt)
    # Compress iteration marks ៗ
    if KHMER_ITERATION in text:
        text = re.sub(f"{KHMER_ITERATION}+", KHMER_ITERATION, text)
    return text


def _normalize_punct(text: str) -> str:
    for src, tgt in PUNCT_MAP.items():
        text = text.replace(src, tgt)
    text = RE_REPEATED_PUNCT.sub(r"\1", text)
    return text


def _handle_emoji(text: str, mode: str) -> str:
    if mode == "keep":
        return text
    if mode == "remove":
        return EMOJI_RE.sub("", text)
    if mode == "map":
        return EMOJI_RE.sub(" <EMOJI> ", text)
    return text


def _latin_ratio(text: str) -> float:
    if not text:
        return 0.0
    lat = len(LATIN_LETTER_RE.findall(text))
    return lat / max(1, len(text))


def _apply_latin_action(text: str, action: str, threshold: float) -> str:
    if action == "none":
        return text
    ratio = _latin_ratio(text)
    if ratio < threshold:
        return text
    if action == "tag":
        return "<LATIN> " + text
    if action == "strip":
        return LATIN_LETTER_RE.sub("", text)
    return text


# ===============================
# Public API
# ===============================

def normalize_text(text: str, config: Optional[Dict[str, object]]) -> str:
    if text is None:
        return ""
    if not config:
        return text

    # Extract options
    nfc = bool(config.get("nfc", False))
    whitespace = bool(config.get("whitespace", False))
    punct = bool(config.get("punct", False))
    elongation = bool(config.get("elongation", False))
    emoji_mode = str(config.get("emoji", "keep"))
    remove_zw = bool(config.get("remove_zero_width", False))
    kh_digits = str(config.get("khmer_digits", "keep"))
    kh_punct = bool(config.get("khmer_punct", False))
    diac_reorder = bool(config.get("diacritics_reorder", False))
    latin_action = str(config.get("latin_action", "none"))
    latin_thresh = float(config.get("latin_threshold", 0.2))

    # Order: remove zero-width -> NFC -> diacritics reorder -> Khmer digits -> Khmer punct -> emoji -> elongation -> general punct -> latin -> whitespace
    if remove_zw:
        text = _remove_zero_width(text)
    if nfc:
        text = unicodedata.normalize("NFC", text)
    if diac_reorder:
        text = _reorder_diacritics(text)
    if kh_digits == "map":
        text = _map_khmer_digits(text)
    if kh_punct:
        text = _normalize_khmer_punct(text)
    if emoji_mode in {"keep", "remove", "map"}:
        text = _handle_emoji(text, emoji_mode)
    if elongation:
        text = RE_ELONGATION.sub(r"\1\1", text)
    if punct:
        text = _normalize_punct(text)
    # Latin handling before whitespace so tag is separated
    if latin_action in {"none", "tag", "strip"}:
        text = _apply_latin_action(text, latin_action, latin_thresh)
    if whitespace:
        text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def normalize_corpus(texts: List[str], config: Optional[Dict[str, object]]) -> List[str]:
    if not config:
        return texts
    return [normalize_text(t, config) for t in texts]
