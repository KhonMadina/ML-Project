# Understanding Guide: Khmer + English Sentiment Analysis (Simple, Practical Overview)

This guide explains the project in clear, simple terms. Read it top to bottom to understand what the system does, why each part exists, and how to use it for your graduate presentation.

---

## 1) What this project does

- Goal: Build a practical sentiment system for Khmer and English (bilingual). It predicts whether text is Positive (POS), Negative (NEG), or Neutral (NEU).
- Focus areas:
  - Clean and normalize noisy social text (especially Khmer-specific issues).
  - Train both a simple baseline and a powerful transformer model.
  - Support bilingual data and code-switching (Khmer + English in the same text).
  - Evaluate not only accuracy but also reliability (calibration) and robustness to noisy inputs.
  - Provide an API + UI to try models interactively and export tables/plots for your thesis.

---

## 2) Key ideas to know

- **Sentiment analysis**: Classify text into POS / NEG / NEU.
- **Normalization**: Clean text (remove zero-width characters, reorder diacritics, map Khmer digits to ASCII, normalize punctuation, handle emojis, compress elongations). This reduces noise and helps models learn better.
- **Tokenizer**: Break text into tokens for transformer models. Khmer-specific tokenization can shorten sequences and improve coverage.
- **Baseline model**: Character n-gram + Logistic Regression. It’s simple, fast, and a strong reference.
- **Transformer model**: Modern multilingual models (e.g., XLM-R) with higher accuracy and bilingual capability.
- **Calibration**: Make probability outputs honest. If the model says 80% confident, it should be correct ~80% of the time. Metrics: ECE, Brier. Visual: reliability diagrams.
- **Robustness/Stress tests**: Test behavior under text noise (e.g., diacritic changes, zero-width inserts, code-switch). Measures how much accuracy drops.
- **Bilingual support**: Treat Khmer + English together; track per-language metrics; optional dual-head classifier (separate heads for each language).

---

## 3) The big picture (pipeline)

1. **Dataset** (CSV): `id, text, label` (optional: `lang, group, split, source`).
2. **Validation**: Check columns, labels, allowed languages, split consistency, code-switch summary.
3. **Normalization**: Apply Khmer-aware and general text cleaning.
4. **Model training**:
   - Baseline (n-gram + Logistic Regression)
   - Transformer (XLM-R, etc.), with bilingual options
5. **Evaluation**:
   - Accuracy, macro-F1 (class-balanced), per-language metrics
   - Calibration (ECE, Brier) + reliability diagrams
6. **Robustness**:
   - Apply perturbation presets (diacritics, zero-width, emoji, code-switch...) and measure performance drop
7. **Presentation**:
   - API + UI for interactive demo
   - Export tables/plots for your report

---

## 4) Important files (where things live)

- `modeling/train_baseline.py`: Train the baseline model.
- `modeling/train_transformer.py`: Train transformer with bilingual options, per-language metrics, calibration outputs.
- `modeling/data.py`: Load CSV, create splits (including language-aware).
- `modeling/text_normalization.py`: Khmer-aware normalization rules.
- `modeling/stress_eval.py`: Run robustness tests with preset perturbations.
- `modeling/calibration_utils.py`: ECE/Brier, reliability bins and plots.
- `tools/validate_dataset.py`: Bilingual CSV checks and summaries.
- `tools/ingest_dataset.py`: Validate + finalize + export splits in one command.
- `tools/compute_iaa.py`: Inter-annotator agreement (Cohen’s kappa) from raw annotations.
- `tools/export_report.py`: Aggregate model results to CSV/plots; optional bootstrap CI linking.
- `api/server.py`: FastAPI endpoints (inference, tokenize, stress, validate, IAA, reports).
- `demo_ui.html`: Web UI to interact with the API (load model, predict, tokenize, transform, validate, report).
- `train_model.sh`: One-click bilingual training pipeline.
- `ui_api.sh`: Start API + open the UI.

---

## 5) Why normalization matters (especially for Khmer)

- Khmer text often contains invisible characters (zero-width), combining marks out of order, script-specific punctuation, Khmer digits, and heavy use of emojis.
- Social text includes elongations (e.g., long repeated characters) and English code-switch.
- Normalization makes inputs consistent and improves both baseline and transformer performance.

Core Khmer-focused operations:
- Unicode NFC + diacritics reorder
- Remove zero-width characters
- Map Khmer digits (០–៩) to ASCII (0–9)
- Standardize Khmer punctuation
- Emoji handling (keep/remove/map)
- Compress elongations
- Handle Latin spans (none/tag/strip)

---

## 6) Baseline vs Transformer

- **Baseline (n-gram + LR)**
  - Pros: fast, simple, requires little compute; can be well-calibrated
  - Cons: usually lower accuracy than transformers
- **Transformer (e.g., XLM-R)**
  - Pros: strong accuracy, multilingual capability, better for code-switching
  - Cons: heavier compute; needs care with sequence length, batch size

Bilingual options for transformers:
- `--lang_column lang` + per-language metrics
- `--stratify_by_lang` for fair random splits
- `--dual_head` (shared encoder, separate heads per language)
- `--class_weighting balanced`, `--focal_loss` for class imbalance

---

## 7) Calibration & Reliability

- Models produce probabilities; calibration measures whether these match reality.
- Metrics: **ECE** (lower is better), **Brier score** (lower is better).
- Reliability diagram: predicted confidence vs actual accuracy.
- The code saves `val_reliability.png` and `test_reliability.png` per transformer run.

---

## 8) Robustness (Stress Testing)

Preset perturbations (with severity/probability):
- `diacritics`: remove/duplicate marks
- `zero_width`: insert zero-width spaces/joiners
- `emoji_burst`: add emojis
- `elongation`: stretch character runs
- `mixed_script`: replace with lookalike characters (Latin/Cyrillic)
- `code_switch`: inject phrases from the other language
- `latinized_khmer`: strip marks and map digits to ASCII

Use `modeling/stress_eval.py` or the UI to measure performance drops under these conditions.

---

## 9) Quick start (hands-on)

1) Train demo models (baseline + transformer + tokenizer):
```bash
bash train_model.sh
```

2) Start API & open UI:
```bash
bash ui_api.sh
```

In the browser (UI):
- Load a model from the dropdown
- Enter text, click Predict → see label + class probabilities
- Click "Preview tokens" for transformer tokenization
- Try stress transforms (diacritics, code_switch, etc.)
- Validate your dataset and compute IAA from CSVs
- Export and browse reports (overall.csv, per_language.csv, calibration.csv, plots)

---

## 10) How to read results

- **Accuracy / Macro-F1**: Macro-F1 balances classes; higher is better.
- **Per-language metrics**: Check Khmer vs English; if one lags, consider dual-head or more data for that language.
- **Calibration**: Lower ECE/Brier is better; plots should be close to diagonal.
- **Robustness**: Compare clean vs stressed accuracy; large drops show weaknesses; adjust normalization or tokenizer if needed.
- **Statistical confidence**: Use paired bootstrap CIs to compare models and report whether differences are meaningful.

---

## 11) Suggested learning path

1. Run dataset validation and read the summary/issue list.
2. Enable `--normalize_all` and train the baseline and transformer on fixed splits.
3. Inspect each model’s `metrics.json` and reliability plots.
4. Try the UI: predict text, preview tokens, try a stress transform.
5. Export and browse reports to see aggregated results.
6. For bilingual experiments: include the `lang` column and turn on `--stratify_by_lang`; examine per-language metrics.

---

## 12) Common pitfalls and tips

- Always fix splits (`--use_splits`) and set `--seed` for fair comparisons.
- Keep training and inference normalization consistent (each run saves `normalization.json`).
- Change one variable at a time when comparing models (e.g., only normalization flag or only tokenizer).
- On limited hardware, reduce `--max_length` and `--batch_size`, or use a smaller model.

---

## 13) Mapping to thesis/report

- **Methods**: Data schema, validation, normalization, baseline vs transformer, bilingual options.
- **Experiments**: Baseline vs transformer; normalization/tokenizer ablations; bilingual dual-head vs single-head; calibration.
- **Results**: Tables/plots from `tools/export_report.py`; reliability diagrams; stress deltas.
- **Discussion**: Why normalization helps; robustness findings; calibration trade-offs.
- **Reproducibility**: Seeds, splits, MLflow tracking, config snapshots, and tests.

---

## 14) Where to go next

- Try new normalization settings and re-train.
- Train a Khmer tokenizer with different vocab sizes (8k/16k/32k) and compare.
- Run bilingual vs Khmer-only training and compare per-language results.
- Expand stress tests (e.g., OCR noise) and examine robustness.
- Add domain-adaptive pretraining or data augmentation for harder domains.

---

## 15) Short glossary

- **ECE (Expected Calibration Error)**: Measures how well probabilities match reality; lower is better.
- **Brier score**: Quadratic error on probability predictions; lower is better.
- **Dual-head**: Shared transformer encoder with separate classifier heads per language.
- **Code-switching**: Mixing languages (Khmer + English) in the same text.
- **Macro-F1**: Average F1 across classes; treats classes equally.