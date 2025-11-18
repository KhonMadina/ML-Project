# Research Guide: Khmer Sentiment Analysis

This guide summarizes recommended research protocols and how to reproduce experiments in this repository. It reflects Step 1 (Reproducibility & Tracking) and Step 2 (Khmer normalization & tokenizer) upgrades.

## 1. Reproducibility and Tracking

### 1.1 Deterministic Setup
- All training scripts accept `--seed` (default 42). The experiment utilities set global seeds for Python, NumPy, and PyTorch (if available) and toggle deterministic cuDNN flags.
- A snapshot of each run’s configuration and environment is written to `<output_dir>/experiment_config.json`.

### 1.2 Experiment Tracking
- MLflow is supported by default, and Weights & Biases (W&B) optionally.
- New flags on training scripts (baseline and transformer):
  - `--tracking {none,mlflow,wandb}`: default `mlflow`.
  - `--experiment_name`: user-defined run name.
  - `--mlflow_tracking_uri`: optional; if not set, MLflow defaults to local `./mlruns`.
  - `--mlflow_experiment`: optional experiment grouping in MLflow.
  - `--wandb_project`, `--wandb_entity`, `--wandb_mode` for W&B.
  - `--config`: optional YAML config with experiment parameters (merged with CLI overrides).

Artifacts automatically logged include `metrics.json`, confusion matrices (baseline), and model artifacts (vectorizers/models/tokenizers).

### 1.3 Example: Baseline with MLflow
```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --output_dir runs/baseline_chargram \
  --experiment_name baseline_chargram \
  --tracking mlflow \
  --seed 123 \
  --normalize_all

# then
mlflow ui --backend-store-uri ./mlruns
```

### 1.4 Example: Transformer with MLflow
```bash
python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_base \
  --model_name xlm-roberta-base \
  --experiment_name transformer_xlmr \
  --tracking mlflow \
  --seed 123 \
  --normalize_all
```

## 2. Khmer Text Normalization

The normalization module now includes Khmer-specific rules and optional configurations. Use `--normalize_all` for a sensible preset suitable for Khmer social text.

### 2.1 Supported Operations
- Zero-width removal (ZWSP, ZWNJ, ZWJ, BOM)
- Unicode NFC and diacritics reordering
- Khmer digits mapping (U+17E0–U+17E9 → ASCII 0–9)
- Khmer punctuation normalization (។, ៕, ៖)
- General punctuation normalization (curly quotes, dashes, ellipsis, repeated punctuation)
- Emoji handling: keep/remove/map
- Elongation reduction: compress >2 repeats to 2
- Whitespace normalization
- Latin code-switch handling: none/tag/strip with threshold

### 2.2 CLI Flags (available to be added/used)
- `--normalize_all`
- `--norm_nfc`, `--norm_whitespace`, `--norm_punct`, `--norm_elongation`
- `--norm_emoji {keep,remove,map}`
- `--norm_zero_width`
- `--norm_khmer_digits {keep,map}`
- `--norm_khmer_punct`
- `--norm_diacritics_reorder`
- `--norm_latin_action {none,tag,strip}`
- `--norm_latin_threshold <float>`

Note: Baseline and transformer scripts already support the first set and will honor additional flags once added to their argparse definitions. The normalization module fully supports all flags today.

### 2.3 Recommended Protocol
1. Start with `--normalize_all` to get consistent gains.
2. Ablate individual operations (e.g., disable Khmer digits mapping) and report accuracy/F1 changes on val/test.
3. Tie error categories (from error/stress evaluation) to normalization operations for qualitative analysis.

## 3. Tokenizer Experiments for Khmer

A new utility trains Unigram/BPE tokenizers on Khmer corpora.

### 3.1 Training a Tokenizer
```bash
# Unigram from CSV
authors/tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram --vocab_size 16000 --normalize_all

# BPE from text files
python tools/train_tokenizer.py \
  --input_txt data/corpus1.txt data/corpus2.txt \
  --output_dir tokenizers/bpe_kh_32k \
  --type bpe --vocab_size 32000 --norm_khmer_punct --norm_khmer_digits map
```
Artifacts:
- `tokenizer.json`
- `tokenizer_report.json` (coverage proxy, average tokens on held-out, normalization snapshot)

### 3.2 Integrating Custom Tokenizers
- For transformers: replace the tokenizer by pointing to the directory containing `tokenizer.json`.
  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("tokenizers/unigram_kh_16k", use_fast=True)
  ```
- Evaluate downstream impact by fine-tuning with the custom tokenizer while keeping the same model (e.g., XLM-R). Consider models that support external tokenizers.

### 3.3 Reporting
- Compare Unigram vs. BPE, multiple vocab sizes (e.g., 8k/16k/32k).
- Report downstream accuracy/F1 and calibration metrics.
- Include qualitative tokenization analysis (wordpiece boundaries around sentiment-bearing morphemes, Khmer digits, punctuation).

## 4. Statistical Rigor
- Report confidence intervals (bootstrap) for key metrics.
- Use significance testing (e.g., paired bootstrap or approx. randomization) for model comparisons.
- Keep experiment configs and MLflow/W&B runs linked in the appendix for reproducibility.

## 5. Checklists
- [ ] Dataset validated with `tools/validate_dataset.py`
- [ ] Normalization config saved (`normalization.json`) next to model/tokenizer
- [ ] Experiment tracking on (MLflow or W&B) with `experiment_config.json` snapshot
- [ ] Tokenizer artifacts versioned by vocab size and normalization setup
- [ ] Report includes ablations and error/stress analysis tied to normalization/tokenizer choices
