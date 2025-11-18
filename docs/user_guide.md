# User Guide: Khmer Sentiment Analysis

This guide explains how to install dependencies, train baseline and transformer models, apply Khmer-specific normalization, and train custom tokenizers.

## 1. Installation

```bash
pip install -r requirements.txt
# CPU-only torch (Windows example)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Or follow https://pytorch.org/get-started/locally/ for CUDA
```

Optional tracking:
- MLflow UI: `mlflow ui --backend-store-uri ./mlruns`
- W&B: `pip install wandb` and `wandb login`

## 2. Reproducibility and Tracking

Both baseline and transformer training scripts accept tracking and experiment flags:
- `--tracking {none,mlflow,wandb}` (default: mlflow)
- `--experiment_name` (run name)
- `--mlflow_tracking_uri`, `--mlflow_experiment`
- `--wandb_project`, `--wandb_entity`, `--wandb_mode`
- `--config` (optional YAML with experiment parameters)
- `--seed` (default: 42)

Artifacts (metrics, models, vectorizers/tokenizers, confusion matrices) are saved to `--output_dir` and logged to the selected tracker.

## 3. Training the Baseline

```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --output_dir runs/baseline_chargram \
  --normalize_all \
  --experiment_name baseline_chargram --tracking mlflow --seed 123
```

Options:
- `--use_splits` to use `final_train.csv`, `final_val.csv`, `final_test.csv` next to `--input`.
- Group-aware splitting: `--group_column`, `--group_stratified`.
- Calibration: `--calibrate {none,platt,isotonic,temperature}`.
- Resampling: `--resample {none,undersample,oversample}` and `--resample_ratio`.

## 4. Training a Transformer

```bash
python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_base \
  --model_name xlm-roberta-base \
  --normalize_all \
  --experiment_name transformer_xlmr --tracking mlflow --seed 123
```

Hyperparameters: `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--warmup_ratio`, `--max_length`, `--grad_accum`, `--fp16`.

## 5. Khmer Text Normalization

Use `--normalize_all` for a balanced Khmer preset:
- NFC + diacritics reordering
- Zero-width removal
- Khmer digits mapping
- Khmer punctuation normalization
- General punctuation normalization
- Elongation reduction
- Emoji mapping
- Latin tagging
- Whitespace normalization

Advanced flags (can be added to the training CLI; normalization module already supports them):
- `--norm_zero_width`
- `--norm_khmer_digits {keep,map}`
- `--norm_khmer_punct`
- `--norm_diacritics_reorder`
- `--norm_latin_action {none,tag,strip}`
- `--norm_latin_threshold <float>`

## 6. Train a Custom Tokenizer

```bash
# Unigram tokenizer from CSV
python tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram --vocab_size 16000 --normalize_all

# BPE tokenizer from text files
python tools/train_tokenizer.py \
  --input_txt data/corpus1.txt data/corpus2.txt \
  --output_dir tokenizers/bpe_kh_32k \
  --type bpe --vocab_size 32000 --norm_khmer_punct --norm_khmer_digits map
```

Outputs:
- `tokenizer.json`: the trained tokenizer.
- `tokenizer_report.json`: summary including coverage proxy and held-out tokenization length.

To use with transformers:
```python
from transformers import AutoTokenizer
custom_tok = AutoTokenizer.from_pretrained("tokenizers/unigram_kh_16k", use_fast=True)
```

## 7. Stress Evaluation and Inference

- Baseline inference and batch evaluation: `python modeling/predict.py --model_dir <dir> --text "..."` or `--input_csv <csv> --output_csv <out>`.
- Stress evaluation by category: `python modeling/stress_eval.py --model_dir <dir> --input_csv <csv> --output_dir <out> [--save_misclassified]`.

## 8. Tips for Reliable Results

- Pin seeds and track every run with MLflow/W&B.
- Save normalization configs (generated automatically as `normalization.json`).
- Validate datasets with `tools/validate_dataset.py` before training.
- For transformer experiments with custom tokenizers: keep the same base model and swap tokenizers to isolate the effect of tokenization.
