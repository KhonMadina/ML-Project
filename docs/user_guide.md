# User Guide: Khmer Sentiment Analysis

This guide explains how to install dependencies, train baseline and transformer models, use Khmer-specific normalization, run experiment grids, and perform inference and analysis.

For research design and thesis mapping, see `docs/research_guide.md`.

---

## 1. Installation

From the project root:

```bash
pip install -r requirements.txt

# CPU-only torch (Windows example)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or follow https://pytorch.org/get-started/locally/ for CUDA
```

Optional tracking tools:
- **MLflow UI:** `mlflow ui --backend-store-uri ./mlruns`
- **Weights & Biases:**
  - `pip install wandb`
  - `wandb login`

---

## 2. Reproducibility and Tracking

Most training scripts share common flags:

- `--tracking {none,mlflow,wandb}` (default: `mlflow`)
- `--experiment_name` (run name)
- `--mlflow_tracking_uri`, `--mlflow_experiment`
- `--wandb_project`, `--wandb_entity`, `--wandb_mode`
- `--config` (optional YAML with experiment parameters)
- `--seed` (default: 42 or 123 depending on script)

Artifacts (metrics, models, vectorizers/tokenizers, confusion matrices, config snapshots) are written under `--output_dir` and logged to the selected tracker.

Example:

```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_example \
  --normalize_all \
  --experiment_name baseline_example \
  --tracking mlflow \
  --mlflow_experiment khmer_examples \
  --seed 123
```

---

## 3. Baseline Model

Train a character n-gram baseline with sensible defaults:

```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_chargram \
  --normalize_all \
  --experiment_name baseline_chargram \
  --tracking mlflow \
  --seed 123
```

Common options:
- `--use_splits` to use `final_train.csv`, `final_val.csv`, `final_test.csv` located next to `--input`.
- Group-aware splitting: `--group_column` and `--group_stratified`.
- Calibration: `--calibrate {none,platt,isotonic,temperature}` and `--calibrate_cv_folds`.
- Resampling: `--resample {none,undersample,oversample}` and `--resample_ratio`.

Outputs:
- `metrics.json` (accuracy, macro-F1, calibration, etc.).
- `confusion_matrix.csv`.
- Serialized vectorizer and model files.
- `experiment_config.json` (config snapshot).

---

## 4. Transformer Model

Train a multilingual transformer (e.g., XLM-RoBERTa):

```bash
python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_base \
  --model_name xlm-roberta-base \
  --normalize_all \
  --experiment_name transformer_xlmr \
  --tracking mlflow \
  --mlflow_experiment khmer_transformers \
  --seed 123
```

Key hyperparameters:
- `--epochs`
- `--batch_size`
- `--lr`
- `--weight_decay`
- `--warmup_ratio`
- `--max_length`
- `--grad_accum`
- `--fp16`

Metrics are saved in `metrics.json` and logged via Hugging Face `Trainer`.

---

## 5. Khmer Text Normalization

Normalization is implemented in `modeling/text_normalization.py` and exposed through CLI flags.

### 5.1 Quick Start

Use `--normalize_all` for a good Khmer preset:

- NFC + diacritics reordering
- Zero-width removal
- Khmer digits mapping
- Khmer punctuation normalization
- General punctuation normalization
- Elongation reduction
- Emoji mapping
- Latin tagging
- Whitespace normalization

Example (baseline):

```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_norm \
  --normalize_all \
  --experiment_name baseline_norm \
  --tracking mlflow
```

### 5.2 Advanced Normalization Flags

These are supported by the normalization module and can be wired into CLIs:

- `--norm_zero_width`
- `--norm_khmer_digits {keep,map}`
- `--norm_khmer_punct`
- `--norm_diacritics_reorder`
- `--norm_emoji {keep,remove,map}`
- `--norm_latin_action {none,tag,strip}`
- `--norm_latin_threshold <float>`

For research details and recommended ablations, see the normalization sections in `research_guide.md`.

---

## 6. Experiment Grids

Many experiments are encoded as YAML grids and executed with a single command using `tools/run_experiment_grid.py`.

### 6.1 Normalization Ablation (Baseline)

```bash
python tools/run_experiment_grid.py \
  --config experiments/normalization_ablation_baseline.yml
```

This sweeps normalization variants (e.g., Khmer digits, punctuation, emoji handling) for the baseline model.

### 6.2 Calibration Grid (Baseline)

```bash
python tools/run_experiment_grid.py \
  --config experiments/calibration_grid_baseline.yml
```

This compares different calibration methods and cross-validation folds.

### 6.3 Transformer and Tokenizer Grid

```bash
python tools/run_experiment_grid.py \
  --config experiments/tokenizer_transformer_grid.yml
```

This compares:
- Different transformer backbones (e.g., `xlm-roberta-base`, `bert-base-multilingual-cased`).
- Different handling of Latin spans (`norm_latin_action`).
- Optionally, different tokenizer paths (when configured).

---

## 7. Custom Tokenizers

Train custom Unigram/BPE tokenizers for Khmer text using `tools/train_tokenizer.py`.

### 7.1 Training Examples

```bash
# Unigram tokenizer from CSV
python tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram \
  --vocab_size 16000 \
  --normalize_all

# BPE tokenizer from text files
python tools/train_tokenizer.py \
  --input_txt data/corpus1.txt data/corpus2.txt \
  --output_dir tokenizers/bpe_kh_32k \
  --type bpe \
  --vocab_size 32000 \
  --norm_khmer_punct \
  --norm_khmer_digits map
```

Outputs:
- `tokenizer.json`: trained tokenizer.
- `tokenizer_report.json`: coverage and length summary.
- (Recommended) `normalization.json`: snapshot of normalization flags.

### 7.2 Using Custom Tokenizers with Transformers

Once `modeling/train_transformer.py` accepts `--tokenizer_path`, you can load a custom tokenizer:

```python
from transformers import AutoTokenizer

custom_tok = AutoTokenizer.from_pretrained("tokenizers/unigram_kh_16k", use_fast=True)
```

In CLI form (once supported):

```bash
python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_unigram_16k \
  --model_name xlm-roberta-base \
  --tokenizer_path tokenizers/unigram_kh_16k \
  --normalize_all \
  --experiment_name xlmr_unigram_16k
```

---

## 8. Inference, Stress Tests, and Error Analysis

### 8.1 Baseline Inference

Single example or batch predictions:

```bash
# Single text
python modeling/predict.py \
  --model_dir runs/baseline_chargram \
  --text "មតិយោបល់របស់អ្នកនៅទីនេះ"

# CSV file
python modeling/predict.py \
  --model_dir runs/baseline_chargram \
  --input_csv data/new_comments.csv \
  --output_csv data/new_comments_scored.csv
```

### 8.2 Stress Evaluation

Evaluate robustness on stress-test sets:

```bash
python modeling/stress_eval.py \
  --model_dir runs/baseline_chargram \
  --input_csv data/stress_tests.csv \
  --output_dir runs/baseline_chargram_stress \
  --save_misclassified
```

### 8.3 Error Analysis

Use `modeling/error_analysis.py` to inspect frequent error types and group-based performance (e.g., code-switched vs. non-code-switched). Typical usage patterns:

- Compare predictions vs. gold labels on the validation/test set.
- Slice performance by groups defined in `annotation/sample_data/groups.csv`.

(Exact CLI arguments may evolve; check the script’s `--help`.)

---

## 9. Data Validation and Annotation Pipeline

Before training on a new dataset:

1. **Validate the dataset**:

   ```bash
   python tools/validate_dataset.py \
     --input your_dataset.csv
   ```

2. **Follow the annotation workflow** (for new projects):
   - See `annotation/README.md` and `annotation/guidelines.md`.
   - Use `annotation/adjudicate.py` and `annotation/adjudication_log.md` for disagreement resolution.
   - Finalize the dataset with `annotation/finalize_dataset.py` and/or `tools/ingest_dataset.py`.

---

## 10. Tips for Reliable Results

- Always set `--seed` and log all runs with MLflow or W&B.
- Use `--normalize_all` as a safe default for Khmer social text.
- Keep dataset splits fixed (`--use_splits`) for fair comparisons.
- Save and track normalization configs (`normalization.json`) and tokenizer configs.
- When comparing models, keep everything else (data, normalization, seed) constant and vary **one factor at a time**.
- For publication/thesis work, link each table/figure back to script + config + MLflow run ID.
