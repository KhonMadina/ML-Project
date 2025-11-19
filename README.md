# Khmer Sentiment Analysis

End-to-end pipeline for Khmer sentiment classification, including:
- Dataset annotation and validation tools
- Baseline n-gram models
- Multilingual transformer models
- Khmer-specific text normalization
- Custom tokenizer training (Unigram/BPE)
- Error/stress evaluation and uncertainty calibration
- Reproducible experiment grids with tracking (MLflow/W&B)

Recent updates focus on:
- **Reproducibility & Tracking**
  - Global seeding and deterministic settings
  - Centralized experiment tracking (MLflow by default, W&B optional)
  - Config snapshots and artifact logging under each run directory
- **Khmer Normalization & Tokenizers**
  - Khmer-aware normalization (digits, punctuation, zero-width, diacritics, emoji, elongation, Latin code-switch)
  - Tokenizer training utility (Unigram/BPE) with normalization integration

---

## Quickstart

### 1. Installation

From the project root:

```bash
pip install -r requirements.txt

# CPU-only Torch on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA, follow: https://pytorch.org/get-started/locally/
```

Optional tracking tools:
- MLflow UI: `mlflow ui --backend-store-uri ./mlruns`
- Weights & Biases: `pip install wandb` and `wandb login`

---

### 2. Train a Baseline Model

Train a character n-gram baseline with Khmer normalization and MLflow tracking:

```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_chargram \
  --normalize_all \
  --experiment_name baseline_chargram \
  --tracking mlflow \
  --mlflow_experiment khmer_baselines \
  --seed 123
```

Key options:
- `--use_splits` to respect pre-defined train/val/test splits.
- `--calibrate {none,platt,isotonic,temperature}` for probability calibration.
- `--resample {none,undersample,oversample}` for class imbalance handling.

Outputs (under `--output_dir`):
- `metrics.json`, `confusion_matrix.csv`
- Serialized vectorizer/model
- `experiment_config.json` and `normalization.json`

---

### 3. Train a Transformer Model

Train a multilingual transformer (e.g., XLM-RoBERTa) with the same dataset and normalization:

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

Core hyperparameters (CLI flags): `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--warmup_ratio`, `--max_length`, `--grad_accum`, `--fp16`.

---

### 4. Train a Khmer Tokenizer

Train a Unigram tokenizer tailored to the Khmer sentiment dataset:

```bash
python tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram \
  --vocab_size 16000 \
  --normalize_all
```

Artifacts:
- `tokenizer.json`: trained tokenizer
- `tokenizer_report.json`: coverage and sequence-length statistics
- `normalization.json`: normalization flags used during training

Custom tokenizers can be plugged into transformer experiments via an optional `--tokenizer_path` argument (see `docs/user_guide.md`).

---

### 5. Inference & Stress Evaluation

Baseline inference:

```bash
python modeling/predict.py \
  --model_dir runs/baseline_chargram \
  --text "មតិយោបល់របស់អ្នកនៅទីនេះ"
```

Stress evaluation:

```bash
python modeling/stress_eval.py \
  --model_dir runs/baseline_chargram \
  --input_csv data/stress_tests.csv \
  --output_dir runs/baseline_chargram_stress
```

For more details (including group-based analysis and error inspection), see the User Guide.

---

## Experiments & Reproducibility

Core experiment grids are defined under `experiments/` and run with:

```bash
python tools/run_experiment_grid.py --config <config.yml>
```

Main grids:
- `experiments/normalization_ablation_baseline.yml`: Khmer normalization ablation for the baseline.
- `experiments/calibration_grid_baseline.yml`: calibration methods and cross-validation strategies.
- `experiments/tokenizer_transformer_grid.yml`: transformer backbones, Latin handling, and tokenizer variants.

Each run logs configuration, metrics, and artifacts to MLflow (by default) under `./mlruns`.

---

## Project Structure (High-Level)

- `annotation/` – Annotation guidelines, adjudication scripts, dataset curation.
- `docs/` –
  - `user_guide.md`: step-by-step usage for practitioners.
  - `research_guide.md`: research design, experiment protocol, thesis mapping.
- `experiments/` – YAML experiment grids for reproducible sweeps.
- `modeling/` – Training, prediction, normalization, calibration, error/stress modules.
- `tools/` – Utility scripts (grid runner, tokenizer training, dataset validation/ingest).
- `tests/` – Unit tests for key components.

---

## Documentation

- **User Guide:** `docs/user_guide.md`
- **Research Guide:** `docs/research_guide.md`
- **Annotation Workflow:** `annotation/README.md`, `annotation/guidelines.md`

These documents provide the recommended commands and protocols for both day-to-day usage and thesis-level research.

---

## Tests

Run unit tests from the project root:

```bash
pytest -q
```

---

## Notes

- Normalization configurations are saved as `normalization.json` in each model/tokenizer directory.
- Each run also stores `experiment_config.json` (CLI + config snapshot) to aid reproducibility.
- MLflow UI can be launched with `mlflow ui --backend-store-uri ./mlruns`.
- For W&B tracking, run `wandb login` and set `--tracking wandb`.
- For research usage, map results to scripts/configs as described in `docs/research_guide.md`.
