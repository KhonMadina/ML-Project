# Research Guide: Khmer Sentiment Analysis

This guide summarizes the research protocol for this repository and how to reproduce the experiments. It is aligned with the implemented code, YAML grids under `experiments/`, and scripts in `modeling/` and `tools/`.

The goals are to:
- Make the main research questions and hypotheses explicit.
- Tie each research axis to concrete scripts/configs.
- Provide copy-pasteable commands to rerun experiments.
- Support direct mapping between thesis sections and repository artifacts.

---

## 1. Research Questions and Hypotheses

Each research question (RQ) is framed so that it can be tested using the existing experiment grids and scripts.

### RQ1: Impact of Khmer-Specific Normalization

- **Question:** How do Khmer-specific text normalization operations affect sentiment classification performance on Khmer social text?
- **Hypotheses:**
  - **H1a:** Using the full preset (`--normalize_all`) improves macro-F1 and accuracy relative to no normalization.
  - **H1b:** Mapping Khmer digits to ASCII (`--norm_khmer_digits map`) improves performance compared to keeping digits in Khmer (`keep`).
  - **H1c:** Enabling Khmer punctuation normalization (`--norm_khmer_punct true`) reduces errors related to sentence/clause segmentation.
- **Code & Config:**
  - Implementation: `modeling/text_normalization.py`.
  - Baseline training: `modeling/train_baseline.py`.
  - Grid: `experiments/normalization_ablation_baseline.yml`.
- **Protocol:** Fix dataset, model, and splits; vary normalization flags via the YAML grid and compare metrics.

### RQ2: Emojis and Latin Code-Switching

- **Question:** What is the effect of different emoji and Latin code-switch handling strategies on Khmer sentiment classification?
- **Hypotheses:**
  - **H2a:** Mapping emojis to abstract sentiment tokens (`--norm_emoji map`) yields better macro-F1 than keeping them raw (`keep`) or removing them (`remove`).
  - **H2b:** Tagging Latin spans (`--norm_latin_action tag`) yields more robust performance on code-switched inputs than leaving Latin text unchanged (`none`).
- **Code & Config:**
  - Normalization implementation: `modeling/text_normalization.py`.
  - Transformer grid: `experiments/tokenizer_transformer_grid.yml` (via `norm_latin_action`) and extensions of `experiments/normalization_ablation_baseline.yml` (via `norm_emoji`).
  - Group definitions (e.g., code-switched vs. non-code-switched): `annotation/sample_data/groups.csv`.
- **Protocol:** Run experiments with different `norm_emoji` and `norm_latin_action` values, then analyze performance overall and by groups.

### RQ3: Baseline vs. Transformer Trade-offs

- **Question:** How do n-gram-based baselines compare to multilingual transformers in accuracy, calibration, and computational cost for Khmer sentiment analysis?
- **Hypotheses:**
  - **H3a:** Multilingual transformers (e.g., `xlm-roberta-base`) outperform the best char n-gram baseline in macro-F1 on identical splits.
  - **H3b:** Calibrated baselines can achieve comparable or better calibration error (ECE, Brier score) than transformers despite lower accuracy.
  - **H3c:** The baseline requires much lower computational resources (training/inference time, memory), making it preferable in constrained settings.
- **Code & Config:**
  - Baseline: `modeling/train_baseline.py`.
  - Transformer: `modeling/train_transformer.py`.
  - Normalization grid: `experiments/normalization_ablation_baseline.yml`.
  - Calibration grid: `experiments/calibration_grid_baseline.yml`.
  - Transformer grid: `experiments/tokenizer_transformer_grid.yml`.
- **Protocol:** Choose a strong baseline configuration and one or more transformer configurations from the grids, then compare metrics and resource usage under a shared setup.

### RQ4: Tokenizer Design for Khmer

- **Question:** How do tokenizer type (Unigram vs. BPE), vocabulary size, and normalization choices affect downstream Khmer sentiment performance?
- **Hypotheses:**
  - **H4a:** Khmer-specific tokenizers trained with appropriate normalization achieve better coverage and shorter sequences than off-the-shelf multilingual tokenizers.
  - **H4b:** For a fixed backbone, using a Khmer-specific tokenizer improves macro-F1 over the default tokenizer.
  - **H4c:** Intermediate vocabulary sizes (e.g., 16k) offer a better trade-off between performance and sequence length than very small (8k) or large (32k) vocabularies.
- **Code & Config:**
  - Tokenizer training: `tools/train_tokenizer.py`.
  - Transformer training: `modeling/train_transformer.py`.
  - Grid: `experiments/tokenizer_transformer_grid.yml` (extended with tokenizer paths).
- **Protocol:** Train several tokenizers, plug them into transformer experiments via `--tokenizer_path`, and compare tokenizer-level and model-level metrics.

### RQ5: Calibration and Uncertainty

- **Question:** How do different calibration methods and conformal prediction techniques affect the reliability of Khmer sentiment models?
- **Hypotheses:**
  - **H5a:** Post-hoc calibration (Platt, isotonic, temperature scaling) reduces ECE compared to uncalibrated models.
  - **H5b:** Temperature scaling gives a favorable balance between simplicity and calibration quality for the baseline.
  - **H5c:** Conformal prediction with calibrated scores achieves near-nominal coverage with compact prediction sets.
- **Code & Config:**
  - Calibration utilities: `modeling/calibration_utils.py`.
  - Conformal prediction: `modeling/conformal_predict.py`.
  - Calibration grid: `experiments/calibration_grid_baseline.yml`.
- **Protocol:** Use the calibration grid to select good settings, then study conformal coverage and set size using the saved calibrated scores.

---

## 2. Reproducibility and Tracking

### 2.1 Deterministic Setup

- All training scripts expose `--seed` (default 42 or 123 depending on script/config). Seeds are applied to Python, NumPy, and PyTorch (where applicable), and cuDNN deterministic flags are set.
- At each run, a snapshot of configuration and environment is saved to `<output_dir>/experiment_config.json`.

### 2.2 Experiment Tracking

- **Tracking backends:**
  - MLflow is enabled by default (`mlruns/` in the project root).
  - Weights & Biases (W&B) is optionally supported.
- **Common flags (baseline and transformer):**
  - `--tracking {none,mlflow,wandb}` (default `mlflow`).
  - `--experiment_name` (logical name for the run).
  - `--mlflow_tracking_uri` (optional; default is local `./mlruns`).
  - `--mlflow_experiment` (grouping under MLflow).
  - `--wandb_project`, `--wandb_entity`, `--wandb_mode` (for W&B).
  - `--config` (YAML experiment config; merged with CLI overrides).

**Artifacts logged:**
- `metrics.json` (core metrics and calibration details).
- Confusion matrices (baseline) as CSV.
- Model artifacts (e.g., vectorizers/models/tokenizers).

### 2.3 Example: Baseline with MLflow

```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_chargram \
  --experiment_name baseline_chargram \
  --tracking mlflow \
  --mlflow_experiment khmer_baselines \
  --seed 123 \
  --normalize_all

mlflow ui --backend-store-uri ./mlruns
```

### 2.4 Example: Transformer with MLflow

```bash
python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_base \
  --model_name xlm-roberta-base \
  --experiment_name transformer_xlmr \
  --tracking mlflow \
  --mlflow_experiment khmer_transformers \
  --seed 123 \
  --normalize_all
```

---

## 3. Khmer Text Normalization

The normalization module in `modeling/text_normalization.py` implements Khmer-specific rules exposed via CLI flags on training scripts.

### 3.1 Supported Operations

- Removal of zero-width characters (ZWSP, ZWNJ, ZWJ, BOM).
- Unicode NFC and diacritics reordering.
- Khmer digit mapping (U+17E0–U+17E9 → ASCII `0–9`).
- Khmer punctuation normalization (e.g., `។`, `៕`, `៖`).
- General punctuation normalization (quotes, dashes, ellipsis, repeated punctuation).
- Emoji handling: `keep` / `remove` / `map`.
- Elongation reduction (compressing > 2 repeats to 2).
- Whitespace normalization.
- Latin code-switch handling: `none` / `tag` / `strip` plus threshold.

### 3.2 CLI Flags

Main preset and components (supported by the normalization module and, where wired, by training scripts):

- `--normalize_all`
- `--norm_nfc`
- `--norm_whitespace`
- `--norm_punct`
- `--norm_elongation`
- `--norm_emoji {keep,remove,map}`
- `--norm_zero_width`
- `--norm_khmer_digits {keep,map}`
- `--norm_khmer_punct`
- `--norm_diacritics_reorder`
- `--norm_latin_action {none,tag,strip}`
- `--norm_latin_threshold <float>`

> Note: Some flags may need to be explicitly added to the argparse definitions in the training scripts if not already present; the normalization module itself supports them.

### 3.3 Recommended Normalization Protocol

1. Use `--normalize_all` as the default for all reported experiments.
2. For ablations, vary one or two normalization components at a time using `experiments/normalization_ablation_baseline.yml`.
3. Connect observed error patterns (e.g., from `modeling/error_analysis.py`) back to specific normalization operations when writing qualitative analysis.

---

## 4. Tokenizer Experiments

Tokenizer experiments are centered on `tools/train_tokenizer.py` and integration into transformer models via `modeling/train_transformer.py`.

### 4.1 Training a Tokenizer

```bash
# Unigram on the main CSV
python tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram \
  --vocab_size 16000 \
  --normalize_all

# BPE on text files
python tools/train_tokenizer.py \
  --input_txt data/corpus1.txt data/corpus2.txt \
  --output_dir tokenizers/bpe_kh_32k \
  --type bpe \
  --vocab_size 32000 \
  --norm_khmer_punct \
  --norm_khmer_digits map
```

Artifacts:
- `tokenizer.json` (used by Hugging Face `AutoTokenizer`).
- `tokenizer_report.json` (coverage proxy, average tokens, length distribution).
- Optionally, `normalization.json` describing training-time normalization flags.

### 4.2 Integrating Tokenizers into Transformer Training

- Extend `modeling/train_transformer.py` to accept an optional `--tokenizer_path` argument.
- Loading pattern:

```python
from transformers import AutoTokenizer

if args.tokenizer_path:
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
else:
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
```

- Log `tokenizer_path` and copy `tokenizer_report.json` into the run directory and experiment tracker.

### 4.3 Grid Configuration Pattern (`experiments/tokenizer_transformer_grid.yml`)

Use fields such as:

- `model_name`: e.g., `xlm-roberta-base`.
- `tokenizer_path`: `null` or a path like `tokenizers/unigram_kh_16k`.
- `norm_latin_action`: `none` / `tag`.

Example grid entries:
- Default tokenizer:
  - `model_name: xlm-roberta-base`
  - `tokenizer_path: null`
- Khmer Unigram 16k:
  - `model_name: xlm-roberta-base`
  - `tokenizer_path: tokenizers/unigram_kh_16k`

Run via:

```bash
python tools/run_experiment_grid.py \
  --config experiments/tokenizer_transformer_grid.yml
```

---

## 5. Baseline vs. Transformer Comparison Protocol

This protocol is designed for a central comparison that can be reported in a thesis or paper.

### 5.1 Common Setup

- **Dataset:** `annotation/sample_data/final_dataset.csv` with `--use_splits`.
- **Normalization:** `--normalize_all` for all models.
- **Random seed:** `--seed 123`.
- **Tracking:** `--tracking mlflow` with a shared `--mlflow_experiment` (e.g., `khmer_baseline_vs_transformer`).
- **Core metrics:** Accuracy and macro-F1 on validation and test.
- **Secondary metrics:** Per-class precision/recall/F1, confusion matrices, ECE, Brier score.
- **Efficiency metrics:** Training time, inference time per example, parameter count.

### 5.2 Baseline Configuration

- **Script:** `modeling/train_baseline.py`.
- **Model:** Char n-gram TF-IDF + Logistic Regression.
- **Typical hyperparameters:** n-grams 3–5, `min_df=2`, `class_weight=balanced`.
- **Calibration:** `calibrate` in `{none, platt, isotonic, temperature}`.

Example (strong baseline with temperature scaling):

```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_best \
  --experiment_name baseline_best \
  --tracking mlflow \
  --mlflow_experiment khmer_baseline_vs_transformer \
  --seed 123 \
  --normalize_all \
  --calibrate temperature
```

### 5.3 Transformer Configuration

- **Script:** `modeling/train_transformer.py`.
- **Backbones:** `xlm-roberta-base` (primary), optionally `bert-base-multilingual-cased`.
- **Typical hyperparameters:** `epochs=3`, `batch_size=16`, `lr=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`, `max_length=192`.

Example:

```bash
python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_best \
  --model_name xlm-roberta-base \
  --experiment_name xlmr_best \
  --tracking mlflow \
  --mlflow_experiment khmer_baseline_vs_transformer \
  --seed 123 \
  --normalize_all
```

### 5.4 Calibration and Uncertainty

- For the baseline, use `experiments/calibration_grid_baseline.yml` to identify good calibration strategies, then report ECE, Brier score, and reliability diagrams using `modeling/calibration_utils.py`.
- For transformers, apply temperature scaling or equivalent at evaluation, compute the same metrics, and compare.
- Optionally, apply `modeling/conformal_predict.py` on calibrated scores for both models and compare empirical coverage and prediction-set size.

### 5.5 Efficiency Measurement

- Log training start/end timestamps and hardware details (GPU/CPU, RAM, CUDA) inside scripts or via MLflow tags.
- For inference, run batched prediction over the test split and measure average latency per example and throughput.
- Record parameter counts from model summaries.

---

## 6. Experiment Grids (`experiments/`)

This section summarizes the main YAML grids and their intended research roles.

### 6.1 Normalization Ablation (Baseline)

- **Config:** `experiments/normalization_ablation_baseline.yml`.
- **Script:** `modeling/train_baseline.py`.
- **Purpose:** Study the effect of Khmer-specific normalization variants on a fixed baseline.
- **Fixed settings:**
  - Dataset: `annotation/sample_data/final_dataset.csv` with `--use_splits`.
  - Model: char n-gram TF-IDF + Logistic Regression.
  - Splits: 0.8 / 0.1 / 0.1.
  - Seed: 123.
  - Calibration: `calibrate=none`.
- **Grid parameters (example):**
  - `normalize_all: [true]`.
  - `norm_khmer_digits: ["keep", "map"]`.
  - `norm_khmer_punct: [false, true]`.
  - `norm_emoji: ["keep", "map"]`.

Run:

```bash
python tools/run_experiment_grid.py \
  --config experiments/normalization_ablation_baseline.yml
```

### 6.2 Calibration Grid (Baseline)

- **Config:** `experiments/calibration_grid_baseline.yml`.
- **Script:** `modeling/train_baseline.py`.
- **Purpose:** Compare calibration methods and CV strategies.
- **Fixed settings:** same dataset/splits/model as normalization ablation, `normalize_all=true`.
- **Grid parameters (example):**
  - `calibrate: ["none", "platt", "isotonic", "temperature"]`.
  - `calibrate_cv_folds: [0, 5]`.

Run:

```bash
python tools/run_experiment_grid.py \
  --config experiments/calibration_grid_baseline.yml
```

### 6.3 Transformer and Tokenizer Grid

- **Config:** `experiments/tokenizer_transformer_grid.yml`.
- **Script:** `modeling/train_transformer.py`.
- **Purpose:** Compare transformer backbones and Latin-handling/tokenizer strategies.
- **Fixed settings (example):**
  - Dataset: `annotation/sample_data/final_dataset.csv` with `--use_splits`.
  - Training: `epochs=3`, `batch_size=16`, `lr=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`, `max_length=192`.
  - Normalization: `normalize_all=true`.
- **Grid parameters (example):**
  - `model_name: ["xlm-roberta-base", "bert-base-multilingual-cased"]`.
  - `norm_latin_action: ["none", "tag"]`.
  - `tokenizer_path`: optionally included to switch between default and Khmer-specific tokenizers.

Run:

```bash
python tools/run_experiment_grid.py \
  --config experiments/tokenizer_transformer_grid.yml
```

---

## 7. Metrics and Evaluation

### 7.1 Baseline Metrics (`modeling/train_baseline.py`)

Saved in `metrics.json` and logged to the tracker:

- Validation and test accuracy.
- Validation and test macro-F1.
- Per-class precision/recall/F1 (via `classification_report`).
- Confusion matrix (`confusion_matrix.csv`).
- Calibration metrics (if calibration is enabled).

### 7.2 Transformer Metrics (`modeling/train_transformer.py`)

Stored under `metrics.json` and logged via Hugging Face `Trainer` integration:

- `eval_accuracy`.
- `eval_f1_macro` (macro-F1 via the `evaluate` library).

Where relevant, calibration and uncertainty metrics can be added based on outputs from `modeling/calibration_utils.py` and `modeling/conformal_predict.py`.

---

## 8. Thesis–Code Mapping

Use this mapping to reference specific scripts/configurations in a thesis or report.

### 8.1 Dataset and Annotation (e.g., Chapter 3)

- Annotation workflow: `annotation/README.md`, `annotation/guidelines.md`.
- Adjudication: `annotation/adjudicate.py`, `annotation/adjudication_log.md`.
- Curation and validation: `annotation/finalize_dataset.py`, `tools/validate_dataset.py`, `tools/ingest_dataset.py`.
- Sample finalized data: `annotation/sample_data/final_dataset.csv`.

### 8.2 Preprocessing and Normalization (e.g., Chapter 4)

- Implementation: `modeling/text_normalization.py`.
- Normalization configs per run: `normalization.json` in each `output_dir`.
- Ablation experiments: `experiments/normalization_ablation_baseline.yml`.

### 8.3 Baseline Models (e.g., Chapter 5)

- Training: `modeling/train_baseline.py`.
- Inference: `modeling/predict.py`.
- Calibration experiments: `experiments/calibration_grid_baseline.yml`.

### 8.4 Transformer Models and Tokenizers (e.g., Chapter 6)

- Transformer training: `modeling/train_transformer.py`.
- Transformer inference: `modeling/predict_transformer.py`.
- Tokenizer training: `tools/train_tokenizer.py`.
- Tokenizer/transformer grids: `experiments/tokenizer_transformer_grid.yml`.

### 8.5 Calibration, Uncertainty, and Error Analysis (e.g., Chapter 7)

- Calibration utilities: `modeling/calibration_utils.py`.
- Conformal prediction: `modeling/conformal_predict.py`.
- Error & stress evaluation: `modeling/error_analysis.py`, `modeling/stress_eval.py`.
- Group analyses: `annotation/sample_data/groups.csv`, `tests/test_group_splits.py`.

### 8.6 Experimental Protocol and Reproducibility (Appendix)

- Experiment utilities and config snapshots: `modeling/utils/experiment.py`, `<output_dir>/experiment_config.json`.
- Experiment grids: `experiments/*.yml`.
- Grid runner: `tools/run_experiment_grid.py`.
- Quickstart examples: project root `README.md`.

---

## 9. Practical Checklists

Use these checklists before finalizing results and writing.

### 9.1 Data and Preprocessing

- [ ] Dataset validated with `tools/validate_dataset.py`.
- [ ] Final splits fixed and documented (`--use_splits`).
- [ ] Normalization config saved (`normalization.json`) next to each model/tokenizer.

### 9.2 Experiments and Tracking

- [ ] All reported experiments run with `--tracking mlflow` or W&B.
- [ ] `experiment_config.json` present in each `output_dir`.
- [ ] MLflow/W&B runs tagged with relevant RQ identifiers (e.g., `rq1_normalization`).

### 9.3 Reporting

- [ ] Ablation results (normalization, tokenizer) summarized in tables with clear configs.
- [ ] Baseline vs. transformer comparison includes accuracy, macro-F1, calibration, and efficiency.
- [ ] Error and stress analyses tied to normalization and tokenizer design.
- [ ] Thesis/report explicitly references the corresponding scripts and configs from this repository.
