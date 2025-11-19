# Research Guide: Khmer Sentiment Analysis

This guide summarizes recommended research protocols and how to reproduce experiments in this repository. It reflects Step 1 (Reproducibility & Tracking), Step 2 (Khmer normalization & tokenizer), and the finalized experimental grids and documentation alignment for thesis work.

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

## 5. Final Experimental Grids

Step 2 (Define and Implement Experimental Program) is encoded as YAML experiment grids under `experiments/` and executed via `tools/run_experiment_grid.py`. These grids correspond to the main research axes in the thesis.

### 5.1 Normalization Ablation (Baseline)

Config: `experiments/normalization_ablation_baseline.yml`

- **Script**: `modeling/train_baseline.py`
- **Purpose**: Study the impact of Khmer-specific normalization operations on baseline performance.
- **Fixed settings** (selected):
  - Dataset: `annotation/sample_data/final_dataset.csv` with `--use_splits`.
  - Model: char n-gram TF-IDF + Logistic Regression.
  - Splits: train/val/test = 0.8/0.1/0.1.
  - N-grams: 3–5; `min_df=2`; `class_weight=balanced`; `calibrate=none`.
  - Seed: 123.
- **Grid parameters**:
  - `normalize_all: [true]`.
  - `norm_khmer_digits: ["keep", "map"]`.
  - `norm_khmer_punct: [false, true]`.
  - `norm_emoji: ["keep", "map"]`.

Run:
```bash
python tools/run_experiment_grid.py \
  --config experiments/normalization_ablation_baseline.yml
```

Each run writes to `runs/norm_ablation_baseline/run_XXX...` and logs parameters + metrics to MLflow (experiment `khmer_norm_ablation`).

### 5.2 Calibration & Uncertainty (Baseline)

Config: `experiments/calibration_grid_baseline.yml`

- **Script**: `modeling/train_baseline.py`
- **Purpose**: Compare calibration methods and CV strategies for the baseline model.
- **Fixed settings** (selected):
  - Same dataset/splits as normalization ablation.
  - Normalization: `normalize_all=true`.
  - Model: same n-gram + Logistic Regression baseline.
- **Grid parameters**:
  - `calibrate: ["none", "platt", "isotonic", "temperature"]`.
  - `calibrate_cv_folds: [0, 5]`.

Run:
```bash
python tools/run_experiment_grid.py \
  --config experiments/calibration_grid_baseline.yml
```

Each run logs calibration settings and resulting metrics to MLflow (experiment `khmer_calibration`).

### 5.3 Tokenizer and Transformer Variants

Config: `experiments/tokenizer_transformer_grid.yml`

- **Script**: `modeling/train_transformer.py`
- **Purpose**: Compare multilingual transformer backbones and Khmer Latin-handling strategies; extendable to custom tokenizers.
- **Fixed settings** (selected):
  - Dataset: `annotation/sample_data/final_dataset.csv` with `--use_splits`.
  - Training: `epochs=3`, `batch_size=16`, `lr=2e-5`, `weight_decay=0.01`, `warmup_ratio=0.1`.
  - Max length: 192; `grad_accum=1`; `fp16=false`; `seed=123`.
  - Normalization: `normalize_all=true`.
- **Grid parameters**:
  - `model_name: ["xlm-roberta-base", "bert-base-multilingual-cased"]`.
  - `norm_latin_action: ["none", "tag"]`.

Run:
```bash
python tools/run_experiment_grid.py \
  --config experiments/tokenizer_transformer_grid.yml
```

To integrate custom tokenizers, add their directories to `model_name` (if compatible with `AutoTokenizer.from_pretrained`).

## 6. Metrics and Tests

Across all experiments, the primary **evaluation metrics** are:

- **Baseline (train_baseline.py)**:
  - Validation/Test accuracy.
  - Validation/Test macro-F1.
  - Per-class precision/recall/F1 (via `classification_report`).
  - Confusion matrices (saved as `confusion_matrix.csv`).
  - Calibration metadata (if enabled) saved in `metrics.json` under `"calibration"`.

- **Transformers (train_transformer.py)**:
  - Evaluation metrics per HuggingFace `Trainer`:
    - `eval_accuracy`.
    - `eval_f1_macro` (macro-F1, via `evaluate` library).
  - These are stored in `metrics.json` and logged via the experiment tracker.

Additional statistical rigor (confidence intervals, significance testing) should be added at the analysis stage, using the MLflow/W&B runs and saved `metrics.json` files as input.

## 7. Thesis–Code Mapping

This section suggests a mapping between typical thesis chapters/sections and concrete code artifacts to aid traceability. Adjust section numbers/titles to your actual thesis.

### 7.1 Dataset and Annotation (Thesis Chapter 3)

- **Code/Artifacts**:
  - Annotation workflow: `annotation/README.md`, `annotation/guidelines.md`.
  - Adjudication scripts/logs: `annotation/adjudicate.py`, `annotation/adjudication_log.md`.
  - Dataset curation & validation: `annotation/finalize_dataset.py`, `tools/validate_dataset.py`, `tools/ingest_dataset.py`.
  - Sample finalized data: `annotation/sample_data/final_dataset.csv` and `final_train/val/test.csv`.

### 7.2 Preprocessing and Normalization (Thesis Chapter 4)

- **Code/Artifacts**:
  - Normalization implementation: `modeling/text_normalization.py`.
  - Normalization configs saved per model: `normalization.json` in each `output_dir`.
  - Normalization ablation experiments: `experiments/normalization_ablation_baseline.yml` + corresponding runs in MLflow (`khmer_norm_ablation`).

### 7.3 Baseline Models (Thesis Chapter 5)

- **Code/Artifacts**:
  - Baseline training: `modeling/train_baseline.py`.
  - Baseline inference/evaluation: `modeling/predict.py`.
  - Calibration and resampling options integrated in baseline script.
  - Calibration experiments: `experiments/calibration_grid_baseline.yml` + MLflow experiment `khmer_calibration`.

### 7.4 Transformer Models and Tokenizers (Thesis Chapter 6)

- **Code/Artifacts**:
  - Transformer training: `modeling/train_transformer.py`.
  - Transformer inference: `modeling/predict_transformer.py`.
  - Tokenizer training utility: `tools/train_tokenizer.py` and resulting `tokenizers/*/tokenizer.json`.
  - Tokenizer/transformer experiments: `experiments/tokenizer_transformer_grid.yml` + MLflow experiment `khmer_tokenizer`.

### 7.5 Calibration, Uncertainty, and Error Analysis (Thesis Chapter 7)

- **Code/Artifacts**:
  - Calibration utilities: `modeling/calibration_utils.py`.
  - Conformal prediction: `modeling/conformal_predict.py`.
  - Error and stress evaluation: `modeling/error_analysis.py`, `modeling/stress_eval.py`.
  - Group-based analyses: `annotation/sample_data/groups.csv`, `tests/test_group_splits.py`.
  - Calibration experiment runs: see MLflow experiment `khmer_calibration` and associated `metrics.json` files.

### 7.6 Experimental Protocol and Reproducibility (Thesis Appendix)

- **Code/Artifacts**:
  - Experiment utilities and config snapshots: `modeling/utils/experiment.py` and `<output_dir>/experiment_config.json`.
  - Experiment grids: all files under `experiments/`.
  - Grid runner: `tools/run_experiment_grid.py`.
  - Reproducibility examples: Section 1 of this guide and Quickstart examples in the project `README.md`.

This mapping can be referenced in the thesis appendix or methodology section to demonstrate how each reported result corresponds to specific scripts, configurations, and tracked runs, ensuring tight coupling between the written document and this repository.

## 8. Checklists
- [ ] Dataset validated with `tools/validate_dataset.py`.
- [ ] Normalization config saved (`normalization.json`) next to model/tokenizer.
- [ ] Experiment tracking on (MLflow or W&B) with `experiment_config.json` snapshot per run.
- [ ] Tokenizer artifacts versioned by vocab size and normalization setup.
- [ ] Report includes ablations and error/stress analysis tied to normalization/tokenizer choices.
- [ ] Experimental grids under `experiments/` correspond to thesis experimental sections and are referenced in the thesis.
