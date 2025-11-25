# User Guide: Khmer + English Sentiment Analysis

This guide explains how to install dependencies, validate and ingest datasets, train baseline and transformer models, use Khmer-specific normalization, run bilingual experiments, perform stress testing, compute statistical comparisons, and export final reports.

For experiment design and thesis mapping, see docs/research_guide.md.

---

## 1. Installation

From the project root:

```bash
pip install -r requirements.txt

# CPU-only PyTorch on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or follow https://pytorch.org/get-started/locally/ for CUDA
```

Optional tracking tools:
- MLflow UI: mlflow ui --backend-store-uri ./mlruns
- Weights & Biases: pip install wandb && wandb login

---

## 2. Dataset Validation, Ingestion, and IAA

### 2.1 Bilingual-aware validation

Validate CSV schema and quality. Supported columns:
- Required: id, text, label
- Optional: lang, group, split, source

```bash
python tools/validate_dataset.py \
  --input annotation/sample_data/final_dataset.csv \
  --lang_column lang --allowed_langs km en \
  --split_column split --group_column group \
  --output_clean annotation/sample_data/final_dataset_clean.csv
```

Checks include unique ids, empty text, label set, allowed languages, split validity, group leakage warnings, and a code-switch summary (km, en, km+en, other).

### 2.2 Ingest a new dataset

Run validation, finalize to final_dataset.csv, and optionally export splits:

```bash
python tools/ingest_dataset.py \
  --raw_csv path/to/annotations.csv \
  --output_dir annotation/sample_data \
  --lang_column lang --allowed_langs km en \
  --groups_csv annotation/sample_data/groups.csv --group_column group \
  --export_splits --group_aware_splits
```

### 2.3 Inter-annotator agreement (IAA)

Compute Cohen’s kappa overall and per language:

```bash
python tools/compute_iaa.py \
  --input annotation/annotations_raw.csv \
  --lang_column lang \
  --output_json reports/annotation_iaa.json
```

---

## 3. Reproducibility and Tracking

Common flags (baseline and transformer):
- --tracking {none,mlflow,wandb}
- --experiment_name
- --mlflow_tracking_uri, --mlflow_experiment
- --wandb_project, --wandb_entity, --wandb_mode
- --config (YAML config), --seed (default 42)

Artifacts, metrics, and config snapshots are saved under --output_dir and logged to the tracker if enabled.

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

## 4. Baseline Model

Character n-gram baseline with Khmer normalization:

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

Options:
- Splits: --use_splits or random splits with --train_ratio/--val_ratio/--test_ratio
- Group-aware: --group_column and --group_stratified
- Calibration: --calibrate {none,platt,isotonic,temperature}
- Resampling: --resample {none,undersample,oversample}

Outputs: metrics.json (acc, macro-F1, calibration), confusion_matrix.csv, serialized vectorizer/model, experiment_config.json.

---

## 5. Transformer Models (Bilingual-enabled)

Fine-tune multilingual backbones (e.g., XLM-R):

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

Key hyperparameters: --epochs, --batch_size, --lr, --weight_decay, --warmup_ratio, --max_length, --grad_accum, --fp16. Optional --tokenizer_path for a custom tokenizer.

Bilingual/robustness options:
- --lang_column lang: name of language column in CSV
- --langs km en: explicit language list (inferred if omitted)
- --stratify_by_lang: stratify random splits by (label,lang)
- --dual_head: shared encoder with language-routed heads
- --class_weighting {none,balanced}
- --focal_loss [--focal_gamma 2.0]

Per-language metrics and calibration (ECE, Brier) are computed and saved (plots and JSON) in the run directory.

---

## 6. Khmer Text Normalization

Use --normalize_all as a strong default. Advanced flags (depending on script wiring):
- --norm_nfc, --norm_whitespace, --norm_punct, --norm_elongation
- --norm_emoji {keep,remove,map}
- --norm_zero_width, --norm_khmer_digits {keep,map}
- --norm_khmer_punct, --norm_diacritics_reorder
- --norm_latin_action {none,tag,strip}, --norm_latin_threshold <float>

Normalization config is saved as normalization.json with each run.

---

## 7. Tokenizers

Train a Khmer-specific tokenizer and plug it into transformers:

```bash
python tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram --vocab_size 16000 \
  --normalize_all

python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_unigram_16k \
  --model_name xlm-roberta-base \
  --tokenizer_path tokenizers/unigram_kh_16k \
  --normalize_all
```

Artifacts: tokenizer.json, tokenizer_report.json (copied to run dir).

---

## 8. Inference, Stress Tests, and Error Analysis

### 8.1 Baseline inference

```bash
python modeling/predict.py \
  --model_dir runs/baseline_chargram \
  --text "មតិយោបល់របស់អ្នកនៅទីនេះ"
```

### 8.2 Stress evaluation (expanded presets)

Supports baseline or transformer checkpoints; presets with severity and probability:
- diacritics, zero_width, emoji_burst, elongation, mixed_script, code_switch, latinized_khmer

```bash
# Baseline
python modeling/stress_eval.py \
  --model_dir runs/baseline_chargram \
  --input_csv data/stress_tests.csv \
  --output_dir runs/baseline_chargram_stress \
  --preset diacritics --severity 0.7 --prob 0.8 --save_misclassified

# Transformer
python modeling/stress_eval.py \
  --transformer_model_dir runs/xlmr_base \
  --input_csv data/stress_tests.csv \
  --output_dir runs/xlmr_stress \
  --preset code_switch --severity 0.5 --prob 0.5
```

### 8.3 Error analysis

Use modeling/error_analysis.py to inspect error types and group-based slices.

---

## 9. Aggregated Reporting and CIs

Export CSVs, plots, and markdown summaries across models; optional paired bootstrap CIs and stress deltas:

```bash
python tools/export_report.py \
  --glob "models/*" \
  --out_dir reports/final \
  --pred_pair runs/baseline/pred_test.csv:runs/xlmr/pred_test.csv:xlmr_vs_baseline \
  --stress models/xlmr_base:reports/stress_xlmr/stress_metrics.json
```

Outputs:
- overall.csv, per_language.csv, calibration.csv
- plots/bar_test_f1.png, plots/bar_test_ece.png (if matplotlib available)
- summary.md (top models table)
- bootstrap/*.json for paired CIs (if requested)

---

## 10. Tips for Reliable Results

- Fix splits (--use_splits) and seed (--seed) for fair comparisons.
- Track all runs (MLflow/W&B) and save normalization/tokenizer artifacts.
- When comparing models, vary one factor at a time.
- Report calibration (ECE, Brier) and efficiency (time, throughput, params) alongside accuracy and macro-F1.
- For bilingual scope, report per-language metrics and robustness (stress presets, code-switch slices).
