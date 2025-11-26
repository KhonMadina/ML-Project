# Khmer Sentiment Analysis

End-to-end pipeline for Khmer–English sentiment classification with graduate-level research rigor.

Includes:
- Dataset annotation, validation, and governance
- Baseline character n-gram models
- Multilingual transformer models (with optional language-routed dual-head)
- Khmer-aware text normalization and tokenizer training (Unigram/BPE)
- Calibration (temperature/isotonic) and conformal prediction
- Stress/robustness and error analysis
- Reproducible experiment grids with tracking (MLflow/W&B)

Recent upgrades:
- Research protocol, dataset card, and experiment playbooks
- Bilingual features, per-language metrics, and calibration summaries
- System diagrams clarifying pipeline and components

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
- `--use_splits` for pre-defined train/val/test splits next to the input CSV
- `--group_column`, `--group_stratified` for group-aware splitting
- `--calibrate {none,platt,isotonic,temperature}`; `--calibrate_cv_folds` for calibration

Outputs:
- `metrics.json`, `confusion_matrix.csv`, serialized vectorizer/model
- `experiment_config.json`, `normalization.json`

---

### 3. Train a Transformer Model

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

Core flags: `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--warmup_ratio`, `--max_length`, `--grad_accum`, `--fp16`, `--tokenizer_path`.

---

### 4. Train a Khmer Tokenizer

```bash
python tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram \
  --vocab_size 16000 \
  --normalize_all
```

Artifacts: `tokenizer.json`, `tokenizer_report.json`, `normalization.json`.

---

### 5. Inference & Stress Evaluation

```bash
python modeling/predict.py --model_dir runs/baseline_chargram --text "មតិយោបល់របស់អ្នកនៅទីនេះ"

python modeling/stress_eval.py \
  --model_dir runs/baseline_chargram \
  --input_csv data/stress_tests.csv \
  --output_dir runs/baseline_chargram_stress
```

---

## Research Protocol and Academic Checklist

- Methodology and RQs: see `docs/research_guide.md`
- System diagrams: see `docs/system_diagrams.md`
- Experiment playbooks (commands/grids): `docs/experiment_playbooks.md`
- Dataset card: `reports/dataset_card.md`

Checklist (thesis-grade):
- [ ] Dataset card with sources, license, ethics, IAA (κ/α with CI)
- [ ] Group-aware fixed splits validated; checksums saved
- [ ] Multi-seed runs for primary comparisons; average and 95% CIs
- [ ] Statistical tests (paired bootstrap, McNemar) with corrections
- [ ] Calibration (ECE/Brier) with reliability diagrams
- [ ] Robustness sweeps across stress presets and severities
- [ ] Error taxonomy and qualitative analyses with examples

---

## Experiments & Reproducibility

Run grids with:

```bash
python tools/run_experiment_grid.py --config <experiments/*.yml>
```

Main grids:
- `experiments/normalization_ablation_baseline.yml`
- `experiments/calibration_grid_baseline.yml`
- `experiments/tokenizer_transformer_grid.yml`
- (add) `experiments/robustness_sweep.yml`, `experiments/backbone_efficiency_grid.yml`, `experiments/bilingual_vs_monolingual.yml`

Use MLflow UI to browse under `./mlruns`.

---

## Project Structure (High-Level)

- `annotation/` – data guidelines, adjudication, dataset curation
- `docs/` – user guide, research guide, experiment playbooks, system diagrams
- `experiments/` – YAML grids
- `modeling/` – training, prediction, normalization, calibration, error/stress
- `tools/` – grid runner, tokenizer training, dataset ingest/validate, reporting
- `tests/` – unit tests

---

## Tests

```bash
pytest -q
```

---

## Notes

- Normalization and experiment configs are saved per run directory.
- Use `mlflow ui --backend-store-uri ./mlruns` for local tracking.
- For W&B, `wandb login` and set `--tracking wandb`.
- See `docs/research_guide.md` for thesis mapping.
