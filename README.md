# Khmer Sentiment Analysis

End-to-end pipeline for Khmer sentiment classification with dataset tools, baseline and transformer models, error/stress evaluation, and uncertainty calibration.

Recent updates:
- Step 1: Reproducibility & Tracking
  - Global seeding and deterministic settings
  - Centralized experiment tracking (MLflow default; W&B optional)
  - Config snapshots and artifact logging
- Step 2: Khmer Normalization & Tokenizer
  - Khmer-aware normalization (digits, punctuation, zero-width, diacritics, emoji, elongation, latin code-switch)
  - Tokenizer training utility (Unigram/BPE) with normalization integration

## Quickstart

Install:
```bash
pip install -r requirements.txt
# CPU Torch on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Train baseline (with tracking):
```bash
python modeling/train_baseline.py \
  --input annotation/sample_data/final_dataset.csv \
  --output_dir runs/baseline_chargram \
  --experiment_name baseline_chargram --tracking mlflow \
  --seed 123 --normalize_all
```

Train transformer:
```bash
python modeling/train_transformer.py \
  --input annotation/sample_data/final_dataset.csv --use_splits \
  --output_dir runs/xlmr_base --model_name xlm-roberta-base \
  --experiment_name transformer_xlmr --tracking mlflow \
  --seed 123 --normalize_all
```

Train a Khmer tokenizer:
```bash
python tools/train_tokenizer.py \
  --input_csv annotation/sample_data/final_dataset.csv \
  --text_column text \
  --output_dir tokenizers/unigram_kh_16k \
  --type unigram --vocab_size 16000 --normalize_all
```

## Documentation
- User Guide: docs/user_guide.md
- Research Guide: docs/research_guide.md
- Annotation Workflow: annotation/README.md and annotation/guidelines.md

## Tests
Run unit tests:
```bash
pytest -q
```

## Notes
- Normalization configs are saved as `normalization.json` in each model/tokenizer directory to ensure reproducibility.
- MLflow UI can be launched with `mlflow ui --backend-store-uri ./mlruns`.
- For W&B, run `wandb login` to enable remote tracking.
