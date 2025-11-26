# Experiment Playbooks: Commands and Protocols (Upgraded)

This document maps research questions (RQs) to exact commands, configs, and reporting steps. It is aligned with the upgraded tooling for multi-seed aggregation, statistical testing, robustness deltas, and consolidated reporting.

Prereqs
- Install requirements: `pip install -r requirements.txt`
- Optional tracking: MLflow UI `mlflow ui --backend-store-uri ./mlruns`
- Dataset ready: `annotation/sample_data/final_dataset.csv` (or your dataset) with `final_{train,val,test}.csv` next to it.

Notation
- DATA_DIR: path to folder holding final_dataset.csv and split CSVs
- OUT: runs/<label>

---

## Multi-seed runs and aggregation (recommended)

Run any grid with multiple seeds and aggregate metrics with 95% CIs:
```
python tools/run_experiment_grid.py \
  --config experiments/backbone_efficiency_grid.yml \
  --seeds 123 456 789 \
  --aggregate
```
- Outputs: `<output_root>/aggregate_summary.json|csv` and `checksums.json`.
- Optional post-run paired bootstrap CI on pairs of runs (directories or CSVs):
```
python tools/run_experiment_grid.py \
  --config experiments/backbone_efficiency_grid.yml \
  --seeds 123 456 789 \
  --aggregate \
  --post_ci_pairs runs/backbone_efficiency_grid/run_001,runs/backbone_efficiency_grid/run_002
```
- CI outputs: `<output_root>/ci/pair_XX/bootstrap_summary.json`.

---

## RQ1: Khmer Normalization Ablation (Baseline)

Grid config: `experiments/normalization_ablation_baseline.yml`

Single run example:
```
python modeling/train_baseline.py \
  --input DATA_DIR/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_norm_full \
  --normalize_all \
  --experiment_name baseline_norm_full \
  --tracking mlflow \
  --mlflow_experiment khmer_baselines \
  --seed 123
```

Paired bootstrap CI between two trained baselines (using test_predictions.csv):
```
python tools/paired_bootstrap_ci.py \
  --predictions_a runs/baseline_norm_full/test_predictions.csv \
  --predictions_b runs/baseline_chargram/test_predictions.csv \
  --output reports/ci_norm_full_vs_chargram \
  --n_bootstrap 10000 --alpha 0.05 --seed 123
```

---

## RQ2: Tokenizer Variants with Transformers

Train tokenizer variants (grid):
```
python tools/train_tokenizer.py \
  --input_csv DATA_DIR/final_dataset.csv \
  --text_column text \
  --output_root tokenizers/kh \
  --grid --types unigram bpe --vocab_sizes 8000 16000 32000 \
  --normalize_all --seed 123
```

Transformer with custom tokenizer:
```
python modeling/train_transformer.py \
  --input DATA_DIR/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_unigram16k \
  --model_name xlm-roberta-base \
  --tokenizer_path tokenizers/kh/unigram_16000 \
  --normalize_all \
  --lang_column lang \
  --experiment_name tok_xlmr_unigram16k \
  --tracking mlflow \
  --mlflow_experiment khmer_transformers \
  --seed 123
```

Compare against default tokenizer runs (no `--tokenizer_path`).

---

## RQ3: Baselines vs Transformers + Efficiency

Baseline run:
```
python modeling/train_baseline.py \
  --input DATA_DIR/final_dataset.csv \
  --use_splits \
  --output_dir runs/baseline_chargram \
  --normalize_all \
  --experiment_name baseline_chargram \
  --tracking mlflow \
  --mlflow_experiment khmer_baselines \
  --seed 123
```

Small backbone run:
```
python modeling/train_transformer.py \
  --input DATA_DIR/final_dataset.csv \
  --use_splits \
  --output_dir runs/distilmbert_small \
  --model_name distilbert-base-multilingual-cased \
  --normalize_all \
  --experiment_name distilmbert_small \
  --tracking mlflow \
  --mlflow_experiment khmer_transformers \
  --seed 123
```

Paired bootstrap CI and McNemar:
```
python tools/paired_bootstrap_ci.py \
  --predictions_a runs/baseline_chargram/test_predictions.csv \
  --predictions_b runs/distilmbert_small/test_predictions.csv \
  --output reports/ci_baseline_vs_small \
  --n_bootstrap 10000 --alpha 0.05 --seed 123

python tools/mcnemar_test.py \
  --predictions_a runs/baseline_chargram/test_predictions.csv \
  --predictions_b runs/distilmbert_small/test_predictions.csv \
  --output reports/mcnemar_baseline_vs_small
```

---

## RQ4: Bilingual Routing (Dual-Head)

Single-head baseline:
```
python modeling/train_transformer.py \
  --input DATA_DIR/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_single \
  --model_name xlm-roberta-base \
  --lang_column lang \
  --normalize_all \
  --experiment_name xlmr_single \
  --tracking mlflow \
  --mlflow_experiment khmer_transformers \
  --seed 123
```

Dual-head run:
```
python modeling/train_transformer.py \
  --input DATA_DIR/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_dualhead \
  --model_name xlm-roberta-base \
  --lang_column lang \
  --dual_head \
  --normalize_all \
  --experiment_name xlmr_dualhead \
  --tracking mlflow \
  --mlflow_experiment khmer_transformers \
  --seed 123
```

Paired tests (optional): use paired bootstrap and McNemar as in RQ3.

---

## RQ5: Calibration and Conformal

Calibration focus (transformer):
```
python modeling/train_transformer.py \
  --input DATA_DIR/final_dataset.csv \
  --use_splits \
  --output_dir runs/xlmr_calib \
  --model_name xlm-roberta-base \
  --normalize_all \
  --experiment_name xlmr_calib \
  --tracking mlflow \
  --mlflow_experiment khmer_transformers \
  --seed 123
```

Conformal sets (baseline models using predict_proba):
```
python modeling/conformal_predict.py \
  --model_dir runs/baseline_chargram \
  --input_csv DATA_DIR/final_test.csv \
  --output_csv runs/baseline_chargram/conformal_test.csv \
  --target_coverage 0.9
```

---

## RQ6: Robustness Sweeps

Clean evaluation (no preset):
```
python modeling/stress_eval.py \
  --transformer_model_dir runs/xlmr_dualhead \
  --input_csv DATA_DIR/final_test.csv \
  --output_dir runs/xlmr_dualhead_stress/clean \
  --batch_size 32 \
  --seed 123
```

Stress evaluations (run per preset/severity):
```
python modeling/stress_eval.py \
  --transformer_model_dir runs/xlmr_dualhead \
  --input_csv DATA_DIR/final_test.csv \
  --output_dir runs/xlmr_dualhead_stress/diacritics_s3 \
  --preset diacritics \
  --severity 0.7 \
  --prob 0.7 \
  --seed 123

python modeling/stress_eval.py \
  --transformer_model_dir runs/xlmr_dualhead \
  --input_csv DATA_DIR/final_test.csv \
  --output_dir runs/xlmr_dualhead_stress/emoji_s1 \
  --preset emoji_burst \
  --severity 0.3 \
  --prob 0.7 \
  --seed 123
```

Compare deltas against clean:
```
python tools/compare_stress_runs.py \
  --clean_dir runs/xlmr_dualhead_stress/clean \
  --stress_dirs runs/xlmr_dualhead_stress/diacritics_s3 runs/xlmr_dualhead_stress/emoji_s1 \
  --output runs/xlmr_dualhead_stress
```
- Outputs: robustness_summary.json/csv and robustness_per_category.csv.

---

## Statistics & Reporting

Paired bootstrap CI on predictions:
```
python tools/paired_bootstrap_ci.py \
  --predictions_a runs/xlmr_unigram16k/test_predictions.csv \
  --predictions_b runs/xlmr_single/test_predictions.csv \
  --output reports/ci_tok_vs_default \
  --n_bootstrap 10000 --alpha 0.05 --seed 123
```

McNemar's test on paired predictions:
```
python tools/mcnemar_test.py \
  --predictions_a runs/xlmr_unigram16k/test_predictions.csv \
  --predictions_b runs/xlmr_single/test_predictions.csv \
  --output reports/mcnemar_tok_vs_default
```

Consolidated report export:
```
python tools/export_report.py \
  --glob "runs/*" \
  --out_dir reports/final \
  --pred_pair runs/baseline_chargram/test_predictions.csv:runs/xlmr_base/test_predictions.csv:xlmr_vs_baseline \
  --stress runs/xlmr_base:runs/xlmr_base_stress/clean/stress_metrics.json \
  --stress runs/baseline_chargram:runs/baseline_chargram_stress/clean/stress_metrics.json
```

Notes
- Prefer ≥3 seeds for primary claims and aggregate with the grid runner.
- Keep MLflow experiments organized by family (khmer_baselines, khmer_transformers).
- Ensure test_predictions.csv exists for statistical tests (both baseline and transformer scripts export this now).
