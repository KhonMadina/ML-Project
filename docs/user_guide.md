# User Guide: Khmer Sentiment Analysis Pipeline

This guide explains how to use the pipeline end-to-end on Windows or any Python environment.

Contents
- Prerequisites
- Quick Start
- Using Your Own Data
- Group-aware Splits (Leakage-safe)
- Training and Calibration
- Predictions and Evaluation
- Error Analysis and Slicing
- Experiment Comparison

## Prerequisites
- Python 3.8+
- Recommended: virtual environment
- Install dependencies (reproducible):

  pip install -r requirements.txt

Optional (plots):

  pip install matplotlib

Testing (optional):

  pip install -r requirements-dev.txt

## Quick Start
Run the Windows demo (installs from requirements.txt, generates artifacts):

  run_demo.bat

Outputs:
- annotation/sample_data/final_dataset.csv
- annotation/sample_data/final_{train,val,test}.csv
- models/baseline_chargram/{vectorizer.pkl, model.pkl, metrics.json}
- reports/baseline_chargram/{misclassified.csv, error_analysis_*, reliability_diagram.png?, calibration_bins.csv?}

## Using Your Own Data
- Prepare CSV with columns: id,text,annotator,label
- Or export JSON from Label Studio with label choices POS/NEG/NEU.
- Compute IAA and generate disagreement queue:

  python annotation/adjudicate.py iaa --input data/annot.csv
  python annotation/adjudicate.py queue --input data/annot.csv --log annotation/adjudication_log.md --limit 100

- Finalize single-label dataset and export splits:

  python annotation/finalize_dataset.py combine --input data/annot.csv --log annotation/adjudication_log.md --output data/final_dataset.csv --strategy majority --export-splits

## Group-aware Splits (Leakage-safe)
To prevent leakage when multiple items come from the same user/thread:
1) Create a mapping CSV data/groups.csv with columns: id,group (or id,<custom>)
2) Finalize with group propagation and group-aware split:

  python annotation/finalize_dataset.py combine \
    --input data/annot.csv \
    --log annotation/adjudication_log.md \
    --output data/final_dataset.csv \
    --strategy majority \
    --groups_csv data/groups.csv --group_column group \
    --export-splits --group_aware_splits

Later, pass --group_column to training.

## Training and Calibration
Train with predefined splits:

  python modeling/train_baseline.py --input data/final_dataset.csv --use_splits --output_dir models/baseline_chargram

Enable probability calibration (Platt or Isotonic):

  python modeling/train_baseline.py --input data/final_dataset.csv --use_splits --output_dir models/baseline_chargram_cal --calibrate isotonic

Use group-aware training and sampling options:

  python modeling/train_baseline.py --input data/final_dataset.csv --use_splits --output_dir models/baseline_chargram_cal --group_column group --calibrate platt --resample oversample --resample_ratio 1.0

Custom class weights via JSON mapping (overrides --class_weight):

  python modeling/train_baseline.py --input data/final_dataset.csv --use_splits --output_dir models/baseline_chargram_w --class_weight_json data/class_weights.json

## Predictions and Evaluation
Single input text:

  python modeling/predict.py --model_dir models/baseline_chargram --text "អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄"

Batch evaluation (with labels):

  python modeling/predict.py --model_dir models/baseline_chargram --input_csv data/final_test.csv --output_csv data/pred_test.csv

## Error Analysis and Slicing
Generate misclassifications, metrics, and reliability plots:

  python modeling/error_analysis.py --model_dir models/baseline_chargram --input_csv data/final_test.csv --output_dir reports/baseline_chargram

Per-slice metrics (e.g., by group) and reliability bins:

  python modeling/error_analysis.py --model_dir models/baseline_chargram_cal --input_csv data/final_test.csv --output_dir reports/baseline_chargram_cal --slice_column group --reliability_bins 15

## Experiment Comparison
Aggregate results across models:

  python tools/compare_experiments.py --glob "models/*"

This prints a CSV table to stdout with accuracy/F1 and sizes for quick comparison.
