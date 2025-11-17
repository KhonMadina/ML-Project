# Khmer Sentiment Analysis Pipeline (POS/NEG/NEU)

End-to-end workflow for annotating, adjudicating, training, evaluating, and curating a Khmer sentiment dataset.

Key features:
- Annotation toolkit with adjudication and IAA (Cohen's κ)
- Dataset finalization with decisions and resolution strategies
- Optional group-aware splits to prevent leakage (user/thread-level)
- Baseline model: char n-gram TF‑IDF + Logistic Regression
- Optional probability calibration (Platt/Isotonic)
- Enhanced error analysis: calibration (ECE/MCE, reliability), per-slice metrics, top features
- Reproducible installs via requirements.txt

## Project Structure

- annotation/
  - adjudication_log.md — record adjudication decisions and action items
  - guidelines.md — annotation rules and examples
  - label_config.json — Label Studio configuration
  - README.md — annotation toolkit usage
  - adjudicate.py — build disagreement queues and compute IAA (κ)
  - finalize_dataset.py — resolve labels and produce finalized dataset (+ optional splits)
  - curate_from_errors.py — turn model misclassifications into a review queue and log stubs
  - sample_data/
    - demo_annotations.csv — small demo dataset with 2 annotators
- modeling/
  - train_baseline.py — character n-gram TF-IDF + Logistic Regression baseline (group-aware split, calibration, experiment logging)
  - predict.py — inference and batch evaluation
  - error_analysis.py — misclassification reports, calibration (ECE/MCE), reliability diagram, per-slice metrics, top features
- run_demo.bat — one-command demo runner for Windows (installs from requirements.txt; demonstrates optional group-aware and calibration flows)

## Quick Start (Windows)

1) Run the end-to-end demo
- Double-click run_demo.bat or run in a terminal from this folder:

  run_demo.bat

This will:
- Upgrade pip and install dependencies from requirements.txt
- Compute inter-annotator agreement on demo data
- Insert adjudication stubs into the log (if new)
- Finalize dataset with a majority fallback and export stratified splits
- Optionally, if annotation/sample_data/groups.csv exists, regenerate splits with group-aware splitting
- Train the baseline
- Optionally, train a calibrated variant if group column exists in splits
- Run a single-text prediction and a batch evaluation
- Generate error analysis reports for both baseline and calibrated models

Outputs to expect:
- annotation/sample_data/final_dataset.csv and final_{train,val,test}.csv (with group column if provided)
- models/baseline_chargram/{vectorizer.pkl, model.pkl, metrics.json, confusion_matrix.csv}
- models/baseline_chargram_cal/{...} (if calibrated training ran)
- reports/baseline_chargram/{misclassified.csv, error_analysis_*, reliability_diagram.png?, calibration_bins.csv?}
- reports/baseline_chargram_cal/{...} (if calibrated analysis ran)

2) Use your own data (Label Studio or CSV)
- Export from Label Studio (JSON) or prepare CSV files with header: id,text,annotator,label
- Compute IAA and generate a disagreement queue:

  python "annotation\adjudicate.py" iaa --input "path\to\export_or_csv"
  python "annotation\adjudicate.py" queue --input "path\to\export_or_csv" --log "annotation\adjudication_log.md" --limit 100

- Optional: provide a groups mapping (id -> group) to enable group-aware experiments

  # groups.csv should have columns: id,group (or a custom column via --group_column)
  # Example row: 12345,user_abc

- Finalize a single-label dataset (apply adjudication decisions, choose fallback):

  python "annotation\finalize_dataset.py" combine --input "path\to\export_or_csv" --log "annotation\adjudication_log.md" --output "data\final_dataset.csv" --strategy majority --export-splits

  # With group propagation and group-aware splits
  python "annotation\finalize_dataset.py" combine --input "path\to\export_or_csv" --log "annotation\adjudication_log.md" --output "data\final_dataset.csv" --strategy majority --groups_csv "data\groups.csv" --group_column group --export-splits --group_aware_splits

- Train the baseline:

  python "modeling\train_baseline.py" --input "data\final_dataset.csv" --use_splits --output_dir "models\baseline_chargram"

  # Train with group-aware setting and probability calibration
  python "modeling\train_baseline.py" --input "data\final_dataset.csv" --use_splits --output_dir "models\baseline_chargram_cal" --group_column group --calibrate isotonic

- Inference/evaluation:

  python "modeling\predict.py" --model_dir "models\baseline_chargram" --text "អរគុណច្រើន សេវាកម្មល្អបំផុត 🙄"
  python "modeling\predict.py" --model_dir "models\baseline_chargram_cal" --input_csv "data\final_test.csv" --output_csv "data\pred_test_cal.csv"

- Error analysis:

  python "modeling\error_analysis.py" --model_dir "models\baseline_chargram" --input_csv "data\final_test.csv" --output_dir "reports\baseline_chargram"
  python "modeling\error_analysis.py" --model_dir "models\baseline_chargram_cal" --input_csv "data\final_test.csv" --output_dir "reports\baseline_chargram_cal" --slice_column group --reliability_bins 15

- Curate from model errors back into adjudication:

  python "annotation\curate_from_errors.py" --errors "reports\baseline_chargram\misclassified.csv" --output_csv "annotation\review_queue.csv" --strategy uncertain --limit 50 --insert_log "annotation\adjudication_log.md"

## Dependencies

- Python 3.8+
- Install from requirements.txt for reproducibility:

  pip install -r requirements.txt

- Optional: for reliability plots

  pip install matplotlib

- The demo runner (run_demo.bat) installs dependencies automatically from requirements.txt.

## Notes and Tips

- Labels are normalized to POS/NEG/NEU. The pipeline is robust to common variants (POSITIVE/NEGATIVE/NEUTRAL, +/−/0).
- Character n-grams work well for Khmer without tokenization, but you can adjust n-gram range and class weights in modeling/train_baseline.py.
- For group-aware experiments, propagate group via finalize_dataset.py (--groups_csv/--group_column) and use --group_column in training.
- Calibration improves probability quality; choose --calibrate platt (sigmoid) or --calibrate isotonic (more flexible, needs more data).
- Error analysis now includes ECE/MCE and optional reliability diagrams; add --slice_column (e.g., group) to inspect per-slice performance.
- Keep adjudication_log.md updated with rationales and action items to continuously improve guidelines and model performance.
- For large datasets, consider dataset versioning (e.g., DVC/Git LFS) and a virtual environment for dependencies.
