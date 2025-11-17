# Annotation Toolkit Usage

This folder contains scripts and guidance for annotation, adjudication, dataset finalization, and curation.

## Files
- adjudicate.py — build disagreement queues and compute IAA (κ)
- finalize_dataset.py — resolve labels, optionally propagate groups, and export splits (stratified or group-aware)
- curate_from_errors.py — turn model misclassifications into a review queue and log stubs
- adjudication_log.md — record adjudication decisions and action items
- guidelines.md — annotation rules and examples
- label_config.json — Label Studio 3-class setup
- sample_data/
  - demo_annotations.csv — demo with two annotators
  - groups.csv — sample mapping id -> group (e.g., user/thread)

## Group Metadata Guidance
Group metadata (e.g., user_id, thread_id) helps prevent leakage between training and evaluation when multiple items belong to the same entity. This pipeline supports group metadata as follows:

1) Provide a mapping CSV with columns: id,group (or id,<custom_name>)
   - Example row: 1001,user_a

2) Finalize dataset with group propagation (and optional group-aware splits):

   python annotation/finalize_dataset.py combine \
     --input annotation/sample_data/demo_annotations.csv \
     --log annotation/adjudication_log.md \
     --output annotation/sample_data/final_dataset.csv \
     --strategy majority \
     --groups_csv annotation/sample_data/groups.csv \
     --group_column group \
     --export-splits --group_aware_splits

3) Train with group-aware splits and calibration (optional):

   python modeling/train_baseline.py \
     --input annotation/sample_data/final_dataset.csv \
     --use_splits \
     --output_dir models/baseline_chargram_cal \
     --group_column group \
     --calibrate platt

4) Analyze errors with per-slice metrics and reliability diagrams:

   python modeling/error_analysis.py \
     --model_dir models/baseline_chargram_cal \
     --input_csv annotation/sample_data/final_test.csv \
     --output_dir reports/baseline_chargram_cal \
     --slice_column group \
     --reliability_bins 15

Notes:
- If you do not provide groups.csv, the pipeline will default to stratified splits and standard training.
- The final_dataset.csv and split CSVs will contain a group column when provided.
