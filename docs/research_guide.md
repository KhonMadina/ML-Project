# Research Guide: Khmer Sentiment Analysis Pipeline

This guide outlines rigorous experiment design, ablations, and robustness evaluations for a research-grade study.

Contents
- Experiment Protocol
- Baselines and Ablations
- Calibration and Thresholding
- Group-aware Evaluation
- Error Taxonomy and Guideline Feedback
- Robustness/Stress Tests
- Active/Weak/Semi-supervised Extensions (roadmap)
- Reporting and Reproducibility

## Experiment Protocol
- Fix random seeds and record environment:
  - Training logs metrics.json include env and args.
  - Error analysis now records env/args.
- Use finalized datasets with clear versioning (commit hash, dated folder, or DVC).
- Always evaluate on a held-out test split never used for model/threshold selection.

## Baselines and Ablations
- Baselines:
  - Char n-gram TF-IDF + Logistic Regression (this repo)
  - Add transformer baselines (XLM-R, mBERT) for modern comparisons.
- Ablations for baseline:
  - n-gram ranges (2–5), min_df, class weights (balanced/json), resampling (none/over/under).
  - Group-aware vs stratified splits (effect on metrics).
  - Calibration methods (none/platt/isotonic) and dataset size sensitivity.
- Use tools/compare_experiments.py to consolidate results.

## Calibration and Thresholding
- Evaluate calibration quality with ECE/MCE and reliability diagrams.
- For applications requiring decision thresholds, compute precision/recall/PR curves and select thresholds on validation data.
- If deploying, prefer calibrated probabilities for stable thresholding.

## Group-aware Evaluation
- When multiple entries share a user/thread, enable leakage-safe splits:
  - In finalize_dataset.py: --groups_csv --group_column --group_aware_splits
  - In training: --group_column
- Report performance under both split regimes where relevant.

## Error Taxonomy and Guideline Feedback
- Cluster misclassifications by features or embeddings; inspect representative samples.
- Tag common phenomena (negation scope, sarcasm markers, emoji polarity, code-mixing) and update annotation guidelines.
- Track κ over time and link improvements to guideline changes and training data updates.

## Robustness/Stress Tests
Design a small suite targeting Khmer-specific challenges:
- Negation scope: contrasting phrases (e.g., មិនល្អទេ vs មិនអាក្រក់ទេ)
- Sarcasm: emojis like 🙄 with positive text
- Code-mixing: common English terms with Khmer context
- Repeated characters/Thai-style laughter (e.g., 555)

Create a CSV with columns id,text,label,category and a script to evaluate by category and report deltas vs standard test metrics.

## Active/Weak/Semi-supervised Extensions (Roadmap)
- Active learning: uncertainty sampling using margins from error_analysis to build adjudication queues.
- Weak supervision: labeling functions based on emojis, negation patterns, lexicons; combine with a label model.
- Semi-supervised: self-training/pseudo-labeling; iterate training with high-confidence unlabeled data.

## Reporting and Reproducibility
- Include:
  - Dataset description (size, balance, sources, group definition)
  - Split methodology (stratified vs group-aware)
  - Model configurations and hyperparameters
  - Calibration diagnostics
  - Error taxonomy and qualitative examples
- Release artifacts:
  - requirements.txt
  - exact command lines
  - models/*/metrics.json; reports/* contents
  - scripts/notebooks used to aggregate/plot results
