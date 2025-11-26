# Research Guide: Khmer + English Sentiment Analysis

This guide summarizes the research protocol aligned with the current repository and documents how to reproduce thesis-level experiments. It maps research questions to concrete scripts/configs in modeling/ and tools/, and to YAML grids in experiments/.

---

## 1. Research Questions and Hypotheses

### RQ1: Khmer-specific Normalization
- Question: How do Khmer-specific normalization operations affect sentiment classification on Khmer social text?
- Hypotheses:
  - H1a: Using the full preset (--normalize_all) improves macro-F1 and accuracy vs. no normalization.
  - H1b: Mapping Khmer digits to ASCII (--norm_khmer_digits map) outperforms keeping Khmer digits.
  - H1c: Khmer punctuation normalization (--norm_khmer_punct) reduces segmentation-related errors.
- Code & Config: modeling/text_normalization.py, experiments/normalization_ablation_baseline.yml.
- Protocol: Fix dataset/splits/model; vary normalization flags; compare metrics and error categories.

### RQ2: Emojis and Code-switch Handling
- Question: What is the effect of emoji handling and Latin code-switch strategies on Khmer sentiment?
- Hypotheses:
  - H2a: Mapping emojis to tokens (--norm_emoji map) outperforms keeping/removing.
  - H2b: Tagging Latin spans (--norm_latin_action tag) is more robust to code-switch than none.
- Code & Config: modeling/text_normalization.py; experiments/tokenizer_transformer_grid.yml.
- Protocol: Sweep norm_emoji and norm_latin_action; evaluate overall, per-group/code-switch slices.

### RQ3: Baseline vs. Transformer Trade-offs
- Question: How do n-gram baselines compare to multilingual transformers in accuracy, calibration, and cost?
- Hypotheses:
  - H3a: XLM-R base > char n-gram baseline in macro-F1 on identical splits.
  - H3b: Calibrated baselines can have lower ECE than transformers.
  - H3c: Baselines are more efficient by orders of magnitude.
- Code & Config: modeling/train_baseline.py, modeling/train_transformer.py, experiments/*.
- Protocol: Fix dataset/seed/splits; compare best configs; report accuracy, macro-F1, calibration, and efficiency.

### RQ4: Tokenizer Design for Khmer
- Question: How do tokenizer type, vocab size, and normalization affect performance?
- Hypotheses:
  - H4a: Khmer-specific tokenizers trained with proper normalization improve coverage and shorten sequences.
  - H4b: Using a Khmer-specific tokenizer with XLM-R improves macro-F1 over default tokenizer.
  - H4c: Medium vocab sizes (e.g., 16k) balance performance and sequence length.
- Code & Config: tools/train_tokenizer.py; modeling/train_transformer.py; experiments/tokenizer_transformer_grid.yml.
- Protocol: Train several tokenizers; integrate via --tokenizer_path; evaluate and compare.

### RQ5: Calibration and Uncertainty
- Question: How do calibration methods and conformal prediction affect reliability?
- Hypotheses:
  - H5a: Post-hoc calibration reduces ECE.
  - H5b: Temperature scaling offers a strong simple baseline.
  - H5c: Conformal prediction with calibrated scores reaches near-nominal coverage.
- Code & Config: modeling/calibration_utils.py; modeling/conformal_predict.py; experiments/calibration_grid_baseline.yml.
- Protocol: Select settings from calibration grid; compute reliability diagrams and conformal coverage.

### RQ6: Bilingual Training and Cross-lingual Performance
- Question: What are the gains of bilingual training and language-routed heads for Khmer and English?
- Hypotheses:
  - H6a: Joint bilingual training improves Khmer test performance over Khmer-only under limited data.
  - H6b: Dual-head routing improves per-language performance vs. a single head.
  - H6c: Stratifying by (label,lang) stabilizes performance across languages.
- Code & Config: modeling/train_transformer.py (--lang_column, --stratify_by_lang, --dual_head).
- Protocol: Compare Khmer-only vs. bilingual joint training; single-head vs. dual-head; report per-language metrics and calibration.

---

## 2. Reproducibility and Tracking

- Seeds: Set --seed; scripts set seeds across Python/NumPy/Torch and deterministic flags where applicable.
- Config snapshots: <output_dir>/experiment_config.json; normalization.json saved with each run.
- Tracking: Use MLflow or W&B (--tracking, --mlflow_experiment).
- Hardware: hardware.json (OS/CPU/RAM/GPU/CUDA) recommended for each run.

---

## 3. Datasets and Validation

- Use tools/validate_dataset.py with --lang_column and --allowed_langs for bilingual checks and code-switch summary.
- Ingest with tools/ingest_dataset.py to write final_dataset.csv and optional final_train/val/test.csv with group-aware splits.
- Compute IAA (tools/compute_iaa.py) and include overall and per-language kappa in the thesis.

Data schema: id,text,label[,lang,group,split,source]. Ensure splits avoid group leakage.

---

## 4. Training Protocols

### 4.1 Baseline (char n-gram)
- Use --use_splits, --normalize_all by default.
- Consider calibration (--calibrate temperature) for improved ECE.

### 4.2 Transformer (bilingual)
- Use --lang_column lang; optionally --langs km en.
- Enable --stratify_by_lang for random splits; prefer --use_splits for fixed comparisons.
- Dual-head via --dual_head for language-routed heads; compare with single-head.
- Optionally enable --class_weighting balanced, --focal_loss for class imbalance.

---

## 5. Stress and Robustness Protocol

Use modeling/stress_eval.py with presets:
- diacritics, zero_width, emoji_burst, elongation, mixed_script, code_switch, latinized_khmer.
- Control intensity via --severity and application rate via --prob.

Compare overall/per-category metrics under perturbations; quantify robustness deltas by comparing to clean test results.

---

## 6. Statistical Testing and Reporting

### 6.1 Paired bootstrap CIs
- Use tools/paired_bootstrap_ci.py or tools/export_report.py --pred_pair a.csv:b.csv[:name].
- Report 95% CIs for macro-F1 and accuracy and whether the delta CI excludes 0.

### 6.2 Aggregated reporting
- Use tools/export_report.py --glob "models/*" --out_dir reports/final.
- Outputs overall.csv, per_language.csv, calibration.csv, plots/, and summary.md.
- Optionally add robustness deltas via --stress model_dir:stress_metrics.json.

---

## 7. Thesis–Code Mapping

- Data & annotation: annotation/*.md, tools/validate_dataset.py, tools/ingest_dataset.py, tools/compute_iaa.py.
- Preprocessing: modeling/text_normalization.py; normalization.json in output dirs.
- Baseline: modeling/train_baseline.py; calibration grids.
- Transformer: modeling/train_transformer.py (bilingual/dual-head options); tokenizer training tools.
- Stress & error: modeling/stress_eval.py, modeling/error_analysis.py.
- Reporting: tools/export_report.py, tools/paired_bootstrap_ci.py.

---

## 8. Checklists

### Data & Ethics
- [ ] Dataset card: size, label distribution, language distribution, sources, license, privacy notes.
- [ ] IAA: overall and per-language kappa; annotation protocol and adjudication.
- [ ] Leakage prevention: group-aware splits validated; document split policy.

### Experiments
- [ ] All runs tracked with MLflow/W&B; seeds fixed.
- [ ] Normalization/tokenizer artifacts saved and referenced.
- [ ] Ablations: normalization components, tokenizer types/sizes; bilingual dual-head vs. single-head.

### Reporting
- [ ] Tables: accuracy, macro-F1, calibration, efficiency; per-language breakdowns.
- [ ] Plots: reliability diagrams (saved per run), bar charts (export_report).
- [ ] Statistical tests: paired bootstrap deltas with 95% CIs.
- [ ] Robustness: stress deltas vs. clean test; code-switch analysis.
