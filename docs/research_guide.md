# Research Guide 2.0: Khmer–English Sentiment Analysis – Graduate-level Methodology and Execution Blueprint

This guide defines the upgraded methodology, datasets protocol, modeling choices, experiment design, statistical testing, robustness, and reporting standards for a Master’s-level thesis based on this repository. It supersedes earlier guidance and maps thesis-ready research questions to concrete scripts/configs and artifacts.

---

## A. Executive Summary of Upgrades and Justifications

Improvements introduced to raise originality, rigor, and academic suitability:
- Dataset governance and ethics
  - Dataset card requirements (size, label/lang distribution, sources, license, collection window, privacy, known biases).
  - Double-annotation ≥20% with Cohen’s κ targets and adjudication records.
  - Group-aware splitting policy to prevent leakage; validation utilities enforced.
  - Justification: Transparent data documentation and leakage control are core to research validity.
- Khmer-aware normalization and tokenizer study
  - Systematic ablations of normalization components and integration with tokenizer training.
  - Tokenizer variants (Unigram/BPE) and vocab sizes with coverage/sequence-length reporting.
  - Justification: Khmer script specifics (diacritics, zero-width, digits/punct, code-switch) strongly affect tokenization and model accuracy.
- Expanded model suite and bilingual routing
  - Baselines: character n-gram + calibrated logistic regression.
  - Transformers: xlm-roberta-base plus lighter backbones (e.g., Distil, MiniLM) for efficiency–accuracy trade-offs.
  - Language-routed dual-head classifier option; per-language metrics and calibration.
  - Justification: Broadens comparisons, quantifies efficiency, and studies bilingual effects on Khmer.
- Calibration and conformal prediction as first-class citizens
  - Temperature scaling and isotonic calibration with reliability diagrams and ECE/Brier.
  - Split conformal prediction with coverage analysis across target α.
  - Justification: Reliability is crucial for real applications; adds substantive methodological depth.
- Robustness and error analysis
  - Stress presets for Khmer phenomena (diacritics reorder, zero-width, emoji bursts, elongation, mixed script, code-switch, latinized Khmer) with severity/prob controls.
  - Structured error categorization (negation, sarcasm, emoji, code-switch, normalization-related tokenization errors).
  - Justification: Evaluates real-world brittleness and informs normalization/tokenizer design.
- Statistical rigor and reporting
  - Paired bootstrap CIs for macro-F1/accuracy; McNemar’s test; Holm–Bonferroni correction across RQs.
  - Standardized tables/plots, per-language/slice metrics, parameter counts, throughput, training time.
  - Justification: Ensures claims are statistically supported and reproducible.

---

## B. System Overview and Artifacts

Pipeline overview:

[Data Collection] → [Validation + IAA + Adjudication] → [Ingestion + Group-aware Splits] → [Normalization] → [Tokenizer (optional custom)] → [Models: Baseline | Transformers (single/dual-head)] → [Training] → [Calibration + Conformal] → [Evaluation: metrics, per-language, slices, robustness] → [Reporting + CIs + Plots] → [Tracking (MLflow/W&B) + Artifacts]

Key artifacts per run:
- normalization.json, experiment_config.json, hardware.json, metrics.json, confusion_matrix.csv
- reliability_diagram.png, calibration.json (ECE/Brier), conformal_coverage.json
- pred_{val,test}.csv with scores; error_samples.csv for misclassifications
- tokenizer_report.json (if custom tokenizer used)

---

## C. Research Questions and Hypotheses

Carry forward and expand prior RQs with precise hypotheses and statistical expectations.

- RQ1: Khmer-specific normalization
  - H1a: Full preset improves macro-F1 vs. none on fixed splits.
  - H1b: Digit and punctuation normalization reduce OOV and improve coverage → tokenization-related error reduction.
  - H1c: Emoji mapping improves robustness under emoji_burst stress.
- RQ2: Emojis and code-switch handling
  - H2a: Emoji mapping or removal improves calibration (ECE) vs keep.
  - H2b: Latin span tagging improves km+en slice macro-F1.
- RQ3: Baseline vs transformers
  - H3a: XLM-R-base bests char n-gram in macro-F1, but baseline achieves lower ECE after calibration.
  - H3b: Lightweight transformers (Distil/MiniLM) retain ≥90% macro-F1 at higher throughput.
- RQ4: Tokenizer design for Khmer
  - H4a: Khmer-specific Unigram 16k improves per-language macro-F1 vs default XLM-R tokenizer.
  - H4b: Vocab 16k balances performance and sequence length.
- RQ5: Calibration and conformal prediction
  - H5a: Temperature scaling reduces ECE without hurting macro-F1; isotonic excels in small-data regimes.
  - H5b: Conformal prediction reaches near-nominal coverage across α on held-out calibration splits.
- RQ6: Bilingual training and routing
  - H6a: Joint bilingual training improves Khmer macro-F1 vs Khmer-only under limited data.
  - H6b: Dual-head routing improves per-language performance vs single-head.

Each Hx must be tested via predefined grids with paired bootstrap CIs and McNemar tests where appropriate.

---

## D. Dataset Methodology and Ethics

Data schema: id, text, label, optional: lang, group, split, source.

- Sourcing and licensing
  - Document sources, collection window, license/ToS compliance, language distribution, and domain(s).
  - Provide dataset card (see Section L) in reports/dataset_card.md.
- Annotation protocol
  - Follow annotation/guidelines.md with clear decision rules.
  - Double-annotate ≥20%; compute Cohen’s κ overall and per-language; target κ ≥ 0.7.
  - Adjudicate disagreements; log in annotation/adjudication_log.md.
- Splits and leakage prevention
  - Use tools/ingest_dataset.py to finalize final_dataset.csv and optional fixed splits.
  - Use group-aware splits; validate with tools/validate_dataset.py; run tests/test_group_splits.py.
- Ethics and privacy
  - PII masking, consent where applicable, and an IRB/ethics statement in the dataset card.
  - Risk assessment and mitigation (e.g., sensitive topics, user-generated content).
- Reproducibility and governance
  - Hash the final CSV; version datasets; store schema checks and validation reports. Save hardware.json for runs.

---

## E. Text Normalization (Khmer-aware)

Default policy: enable --normalize_all for production; perform ablations for research.

Components to consider:
- Unicode NFC; whitespace and punctuation normalization (including Khmer punct)
- Khmer digits mapping; zero-width removal; diacritics reorder
- Elongation normalization; emoji policy {keep, remove, map}
- Latin handling {none, tag, strip} with threshold on ratio of Latin chars

Evaluation signals:
- Tokenizer coverage improvements; reduced sequence length; downstream macro-F1 and ECE gains
- Robustness deltas under stress presets (emoji_burst, mixed_script, code_switch)

---

## F. Tokenizer Design Study

Train tokenizers with tools/train_tokenizer.py using multiple configurations:
- Types: Unigram, BPE
- Vocab sizes: 8k, 16k, 32k
- With/without normalization integration

Report for each tokenizer:
- Coverage (OOV proxy), mean/median sequence length, 95th percentile length
- Impact on model metrics (macro-F1, ECE), throughput, and memory footprint

Integration: modeling/train_transformer.py with --tokenizer_path. Copy tokenizer_report.json to run dir for tracking.

---

## G. Model Architectures and Training

Baselines
- Character n-gram features + Logistic Regression (default) with calibration support
- Options: class weighting, resampling; report parameter counts and inference throughput

Transformers
- Backbones: xlm-roberta-base (reference), distiluse/mMiniLM-like multilingual small model (lightweight), and optionally a larger model for sensitivity
- Language routing: --dual_head option with shared encoder and language-specific heads
- Class imbalance: --class_weighting balanced, or focal loss (--focal_loss, --focal_gamma)
- Key hyperparameters: epochs, batch_size, lr, weight_decay, warmup_ratio, max_length, grad_accum, fp16 (when available)

Training protocol
- Fixed seeds and fixed splits for primary comparisons
- Save best checkpoint by macro-F1 on validation; early stopping optional with patience
- Log parameter counts; train_seconds and inference throughput

---

## H. Calibration and Conformal Prediction

Calibration
- Methods: temperature scaling (default), isotonic regression
- Metrics: ECE (multi-bin), Brier score; reliability diagrams per run saved to disk
- Protocol: fit calibration on validation split; evaluate on test; compare pre/post calibration

Conformal prediction
- Split-conformal using validation as calibration set
- Targets: analyze coverage for α ∈ {0.05, 0.1, 0.2}; report set sizes and coverage gaps
- Optional abstention analysis via confidence thresholds

---

## I. Robustness and Error Analysis

Stress presets (modeled in modeling/stress_eval.py):
- diacritics, zero_width, emoji_burst, elongation, mixed_script, code_switch, latinized_khmer
- Parameters: --severity, --prob; evaluate deltas vs clean test for accuracy, macro-F1, ECE

Error analysis
- Categorize common errors: negation scope, sarcasm, emoji sentiment mismatch, code-switch spans, normalization/tokenization artifacts
- Output error_samples.csv with predicted vs gold, confidence, and category tags; summarize distributions

---

## J. Experimental Design and Statistical Testing

Primary metrics
- Macro-F1 (primary), accuracy (secondary), calibration (ECE, Brier), efficiency (params, throughput, train_seconds)
- Per-language metrics (km, en) and per-slice (km+en code-switch)

Design principles
- Fix dataset and splits; fix seeds; vary one factor per comparison
- Minimum of 3 independent seeds for sensitivity analyses when compute allows

Statistical testing
- Paired bootstrap CIs (95%) for macro-F1 and accuracy between systems on the same test set
- McNemar’s test on paired predictions where appropriate
- Multiple comparisons: Holm–Bonferroni correction across tests within each RQ family
- Report effect sizes (absolute macro-F1 delta, relative ECE reduction)

Reporting expectations
- State whether CI excludes 0; include p-values for McNemar
- Include reliability plots and confusion matrices per model

---

## K. Experiment Grids and Mapping to Code

Use tools/run_experiment_grid.py with the following grids (to be maintained in experiments/):
- normalization_ablation_baseline.yml: toggle normalization components on baseline; fixed splits and seed
- calibration_grid_baseline.yml: {none, temperature, isotonic}, varying binning schemes for ECE plots
- tokenizer_transformer_grid.yml: backbones × tokenizer types × vocab sizes × Latin handling
- bilingual_vs_monolingual.yml: Khmer-only vs bilingual; single-head vs dual-head; class weighting/focal loss variants
- backbone_efficiency_grid.yml: small vs base vs large backbones; throughput and parameter counts
- robustness_sweep.yml: evaluate best models under all stress presets and severities

Each grid must specify:
- Input CSV, fixed splits flag, group-aware policy, seed, tracking config
- Output_dir prefix; MLflow experiment name; save normalization/tokenizer artifacts

---

## L. Dataset Card Template (reports/dataset_card.md)

Include at minimum:
- Title, version, date, maintainers/contact
- Summary and intended use
- Data sources and collection procedure; time window; domains
- Size, label distribution, per-language distribution; split sizes; group policy
- Licensing and ToS compliance; privacy considerations; PII handling
- Annotation protocol: annotator demographics (if available), double-annotation rate, κ overall and per-language, adjudication protocol and outcomes
- Preprocessing/normalization applied; code-switch summary
- Known limitations, potential biases, and risks; ethical considerations and IRB status (if applicable)
- How to cite; access policies

---

## M. Reproducibility and Tracking

- MLflow or W&B tracking for all primary experiments; keep local mlruns as default
- Save experiment_config.json, normalization.json, tokenizer_report.json (if any), reliability_diagram.png, calibration.json
- Save hardware.json with OS/CPU/RAM/GPU/CUDA; log Python/package versions
- Hash input CSV and record checksum and path in experiment_config.json

---

## N. Reporting Templates and Deliverables

Per RQ deliverables:
- Tables: macro-F1/accuracy, ECE/Brier, params, throughput; per-language breakdowns
- Plots: reliability diagrams, bar charts across models, robustness deltas
- Statistical appendices: paired bootstrap CI JSONs, McNemar results
- Error analysis summaries with samples

Thesis structure mapping:
- Related work (Khmer NLP, tokenization, normalization, bilingual routing, calibration/conformal)
- Data and ethics (dataset card)
- Methods (normalization, tokenizer, models, bilingual/dual-head, calibration/conformal)
- Experiments and results with statistics; robustness and error analyses
- Discussion, limitations, and future work

---

## O. Roadmap and Checklist

Immediate actions
- Finalize dataset and produce dataset_card.md with IAA and adjudication summary
- Freeze group-aware splits; validate and save checksums
- Run normalization ablations (baseline), tokenizer/backbone grids (transformers)
- Calibrate top models; generate reliability plots and calibration.json
- Run conformal coverage; save conformal_coverage.json
- Execute robustness sweeps for best models; export deltas
- Export unified report via tools/export_report.py with CIs and plots

Enhancements
- Add additional lightweight backbone option if not present
- Extend error analysis categorization and automate tagging
- Consider releasing (part of) the dataset where permissible

Checklist (extract)
- [ ] Dataset card present with license and ethics
- [ ] IAA computed and reported (overall and per-language)
- [ ] Group-aware splits validated and fixed
- [ ] All primary experiments tracked with MLflow/W&B
- [ ] Statistical tests run for all RQs with CIs and corrections
- [ ] Robustness results reported with deltas

---

## P. Limitations and Risk Management

- Limited compute: prioritize small backbones and thorough baselines; report efficiency transparently
- Dataset constraints: if unreleasable, maximize documentation and reproducibility
- Domain shift: include slice analyses and robustness to mixed scripts and code-switch

---

This document defines the standard to which the codebase, configurations, and final report should adhere. Subsequent commits will align code, configs, tests, and documentation to these requirements.