# System Diagrams and Architecture Overview

This document presents the upgraded end-to-end system for Khmer–English sentiment analysis, including high-level pipeline, components, and data/experiment flows. Use these diagrams in your thesis write-up to clarify methodology and design decisions.

---

## 1) High-level Pipeline

```
[Data Sources]
   │
   ▼
[Collection + Governance]
   ├─ Licensing/ToS compliance
   ├─ PII filtering & ethics review
   └─ Dataset card metadata
   │
   ▼
[Annotation]
   ├─ Guidelines & label schema
   ├─ Double-annotation ≥20%
   └─ Adjudication & IAA (κ/α)
   │
   ▼
[Ingestion & Validation]
   ├─ Schema checks & label validation
   ├─ Group-aware split creation (train/val/test)
   ├─ Leakage/near-duplicate checks across splits
   └─ Split checksums & versioning
   │
   ▼
[Normalization]
   ├─ NFC, whitespace, punctuation
   ├─ Zero-width & diacritics reorder
   ├─ Elongation & emoji policy
   └─ Khmer digits + Latin handling
   │
   ▼
[Tokenizer]
   ├─ HF default (e.g., XLM-R)
   └─ Custom (Unigram/BPE; 8k/16k/32k)
   │
   ▼
[Modeling]
   ├─ Baseline: Char n-gram + LR
   ├─ Transformers: Base/Small/Large
   └─ Dual-head language-routed classifier (optional)
   │
   ▼
[Training]
   ├─ Seeds & deterministic settings
   ├─ Class weighting / focal loss (optional)
   └─ MLflow/W&B tracking
   │
   ▼
[Calibration & Conformal]
   ├─ Temperature/Isotonic calibration
   └─ Split-conformal prediction (α ∈ {0.05, 0.1, 0.2})
   │
   ▼
[Evaluation]
   ├─ Overall + per-language metrics (macro-F1, acc)
   ├─ Calibration (ECE, Brier, reliability diagrams)
   ├─ Robustness (stress presets & severities)
   └─ Error analysis (taxonomy + samples)
   │
   ▼
[Reporting]
   ├─ Tables & plots
   ├─ Statistical tests (bootstrap CIs, McNemar)
   └─ Artifacts + dataset card
```

---

## 2) Component Interaction Diagram

```
+--------------------+     +-------------------+     +-------------------+
| Annotation Artifacts| -->| Ingestion/Validate| --> | Group-aware Splits|
+--------------------+     +-------------------+     +-------------------+
                                      │                      │
                                      ▼                      ▼
                             +-------------------+   +---------------------+
                             | Normalization     |   | Tokenizer           |
                             | (configurable)    |   | (HF or custom)      |
                             +-------------------+   +---------------------+
                                      │                      │
                                      └──────────┬───────────┘
                                                 ▼
                                   +-------------------------------+
                                   | Modeling                      |
                                   |  - Baseline (char n-gram + LR)|
                                   |  - Transformers (base/small)  |
                                   |  - Dual-head (optional)       |
                                   +-------------------------------+
                                                 │
                                                 ▼
                         +----------------------------------------------+
                         | Calibration & Conformal                      |
                         | - Temperature / Isotonic                    |
                         | - Split conformal sets (coverage analysis)  |
                         +----------------------------------------------+
                                                 │
                                                 ▼
                              +---------------------------------+
                              | Evaluation & Reporting          |
                              | - Metrics (overall/per-lang)    |
                              | - Reliability diagrams, ECE     |
                              | - Robustness & error analysis   |
                              | - Bootstrap CIs & McNemar       |
                              +---------------------------------+
```

---

## 3) Data Flow & Tracking

```
CSV/Parquet ──> ingest_dataset.py ──> final_dataset.csv + final_{train,val,test}.csv
                                 └─> reports/ingest_validation.json
                                 └─> reports/dataset_checksums.json

train_* ──┐    ┌─> train_baseline.py / train_transformer.py
val_*   ──┼──> │    └─> runs/<run_id>/{metrics.json, config, norm.json, tok report}
test_*  ──┘    └─> calibration, conformal, stress_eval

mlruns/ (MLflow) <= tracker logs: metrics, artifacts, plots
```

---

## 4) Justification of Design Choices

- Group-aware splits and leakage checks: prevent over-optimistic results due to near-duplicate leakage.
- Khmer-aware normalization and tokenizer study: addresses script-specific challenges (diacritics, ZW chars, digits, code-switch, emoji) for originality and real-world robustness.
- Dual-head routing: captures language-specific classification boundaries without duplicating encoders.
- Calibration + conformal: raises reliability and decision-theoretic interpretability.
- Robustness testing: quantifies brittleness and validates mitigation from normalization/tokenizer choices.
- MLflow/W&B tracking: ensures reproducibility and thesis-grade auditability.
