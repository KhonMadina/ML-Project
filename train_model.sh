#!/usr/bin/env bash
set -euo pipefail

# Enhanced pipeline: validate dataset, train baseline and transformer (bilingual-aware),
# train tokenizer, produce quick reports.
# Intended for Bash (Git Bash on Windows is fine) from project root.

cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR="annotation/sample_data"
OUT_ROOT="runs"
BASELINE_DIR="${OUT_ROOT}/baseline_chargram"
TRANSFORMER_DIR="${OUT_ROOT}/xlmr_bilingual"
TOKENIZER_DIR="tokenizers/unigram_kh_16k"
ALLOWED_LANGS=(km en)
SEED=123

echo "[1/7] Ensure dependencies"
python -m pip install --upgrade pip setuptools wheel || echo "Warn: pip upgrade failed; continuing"
pip install -r requirements.txt

# Optional API deps
pip install fastapi uvicorn[standard] || true

echo "\n[2/7] Validate dataset (bilingual-aware)"
if [[ ! -f "${DATA_DIR}/final_dataset.csv" ]]; then
  echo "ERROR: ${DATA_DIR}/final_dataset.csv not found. Use tools/ingest_dataset.py or place your CSVs." >&2
  exit 1
fi
python tools/validate_dataset.py \
  --input "${DATA_DIR}/final_dataset.csv" \
  --lang_column lang --allowed_langs "${ALLOWED_LANGS[@]}" \
  --output_clean "${DATA_DIR}/final_dataset_clean.csv" || true

# Ensure splits
if [[ ! -f "${DATA_DIR}/final_train.csv" || ! -f "${DATA_DIR}/final_val.csv" || ! -f "${DATA_DIR}/final_test.csv" ]]; then
  echo "Generating splits from final_dataset.csv"
  python annotation/finalize_dataset.py split \
    --input "${DATA_DIR}/final_dataset.csv" \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
fi

[[ -f "${DATA_DIR}/final_train.csv" ]] || { echo "Missing ${DATA_DIR}/final_train.csv"; exit 1; }
[[ -f "${DATA_DIR}/final_val.csv" ]] || { echo "Missing ${DATA_DIR}/final_val.csv"; exit 1; }
[[ -f "${DATA_DIR}/final_test.csv" ]] || { echo "Missing ${DATA_DIR}/final_test.csv"; exit 1; }

mkdir -p "${OUT_ROOT}" || true

echo "\n[3/7] Train baseline (char n-gram)"
python -m modeling.train_baseline \
  --input "${DATA_DIR}/final_dataset.csv" \
  --use_splits \
  --output_dir "${BASELINE_DIR}" \
  --experiment_name baseline_chargram \
  --tracking none \
  --seed ${SEED} \
  --normalize_all || echo "Warn: baseline training failed"

[[ -f "${BASELINE_DIR}/model.pkl" ]] || echo "Warn: baseline model missing at ${BASELINE_DIR}"

echo "\n[4/7] Train tokenizer (unigram 16k)"
python -m tools.train_tokenizer \
  --input_csv "${DATA_DIR}/final_train.csv" \
  --text_column text \
  --output_dir "${TOKENIZER_DIR}" \
  --type unigram --vocab_size 16000 \
  --normalize_all || echo "Warn: tokenizer training failed"

echo "\n[5/7] Train transformer (bilingual options)"
python -m modeling.train_transformer \
  --input "${DATA_DIR}/final_dataset.csv" \
  --use_splits \
  --output_dir "${TRANSFORMER_DIR}" \
  --model_name xlm-roberta-base \
  --lang_column lang --langs "${ALLOWED_LANGS[@]}" --stratify_by_lang \
  --epochs 1 --batch_size 4 --grad_accum 1 --lr 5e-5 \
  --experiment_name transformer_xlmr_bilingual \
  --tracking none \
  --seed ${SEED} \
  --normalize_all || echo "Warn: transformer training failed"

echo "\n[6/7] Optional: quick stress evaluation on baseline (if stress CSV present)"
if [[ -f "${DATA_DIR}/final_test.csv" ]]; then
  python -m modeling.stress_eval \
    --model_dir "${BASELINE_DIR}" \
    --input_csv "${DATA_DIR}/final_test.csv" \
    --output_dir "${BASELINE_DIR}_stress" \
    --preset emoji_burst --severity 0.5 --prob 0.7 || true
fi

echo "\n[7/7] Done. Artifacts:"
echo "  Baseline:     ${BASELINE_DIR}"
echo "  Transformer:  ${TRANSFORMER_DIR}"
echo "  Tokenizer:    ${TOKENIZER_DIR}"
