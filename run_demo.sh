#!/usr/bin/env bash
set -euo pipefail

# Ensure we are in the script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR="annotation/sample_data"
MODEL_BASE="models/baseline_chargram"
REPORT_BASE="reports/baseline_chargram"
TOKENIZER_DIR="models/khmer_tokenizer_demo"
TRANSFORMER_DIR="models/transformer_khmer_demo"

echo "[1/10] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "[2/10] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[3/10] Computing inter-annotator agreement on demo data..."
python annotation/adjudicate.py iaa --input "${DATA_DIR}/demo_annotations.csv"

echo "[4/10] Inserting adjudication stubs into the log (if any new)..."
python annotation/adjudicate.py queue \
  --input "${DATA_DIR}/demo_annotations.csv" \
  --log annotation/adjudication_log.md \
  --limit 20

echo "[5/10] Finalizing dataset with majority fallback and exporting stratified splits..."
# If a prebuilt final_dataset.csv exists, reuse it; else finalize from demo_annotations
if [[ -f "${DATA_DIR}/final_dataset.csv" ]]; then
  echo "[5/10] Found prebuilt final_dataset.csv; generating stratified splits via split subcommand..."
  python annotation/finalize_dataset.py split \
    --input "${DATA_DIR}/final_dataset.csv" \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
else
  echo "[5/10] Finalizing dataset with majority fallback and exporting stratified splits..."
  python annotation/finalize_dataset.py combine \
    --input "${DATA_DIR}/demo_annotations.csv" \
    --log annotation/adjudication_log.md \
    --output "${DATA_DIR}/final_dataset.csv" \
    --strategy majority \
    --export-splits
fi

# Sanity check that splits exist
[[ -f "${DATA_DIR}/final_train.csv" ]] || { echo "ERROR: Missing ${DATA_DIR}/final_train.csv"; exit 1; }
[[ -f "${DATA_DIR}/final_val.csv" ]] || { echo "ERROR: Missing ${DATA_DIR}/final_val.csv"; exit 1; }
[[ -f "${DATA_DIR}/final_test.csv" ]] || { echo "ERROR: Missing ${DATA_DIR}/final_test.csv"; exit 1; }

echo "[6/12] Training baseline model (char n-gram TF-IDF + Logistic Regression)..."
python -m modeling.train_baseline \
  --input "${DATA_DIR}/final_dataset.csv" \
  --use_splits \
  --output_dir "${MODEL_BASE}"

# Verify baseline artifacts exist
[[ -f "${MODEL_BASE}/vectorizer.pkl" ]] || { echo "ERROR: Missing ${MODEL_BASE}/vectorizer.pkl"; exit 1; }
[[ -f "${MODEL_BASE}/model.pkl" ]] || { echo "ERROR: Missing ${MODEL_BASE}/model.pkl"; exit 1; }

echo "[7/12] Single-text prediction (sanity check)..."
python -m modeling.predict \
  --model_dir "${MODEL_BASE}" \
  --text "អរគុណច្រើន សេវាកម្មល្��បំផុត 🙄"

echo "[8/12] Batch prediction and evaluation on test split..."
python -m modeling.predict \
  --model_dir "${MODEL_BASE}" \
  --input_csv "${DATA_DIR}/final_test.csv" \
  --output_csv "${DATA_DIR}/pred_test.csv"

echo "[9/12] Error analysis and reports (baseline)..."
python -m modeling.error_analysis \
  --model_dir "${MODEL_BASE}" \
  --input_csv "${DATA_DIR}/final_test.csv" \
  --output_dir "${REPORT_BASE}"

echo "[10/12] Training Khmer tokenizer on training split..."
python -m tools.train_tokenizer \
  --input_csv "${DATA_DIR}/final_train.csv" \
  --text_column text \
  --output_dir "${TOKENIZER_DIR}" || echo "Warning: tokenizer training failed; continuing demo."

echo "[11/12] Training transformer model (Khmer sentiment)..."
python -m modeling.train_transformer \
  --input "${DATA_DIR}/final_dataset.csv" \
  --use_splits \
  --output_dir "${TRANSFORMER_DIR}" \
  --model_name "xlm-roberta-base" \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 5e-5 \
  --tracking none || echo "Warning: transformer training failed; continuing demo."

echo "[12/12] Completed all demo steps (baseline + tokenizer + transformer)."
echo
echo "Demo completed successfully." 
