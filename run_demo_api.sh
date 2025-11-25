#!/usr/bin/env bash
set -euo pipefail

# One-click script: retrain demo model and start Khmer Sentiment API + UI (for Bash / Git Bash).

# Ensure we are in the script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR="annotation/sample_data"
MODEL_DIR="models"
REPORT_DIR="reports/baseline_chargram"
TOKENIZER_DIR="models/khmer_tokenizer_demo"
TRANSFORMER_DIR="models/transformer_khmer_demo"
TAU="0.5"
HOST="127.0.0.1"
PORT="8000"

echo "[1/6] Ensuring Python dependencies are installed..."
python -m pip install --upgrade pip setuptools wheel || echo "Warning: pip upgrade may have failed; continuing..."
pip install -r requirements.txt
pip install fastapi uvicorn[standard]

echo
echo "[2/6] Preparing dataset and splits (align with run_demo.sh)..."
# If a prebuilt final_dataset.csv exists, just generate splits; else run the adjudication combine path
if [[ -f "${DATA_DIR}/final_dataset.csv" ]]; then
  echo "Found prebuilt final_dataset.csv; generating stratified splits via split subcommand..."
  python annotation/finalize_dataset.py split \
    --input "${DATA_DIR}/final_dataset.csv" \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
else
  echo "Finalizing dataset with majority fallback and exporting stratified splits..."
  python annotation/adjudicate.py iaa --input "${DATA_DIR}/demo_annotations.csv" || true
  python annotation/adjudicate.py queue \
    --input "${DATA_DIR}/demo_annotations.csv" \
    --log annotation/adjudication_log.md \
    --limit 20 || true
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

echo
echo "[3/6] Training baseline model (char n-gram TF-IDF + Logistic Regression)..."
python -m modeling.train_baseline \
  --input "${DATA_DIR}/final_dataset.csv" \
  --use_splits \
  --output_dir "${MODEL_DIR}"

# Verify baseline artifacts exist
[[ -f "${MODEL_DIR}/vectorizer.pkl" ]] || { echo "ERROR: Missing ${MODEL_DIR}/vectorizer.pkl"; exit 1; }
[[ -f "${MODEL_DIR}/model.pkl" ]] || { echo "ERROR: Missing ${MODEL_DIR}/model.pkl"; exit 1; }

# Optional: produce quick error analysis reports similarly to run_demo.sh
if [[ -f "${DATA_DIR}/final_test.csv" ]]; then
  mkdir -p "${REPORT_DIR}" || true
  python -m modeling.error_analysis \
    --model_dir "${MODEL_DIR}" \
    --input_csv "${DATA_DIR}/final_test.csv" \
    --output_dir "${REPORT_DIR}" || true
fi

echo
echo "[4/6] Training Khmer tokenizer on training split..."
python -m tools.train_tokenizer \
  --input_csv "${DATA_DIR}/final_train.csv" \
  --text_column text \
  --output_dir "${TOKENIZER_DIR}" || echo "Warning: tokenizer training failed; continuing to API."

# echo "[5/6] Training transformer model (Khmer sentiment)..."
# python -m modeling.train_transformer \
#   --input "${DATA_DIR}/final_dataset.csv" \
#   --use_splits \
#   --output_dir "${TRANSFORMER_DIR}" \
#   --model_name "xlm-roberta-base" \
#   --epochs 1 \
#   --batch_size 4 \
#   --grad_accum 1 \
#   --lr 5e-5 \
#   --tracking none || echo "Warning: transformer training failed; continuing to API."

echo
echo "[6/6] Starting Khmer Sentiment Demo API..."
echo "       Base models directory: ${MODEL_DIR} (expects ${MODEL_DIR}/baseline_chargram and ${MODEL_DIR}/transformer_khmer_demo)"
echo "       tau: ${TAU}"
echo "       URL: http://${HOST}:${PORT}/"

# Start API server in background
python demo_api.py \
  --model_dir "${MODEL_DIR}" \
  --tau "${TAU}" \
  --host "${HOST}" \
  --port "${PORT}" &
API_PID=$!

# Give the server a few seconds to start
sleep 4

echo "[6/6] Opening browser to the demo UI (if xdg-open/start available)..."
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://${HOST}:${PORT}/" || true
elif command -v start >/dev/null 2>&1; then
  start "" "http://${HOST}:${PORT}/" || true
else
  echo "Please open your browser and go to: http://${HOST}:${PORT}/"
fi

echo
echo "API is running with PID ${API_PID}. Press Ctrl+C to stop it if running in foreground terminal."