#!/usr/bin/env bash
set -euo pipefail

# Start the FastAPI server and open the enhanced UI.
# Intended for Bash (Git Bash ok) from project root.

cd "$(dirname "${BASH_SOURCE[0]}")"

HOST="127.0.0.1"
PORT="8000"

echo "[1/3] Ensure dependencies for API/UI"
python -m pip install --upgrade pip setuptools wheel || echo "Warn: pip upgrade failed; continuing"
pip install -r requirements.txt || true
pip install fastapi uvicorn[standard] || true

# Prefer runs/* models if present
DEFAULT_MODEL=""
if [[ -d runs/xlmr_bilingual ]]; then DEFAULT_MODEL="runs/xlmr_bilingual"; fi
if [[ -d runs/baseline_chargram ]]; then DEFAULT_MODEL="${DEFAULT_MODEL:-runs/baseline_chargram}"; fi
if [[ -z "${DEFAULT_MODEL}" && -d models ]]; then DEFAULT_MODEL="models"; fi

echo
echo "[2/3] Launch API server (uvicorn)"
# Start API in background
uvicorn api.server:app --host "${HOST}" --port "${PORT}" --reload &
API_PID=$!
# Give server time to start
sleep 3 || true

# If a default model exists, try to pre-load via curl
if [[ -n "${DEFAULT_MODEL}" ]]; then
  echo "Pre-loading model: ${DEFAULT_MODEL}"
  curl -fsS -X POST "http://${HOST}:${PORT}/v1/models/select" -H 'Content-Type: application/json' -d "{\"model_dir\": \"${DEFAULT_MODEL}\"}" >/dev/null || true
fi

echo "[3/3] Open UI in default browser"
UI_URL="http://${HOST}:${PORT}/"
# Try to serve the static HTML via a file:// URL; but we can also open directly
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${UI_URL}" || true
elif command -v start >/dev/null 2>&1; then
  start "" "${UI_URL}" || true
else
  echo "Open your browser: ${UI_URL}"
fi

echo "API PID: ${API_PID}. Press Ctrl+C to stop (if foreground)."
