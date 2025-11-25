#!/usr/bin/env bash
set -Eeuo pipefail

# ui_api.sh — Start FastAPI server and open UI (robust cross-platform)
# - Works in Bash (Git Bash on Windows, Linux, macOS)
# - Activates .venv if present
# - Installs dependencies (can be disabled)
# - Health-check wait loop
# - Opens Swagger UI (/docs) with multiple OS fallbacks
#
# Usage examples:
#   bash ui_api.sh
#   bash ui_api.sh --host 0.0.0.0 --port 8080
#   bash ui_api.sh --no-install --no-browser
#   bash ui_api.sh --timeout 60 --open-path /docs

cd "$(dirname "${BASH_SOURCE[0]}")"

# Defaults
HOST="127.0.0.1"
PORT="8000"
TIMEOUT="30"
OPEN_PATH="/docs"
NO_INSTALL="0"
NO_BROWSER="0"
MODEL_DIR=""

usage() {
  cat <<USAGE
Usage: bash ui_api.sh [options]
Options:
  --host HOST         Host to bind (default: ${HOST})
  --port PORT         Port to bind (default: ${PORT})
  --timeout SECS      Health check timeout seconds (default: ${TIMEOUT})
  --open-path PATH    Path to open in browser (default: ${OPEN_PATH})
  --model PATH        Model directory to preload (default: auto-detect)
  --no-install        Skip dependency installation
  --no-browser        Do not open a browser
  -h, --help          Show this help
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --open-path) OPEN_PATH="$2"; shift 2;;
    --model) MODEL_DIR="$2"; shift 2;;
    --no-install) NO_INSTALL="1"; shift;;
    --no-browser) NO_BROWSER="1"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

log() { echo "[ui_api] $*"; }

# Activate local virtual environment if present
if [[ -d ".venv" ]]; then
  if [[ -f ".venv/Scripts/activate" ]]; then
    # Windows venv under Git Bash
    # shellcheck disable=SC1091
    source ".venv/Scripts/activate"
  elif [[ -f ".venv/bin/activate" ]]; then
    # Unix venv
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
  fi
fi

# Helper: check if API responds
api_is_up() {
  local url="http://${HOST}:${PORT}/v1/health"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "$url" >/dev/null 2>&1
  else
    # Fallback to Python if curl missing
    python - "$url" >/dev/null 2>&1 <<'PY'
import sys, json, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=1) as r:
        print(r.status)
except Exception:
    sys.exit(1)
PY
  fi
}

# Ensure dependencies
log "[1/3] Ensure dependencies for API/UI"
if [[ "$NO_INSTALL" != "1" ]]; then
  python -m pip install --upgrade pip setuptools wheel || log "Warn: pip upgrade failed; continuing"
  # Base project deps
  python -m pip install -r requirements.txt || log "Warn: requirements install failed; continuing"
  # Ensure fastapi + uvicorn are available
  if ! python - <<'PY' >/dev/null 2>&1
import importlib; importlib.import_module('fastapi'); importlib.import_module('uvicorn')
PY
  then
    python -m pip install fastapi 'uvicorn[standard]' || log "Warn: api deps install failed; continuing"
  fi
else
  log "Skipping dependency installation (--no-install)"
fi

# Auto-detect model if not provided
if [[ -z "$MODEL_DIR" ]]; then
  if [[ -d runs/xlmr_bilingual ]]; then MODEL_DIR="runs/xlmr_bilingual"; fi
  if [[ -z "$MODEL_DIR" && -d runs/baseline_chargram ]]; then MODEL_DIR="runs/baseline_chargram"; fi
  if [[ -z "$MODEL_DIR" && -d models ]]; then MODEL_DIR="models"; fi
fi

# Launch API
log "\n[2/3] Launch API server (uvicorn)"
UI_URL="http://${HOST}:${PORT}${OPEN_PATH}"

# Start API in background
set +e
python -m uvicorn api.server:app --host "${HOST}" --port "${PORT}" --reload &
API_PID=$!
set -e

# Wait for server health endpoint to respond
log "Waiting for health: http://${HOST}:${PORT}/v1/health (timeout=${TIMEOUT}s)"
START_TS=$(date +%s || echo 0)
while ! api_is_up; do
  sleep 1 || true
  NOW_TS=$(date +%s || echo 999999)
  if [[ $((NOW_TS - START_TS)) -ge ${TIMEOUT} ]]; then
    log "ERROR: API failed to become healthy within ${TIMEOUT}s"
    kill ${API_PID} >/dev/null 2>&1 || true
    exit 1
  fi
done
log "API is healthy."

# Pre-load model if detected
if [[ -n "${MODEL_DIR}" ]]; then
  log "Pre-loading model: ${MODEL_DIR}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -X POST "http://${HOST}:${PORT}/v1/models/select" \
      -H 'Content-Type: application/json' \
      -d "{\"model_dir\": \"${MODEL_DIR}\"}" \
      >/dev/null 2>&1 || log "Warn: preload failed"
  else
    python - <<PY "http://${HOST}:${PORT}/v1/models/select" "${MODEL_DIR}" >/dev/null 2>&1 || echo ""
import sys, json, urllib.request
req = urllib.request.Request(sys.argv[1], data=json.dumps({"model_dir": sys.argv[2]}).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST')
try:
    urllib.request.urlopen(req, timeout=2)
except Exception:
    pass
PY
  fi
fi

# Open UI in default browser (robust cross-platform)
open_url() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || return 1
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || return 1
    return 0
  fi
  if command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "$url" >/dev/null 2>&1 || return 1
    return 0
  fi
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -NonInteractive -Command "Start-Process '$url'" >/dev/null 2>&1 || return 1
    return 0
  fi
  if command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "$url" >/dev/null 2>&1 || return 1
    return 0
  fi
  return 1
}

log "\n[3/3] Open UI in default browser -> ${UI_URL}"
if [[ "$NO_BROWSER" != "1" ]]; then
  if ! open_url "${UI_URL}"; then
    log "Could not auto-open browser. Open manually: ${UI_URL}"
  fi
else
  log "Skipping browser open (--no-browser). URL: ${UI_URL}"
fi

log "API PID: ${API_PID}. Press Ctrl+C to stop (if foreground)."
# Script exits but API keeps running in background.
exit 0
