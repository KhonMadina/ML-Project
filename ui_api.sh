#!/usr/bin/env bash
set -Eeuo pipefail

# ui_api.sh — Start FastAPI server and open UI (robust cross-platform)
# - Works in Bash (Git Bash on Windows, Linux, macOS)
# - Activates .venv if present
# - Installs dependencies (can be disabled)
# - Health-check wait loop
# - Opens Swagger UI (/docs) or Demo UI (demo_ui.html) with multiple OS fallbacks
#
# Usage examples:
#   bash ui_api.sh
#   bash ui_api.sh --host 0.0.0.0 --port 8080
#   bash ui_api.sh --no-install --no-browser
#   bash ui_api.sh --timeout 60 --open-path /docs
#   bash ui_api.sh --demo               # opens demo_ui.html on a static server
#   bash ui_api.sh --demo --ui-port 5501 --ui-path demo_ui.html

cd "$(dirname "${BASH_SOURCE[0]}")"

# Defaults
HOST="127.0.0.1"
PORT="8000"
TIMEOUT="30"
OPEN_PATH="/docs"
NO_INSTALL="0"
NO_BROWSER="0"
MODEL_DIR=""
DEMO="0"             # if 1, also serve demo_ui.html
UI_PORT="5500"
UI_PATH="demo_ui.html"

usage() {
  cat <<USAGE
Usage: bash ui_api.sh [options]
Options:
  --host HOST         Host to bind (default: ${HOST})
  --port PORT         Port to bind (default: ${PORT})
  --timeout SECS      Health check timeout seconds (default: ${TIMEOUT})
  --open-path PATH    Path to open in browser for API UI (default: ${OPEN_PATH})
  --model PATH        Model directory to preload (default: auto-detect)
  --demo              Also run a static server and open demo_ui.html
  --ui-port PORT      Port for static UI server (default: ${UI_PORT})
  --ui-path PATH      File to open for demo UI (default: ${UI_PATH})
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
    --demo) DEMO="1"; shift;;
    --ui-port) UI_PORT="$2"; shift 2;;
    --ui-path) UI_PATH="$2"; shift 2;;
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

# Helper: check if UI static file responds
ui_is_up() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "$url" >/dev/null 2>&1
  else
    python - "$url" >/dev/null 2>&1 <<'PY'
import sys, urllib.request
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

# Open URL helper (robust cross-platform)
open_url() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1 || return 1; return 0; fi
  if command -v open >/dev/null 2>&1; then open "$url" >/dev/null 2>&1 || return 1; return 0; fi
  if command -v cmd.exe >/dev/null 2>&1; then cmd.exe /c start "" "$url" >/dev/null 2>&1 || return 1; return 0; fi
  if command -v powershell.exe >/dev/null 2>&1; then powershell.exe -NoProfile -NonInteractive -Command "Start-Process '$url'" >/dev/null 2>&1 || return 1; return 0; fi
  if command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$url" >/dev/null 2>&1 || return 1; return 0; fi
  return 1
}

# Decide which UI to open
STATIC_PID=""
API_UI_URL="http://${HOST}:${PORT}${OPEN_PATH}"
DEMO_UI_URL=""
if [[ "${DEMO}" == "1" ]]; then
  if [[ ! -f "${UI_PATH}" ]]; then
    log "ERROR: demo UI file not found: ${UI_PATH}"
    kill ${API_PID} >/dev/null 2>&1 || true
    exit 1
  fi
  API_BASE="http://${HOST}:${PORT}"
  TMP_UI_DIR=".run_demo_ui"
  mkdir -p "${TMP_UI_DIR}" || true
  # Build a patched copy of the demo UI that targets the running API base
  log "Patching demo UI -> ${TMP_UI_DIR}/index.html (API_BASE=${API_BASE})"
  python - "$API_BASE" "${UI_PATH}" "${TMP_UI_DIR}/index.html" <<'PY'
import sys, os
api_base, src, dst = sys.argv[1:4]
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()
# Inject helper right after the first <script> and route fetch() via helper
inject = """
    const __API_BASE = '%s';
    function __withBase(u){ try{ if(typeof u==='string' && u.startsWith('/')) return __API_BASE + u; }catch(e){} return u; }
""" % api_base
content = content.replace("<script>", "<script>\n" + inject, 1)
content = content.replace("fetch(url", "fetch(__withBase(url)")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, 'w', encoding='utf-8') as f:
    f.write(content)
PY
  # Start static server from patched dir
  log "Starting static UI server on http://127.0.0.1:${UI_PORT} (serving ${TMP_UI_DIR})"
  rm -f .ui_http_server.pid >/dev/null 2>&1 || true
  (
    cd "${TMP_UI_DIR}" || exit 1
    python -m http.server "${UI_PORT}" >/dev/null 2>&1 &
    echo $! > ../.ui_http_server.pid
  )
  sleep 0.4
  STATIC_PID=$(cat .ui_http_server.pid 2>/dev/null || echo "")
  DEMO_UI_URL="http://127.0.0.1:${UI_PORT}/index.html"
  # Wait for UI server
  log "Waiting for demo UI: ${DEMO_UI_URL} (timeout=${TIMEOUT}s)"
  START_TS=$(date +%s || echo 0)
  while ! ui_is_up "${DEMO_UI_URL}"; do
    sleep 1 || true
    NOW_TS=$(date +%s || echo 999999)
    if [[ $((NOW_TS - START_TS)) -ge ${TIMEOUT} ]]; then
      log "ERROR: Demo UI failed to become ready within ${TIMEOUT}s"
      kill ${API_PID} >/dev/null 2>&1 || true
      [[ -n "${STATIC_PID}" ]] && kill ${STATIC_PID} >/dev/null 2>&1 || true
      exit 1
    fi
  done
fi

# Open UIs
log "\n[3/3] Open UIs in default browser"
if [[ "$NO_BROWSER" != "1" ]]; then
  # Open API Swagger first
  if ! open_url "${API_UI_URL}"; then log "Could not auto-open API UI. Open manually: ${API_UI_URL}"; fi
  # Open Demo UI if requested
  if [[ "${DEMO}" == "1" ]]; then
    if ! open_url "${DEMO_UI_URL}"; then log "Could not auto-open Demo UI. Open manually: ${DEMO_UI_URL}"; fi
  fi
else
  log "Skipping browser open (--no-browser). API UI: ${API_UI_URL}$( [[ "${DEMO}" == "1" ]] && echo ", DEMO: ${DEMO_UI_URL}" )"
fi

log "API PID: ${API_PID}. $( [[ -n "${STATIC_PID}" ]] && echo "UI PID: ${STATIC_PID}. ")Press Ctrl+C to stop (if foreground)."
# Script exits but servers keep running in background.
exit 0
