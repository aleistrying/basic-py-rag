#!/usr/bin/env bash
#
# RAG System — macOS Installer
# Works on Apple Silicon (M1/M2/M3) and Intel Macs.
# Requires NO Docker. Uses Ollama natively + local Qdrant file storage.
#
# Usage:
#   ./install_mac.sh
#
set -euo pipefail

# ─── Colours ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

info()    { echo -e "${CYAN}  ℹ  $*${RESET}"; }
success() { echo -e "${GREEN}  ✓  $*${RESET}"; }
warn()    { echo -e "${YELLOW}  ⚠  $*${RESET}"; }
error()   { echo -e "${RED}  ✗  $*${RESET}" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}━━━━  $*${RESET}\n"; }
ask()     { echo -en "${BOLD}  ?  $*${RESET}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
DATA_DIR="$SCRIPT_DIR/data"
ENV_FILE="$SCRIPT_DIR/.env.local"

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║          RAG System  —  macOS Setup          ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${RESET}"
echo ""

# ─── 1. Platform check ───────────────────────────────────────────────────────
header "Step 1 / 7  —  Platform"

if [[ "$(uname -s)" != "Darwin" ]]; then
    error "This installer is for macOS only. On Linux use:  ./install_linux.sh"
    exit 1
fi

ARCH=$(uname -m)
OS_VER=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
info "macOS ${OS_VER} on ${ARCH}"

if [[ "$ARCH" == "arm64" ]]; then
    success "Apple Silicon detected — Ollama will use Metal GPU automatically."
else
    success "Intel Mac detected — models will run on CPU (slower but works)."
fi

# ─── 2. Xcode Command Line Tools ─────────────────────────────────────────────
header "Step 2 / 7  —  Xcode Command Line Tools"

if ! xcode-select -p &>/dev/null 2>&1; then
    warn "Xcode Command Line Tools not installed (required by Homebrew and Python)."
    warn "A system dialog will appear — click 'Install' and wait for it to finish."
    xcode-select --install 2>/dev/null || true
    # Poll until installed (the dialog is async)
    info "Waiting for Xcode CLT installation to complete..."
    until xcode-select -p &>/dev/null 2>&1; do
        sleep 10
        printf "."
    done
    echo ""
    success "Xcode Command Line Tools installed"
else
    success "Xcode Command Line Tools present ($(xcode-select -p))"
fi

# ─── 3. Homebrew ─────────────────────────────────────────────────────────────
header "Step 3 / 7  —  Homebrew"

if ! command -v brew &>/dev/null; then
    warn "Homebrew not found. Installing... (you may be asked for your password)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH — Apple Silicon uses /opt/homebrew, Intel uses /usr/local
    if [[ "$ARCH" == "arm64" ]] && [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    success "Homebrew installed"
else
    success "Homebrew already installed"
    # Ensure brew is in PATH for Apple Silicon sessions that haven't set it up
    if [[ "$ARCH" == "arm64" ]] && ! command -v brew &>/dev/null 2>&1; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# ─── 4. Docker (optional) ─────────────────────────────────────────────────────
header "Step 4 / 7  —  Docker Desktop (optional)"

DOCKER_INSTALLED=false
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version 2>/dev/null | head -1 || echo "version unknown")
    success "Docker already installed ($DOCKER_VER)"
    DOCKER_INSTALLED=true
else
    echo ""
    info "Docker lets you run the full stack with one command:  docker compose up"
    info "It is optional — this installer also works without it (local mode)."
    echo ""
    ask "Install Docker Desktop via Homebrew? [Y/n]: "
    read -r DOCKER_CHOICE
    DOCKER_CHOICE="${DOCKER_CHOICE:-Y}"

    if [[ "$DOCKER_CHOICE" =~ ^[Yy]$ ]]; then
        info "Downloading Docker Desktop (~500 MB) — please wait..."
        brew install --cask docker
        success "Docker Desktop installed"
        info "First launch requires accepting the license agreement."
        info "Open it from Applications or run:  open /Applications/Docker.app"
        DOCKER_INSTALLED=true
    else
        info "Skipping Docker — system will run in local mode (no Docker required)."
    fi
fi

# ─── 5. Python ─────────────────────────────────────────────────────────────
header "Step 5 / 7  —  Python"

PYTHON=""
for py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        ver=$("$py" --version 2>&1 | awk '{print $2}')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
            PYTHON="$py"
            success "Found $py ($ver)"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    warn "Python 3.10+ not found. Installing Python 3.11 via Homebrew..."
    brew install python@3.11
    PYTHON="$(brew --prefix)/bin/python3.11"
    success "Python 3.11 installed"
fi

# ─── 4. Ollama ───────────────────────────────────────────────────────────────
header "Step 6 / 7  —  Ollama (local AI engine)"

if ! command -v ollama &>/dev/null; then
    warn "Ollama not found. Installing via Homebrew..."
    brew install ollama
    success "Ollama installed"
else
    OLLAMA_VER=$(ollama --version 2>/dev/null | head -1 || echo "version unknown")
    success "Ollama already installed ($OLLAMA_VER)"
fi

# Start Ollama in background if not already running
OLLAMA_WAS_RUNNING=false
if curl -s --max-time 3 http://localhost:11434/api/tags &>/dev/null; then
    OLLAMA_WAS_RUNNING=true
    success "Ollama server already running"
else
    info "Starting Ollama server in background..."
    ollama serve &>/tmp/ollama_rag.log &
    OLLAMA_BG_PID=$!
    # Wait up to 15 s for it to come up
    for i in $(seq 1 15); do
        sleep 1
        if curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
            success "Ollama server started (PID $OLLAMA_BG_PID)"
            break
        fi
    done
    if ! curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
        warn "Ollama server didn't respond in time — it may still be starting."
        warn "If model pull fails, run:  ollama serve   in a separate terminal."
    fi
fi

# ─── Model selection ─────────────────────────────────────────────────────────

# Detect total RAM
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
success "Detected ${RAM_GB} GB RAM"

# Detect Mac model identifier and year (e.g. "MacBookAir10,1" or "MacBookPro16,1")
MAC_MODEL=$(sysctl -n hw.model 2>/dev/null || system_profiler SPHardwareDataType 2>/dev/null \
    | awk -F': ' '/Model Identifier/{print $2; exit}' || echo "unknown")
MAC_YEAR=$(system_profiler SPHardwareDataType 2>/dev/null \
    | awk -F': ' '/Model Name|Model Year/{print $2}' | tail -1 || echo "")
info "Mac model: ${MAC_MODEL}${MAC_YEAR:+  (${MAC_YEAR})}"

# On Apple Silicon (arm64) Ollama uses Metal/MLX — full GPU acceleration.
# On Intel (x86_64) Ollama is CPU-only — inference is ~3–6× slower.
# For Intel, we drop one tier so the model actually runs at bearable speed.
if [[ "$ARCH" == "arm64" ]]; then
    _GPU_NOTE="Apple Silicon — Metal GPU acceleration active (all ${RAM_GB} GB RAM available to model)"
    _INTEL=false
else
    _GPU_NOTE="Intel Mac — CPU-only inference (no Metal GPU). Smaller models are faster."
    _INTEL=true
    warn "Intel Mac detected: inference runs on CPU only. Models will be slower."
    warn "Recommendation adjusted downward by one tier for comfortable speed."
fi
info "$_GPU_NOTE"

# Pick the best default model for this machine
# Intel gets one tier lower because CPU-only makes larger models painfully slow
if [[ "$_INTEL" == "true" ]]; then
    # Intel tiers (CPU-only — prioritise speed)
    if   [[ $RAM_GB -ge 16 ]]; then
        _DEFAULT_MODEL_NUM=1; _SUGGEST_MODEL="qwen3:4b"
        _SUGGEST_NOTE="(best balance on Intel ${RAM_GB} GB — thinking mode, still usable on CPU)"
    elif [[ $RAM_GB -ge 8 ]]; then
        _DEFAULT_MODEL_NUM=5; _SUGGEST_MODEL="qwen3:1.7b"
        _SUGGEST_NOTE="(recommended for Intel ${RAM_GB} GB — small enough to run at good speed on CPU)"
    else
        _DEFAULT_MODEL_NUM=5; _SUGGEST_MODEL="qwen3:1.7b"
        _SUGGEST_NOTE="(only safe choice for Intel ${RAM_GB} GB on CPU)"
    fi
else
    # Apple Silicon tiers (Metal GPU — full RAM is GPU memory)
    if   [[ $RAM_GB -ge 20 ]]; then
        _DEFAULT_MODEL_NUM=7; _SUGGEST_MODEL="qwen3:14b"
        _SUGGEST_NOTE="(best quality for legal research — 9 GB, your ${RAM_GB} GB unified mem handles it)"
    elif [[ $RAM_GB -ge 12 ]]; then
        _DEFAULT_MODEL_NUM=6; _SUGGEST_MODEL="qwen3:8b"
        _SUGGEST_NOTE="(noticeably stronger reasoning — 5 GB, fits your ${RAM_GB} GB unified mem)"
    elif [[ $RAM_GB -ge 7 ]]; then
        _DEFAULT_MODEL_NUM=1; _SUGGEST_MODEL="qwen3:4b"
        _SUGGEST_NOTE="(best fit for ${RAM_GB} GB M-series — thinking mode, 256K ctx)"
    else
        _DEFAULT_MODEL_NUM=5; _SUGGEST_MODEL="qwen3:1.7b"
        _SUGGEST_NOTE="(recommended for ${RAM_GB} GB — ultra-light Qwen3 with thinking)"
    fi
fi

echo ""
echo -e "  ${BOLD}Model recommendation for your machine (${RAM_GB} GB${_INTEL:+, CPU-only}):${RESET}"
echo -e "  ${BOLD}${GREEN}  →  ${_SUGGEST_MODEL}  ${DIM}${_SUGGEST_NOTE}${RESET}"
echo ""
echo -e "  All options (Qwen3 family uses thinking mode for legal reasoning):"
echo ""
echo -e "  ${DIM}  1.  qwen3:4b     (2.5 GB — rivals models 18× its size · thinking mode · 256K ctx)${RESET}"
echo -e "  ${DIM}  2.  phi4-mini     (2.3 GB — Microsoft Phi-4 Mini · top-tier reasoning · multilingual)${RESET}"
echo -e "  ${DIM}  3.  gemma3:4b     (2.5 GB — Google Gemma 3 · 128K ctx · 35M+ downloads · proven)${RESET}"
echo -e "  ${DIM}  4.  llama3.2:3b   (2.0 GB — Meta · lightweight classic · good all-round)${RESET}"
echo -e "  ${DIM}  5.  qwen3:1.7b    (1.4 GB — ultra-light · Qwen3 quality in tiny size)${RESET}"
echo -e "  ${DIM}  6.  qwen3:8b      (5.0 GB — noticeably stronger reasoning · needs 8 GB+ Apple Silicon)${RESET}"
echo -e "  ${DIM}  7.  qwen3:14b     (9.0 GB — best open legal reasoning model · needs 16 GB+ Apple Silicon)${RESET}"
echo -e "  ${DIM}  8.  Custom — type any Ollama model name${RESET}"
echo ""
if [[ "$_INTEL" == "false" ]]; then
    echo -e "  ${DIM}Tip: Ollama uses Apple MLX/Metal backend — ~3× faster than CPU inference.${RESET}"
    echo -e "  ${DIM}     On Apple Silicon, unified memory = all RAM is available to the model.${RESET}"
else
    echo -e "  ${DIM}Tip: Intel Macs run inference on CPU. qwen3:1.7b gives the best speed/quality trade-off.${RESET}"
    echo -e "  ${DIM}     A 2020+ Mac with Apple Silicon would run qwen3:4b or larger comfortably.${RESET}"
fi
echo ""
ask "Choose model [${_DEFAULT_MODEL_NUM}]: "
read -r MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-${_DEFAULT_MODEL_NUM}}"

case "$MODEL_CHOICE" in
    1)    MODEL="qwen3:4b" ;;
    2)    MODEL="phi4-mini" ;;
    3)    MODEL="gemma3:4b" ;;
    4)    MODEL="llama3.2:3b" ;;
    5)    MODEL="qwen3:1.7b" ;;
    6)    MODEL="qwen3:8b" ;;
    7)    MODEL="qwen3:14b" ;;
    8)    ask "Enter model name: "; read -r MODEL ;;
    "")   MODEL="$_SUGGEST_MODEL" ;;
    *)    MODEL="$MODEL_CHOICE" ;;
esac

info "Pulling model: ${BOLD}${MODEL}${RESET}  (downloads once, may take several minutes on first run)..."
if ollama pull "$MODEL" 2>&1; then
    success "Model '${MODEL}' is ready"
else
    error "Could not pull '${MODEL}'. Check your internet connection."
    warn "You can pull it later:  ollama pull ${MODEL}"
    MODEL="${MODEL}"   # keep as configured, will work offline if already cached
fi

# ─── 5. Python deps ──────────────────────────────────────────────────────────
header "Step 7 / 7  —  Python dependencies & configuration"

VENV_PY="$VENV_DIR/bin/python"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating Python virtual environment at .venv/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

info "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip --quiet 2>&1 | grep -v "^$" || true

info "Installing Python dependencies (this may take 5–10 minutes on first run)..."
if ! "$VENV_PY" -m pip install -r "$SCRIPT_DIR/requirements_local.txt" --progress-bar=off; then
    # Retry without psycopg2-binary — needs PostgreSQL dev headers (optional package)
    warn "Retrying without optional PostgreSQL driver (psycopg2-binary)..."
    TEMP_REQ="$VENV_DIR/requirements_no_psql.txt"
    grep -v "^psycopg2" "$SCRIPT_DIR/requirements_local.txt" > "$TEMP_REQ"
    if ! "$VENV_PY" -m pip install -r "$TEMP_REQ" --progress-bar=off; then
        rm -f "$TEMP_REQ"
        error "Python package installation failed — see errors above."
        error "Common fix:  xcode-select --install"
        error "Then re-run: ./install_mac.sh"
        exit 1
    fi
    rm -f "$TEMP_REQ"
    warn "psycopg2-binary skipped — PostgreSQL features disabled."
    warn "To enable later:  brew install libpq && pip install psycopg2-binary"
fi
success "Python dependencies installed"

# Verify that critical packages actually imported
info "Verifying installation..."
if ! "$VENV_PY" - <<'PYVERIFY'
import sys
checks = [
    ("fastapi",               "FastAPI web framework"),
    ("pandas",                "Data analysis (pandas)"),
    ("qdrant_client",         "Vector database (Qdrant)"),
    ("sentence_transformers", "Embedding model"),
    ("ollama",                "Ollama AI client"),
    ("uvicorn",               "Web server (uvicorn)"),
]
failed = []
for mod, label in checks:
    try:
        __import__(mod)
        print(f"    \u2713  {label}")
    except ImportError:
        print(f"    \u2717  {label}  \u2190 MISSING", file=sys.stderr)
        failed.append(mod)
if failed:
    print(f"\n  ERROR: Missing packages: {', '.join(failed)}", file=sys.stderr)
    sys.exit(1)
PYVERIFY
then
    error "Required packages are missing — see errors above."
    exit 1
fi
success "All core packages verified"

# Pre-download embedding model
info "Downloading embedding model (one-time, ~280 MB)..."
MODEL_CACHE="$DATA_DIR/models/embeddings"
mkdir -p "$MODEL_CACHE"
"$VENV_PY" - <<PYEOF
import sys
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('intfloat/multilingual-e5-base', cache_folder='${MODEL_CACHE}')
    print('  ✓  Embedding model ready')
except Exception as err:
    print('  ⚠  Will download on first use: ' + str(err), file=sys.stderr)
PYEOF

# ─── Write .env.local ────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/clean" "$DATA_DIR/qdrant_local" "$DATA_DIR/models/embeddings"

cat > "$ENV_FILE" <<ENVEOF
# .env.local — RAG System local/macOS configuration
# Generated by install_mac.sh — edit freely

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=${MODEL}

# Qdrant in local file mode (no Docker needed)
QDRANT_LOCAL_PATH=${DATA_DIR}/qdrant_local
QDRANT_COLLECTION=course_docs_clean

# Disable PostgreSQL for local use
USE_PGVECTOR=false
USE_QDRANT=true

# Embedding model cache (avoids re-downloading)
HF_HOME=${DATA_DIR}/models/huggingface
SENTENCE_TRANSFORMERS_HOME=${DATA_DIR}/models/embeddings

# Web interface port
APP_PORT=8080
ENVEOF

success "Configuration written to .env.local"

# ─── Ensure start.sh is executable ───────────────────────────────────────────
chmod +x "$SCRIPT_DIR/start.sh"
success "start.sh is ready"

# ─── End-to-end verification ─────────────────────────────────────────────────
header "Verification  —  sanity-checking the full installation"

PASS=0; FAIL=0
check_pass() { echo -e "    ${GREEN}✓  $*${RESET}"; PASS=$((PASS+1)); }
check_fail() { echo -e "    ${RED}✗  $*${RESET}"; FAIL=$((FAIL+1)); }
check_warn() { echo -e "    ${YELLOW}⚠  $*${RESET}"; }

# 1. Python venv
[[ -f "$VENV_DIR/bin/python" ]] && check_pass "Python venv present" || check_fail "Python venv missing — re-run installer"

# 2. Critical packages
"$VENV_DIR/bin/python" - <<'PYCHECK' && check_pass "Core Python packages importable" || check_fail "Some Python packages failed to import — re-run installer"
import fastapi, qdrant_client, sentence_transformers, uvicorn, ollama
PYCHECK

# 3. .env.local written and has QDRANT_LOCAL_PATH
if [[ -f "$ENV_FILE" ]] && grep -q "QDRANT_LOCAL_PATH" "$ENV_FILE"; then
    check_pass ".env.local written with QDRANT_LOCAL_PATH"
else
    check_fail ".env.local missing or incomplete"
fi

# 4. Qdrant data directory
[[ -d "$DATA_DIR/qdrant_local" ]] && check_pass "Qdrant local directory exists" || check_fail "Qdrant directory missing: $DATA_DIR/qdrant_local"

# 5. Embedding model cached
if ls "$DATA_DIR/models/embeddings/models--intfloat--multilingual-e5-base/snapshots" &>/dev/null 2>&1; then
    check_pass "Embedding model cached"
else
    check_warn "Embedding model not yet cached — it will download on first use (~280 MB)"
fi

# 6. Ollama running + model available
if curl -s --max-time 5 "http://localhost:11434/api/tags" &>/dev/null; then
    check_pass "Ollama server reachable"
    MODELS_JSON=$(curl -s --max-time 5 "http://localhost:11434/api/tags" 2>/dev/null || echo "{}")
    if echo "$MODELS_JSON" | grep -q "\"name\""; then
        check_pass "Ollama has at least one model loaded"
    else
        check_warn "Ollama running but no models listed — try:  ollama pull ${MODEL}"
    fi
else
    check_warn "Ollama server not running — it will start automatically when you run start.sh"
fi

# 7. start.sh is executable and contains cd guard
if [[ -x "$SCRIPT_DIR/start.sh" ]] && grep -q 'cd "\$SCRIPT_DIR"' "$SCRIPT_DIR/start.sh"; then
    check_pass "start.sh is executable with working-directory guard"
else
    check_fail "start.sh missing or malformed"
fi

# 8. Quick app import smoke-test (no server started)
cd "$SCRIPT_DIR"
"$VENV_DIR/bin/python" -c "
import sys, os
os.environ.setdefault('QDRANT_LOCAL_PATH', '${DATA_DIR}/qdrant_local')
from app.rag import generate_llm_answer, search_knowledge_base
from app.pipeline_rag import pipeline_search, analyze_question
print('ok')
" 2>/dev/null && check_pass "App modules import cleanly" || check_warn "App import check failed (may be OK if you haven't ingested docs yet)"

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo -e "  ${BOLD}${GREEN}All checks passed${RESET} ($PASS passed, 0 failed)"
else
    echo -e "  ${BOLD}${YELLOW}$FAIL check(s) failed${RESET} — see items marked ✗ above"
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║            Installation complete!            ║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}To add your documents later:${RESET}"
echo -e "    Copy PDF or text files to  ${BOLD}./data/raw/${RESET}"
echo -e "    Then run the CLI and choose ${BOLD}\"Ingest documents\"${RESET}"
echo ""
echo -e "  ${DIM}Ollama model: ${MODEL}${RESET}"
echo -e "  ${DIM}Vector DB:    local file storage (./data/qdrant_local)${RESET}"
echo -e "  ${DIM}No Docker required.${RESET}"
echo ""
ask "Launch the Research UI now? [Y/n]: "
read -r LAUNCH_NOW
LAUNCH_NOW="${LAUNCH_NOW:-Y}"
if [[ "$LAUNCH_NOW" =~ ^[Yy]$ ]]; then
    exec "$SCRIPT_DIR/start.sh"
else
    echo ""
    echo -e "  Run  ${BOLD}./start.sh${RESET}  whenever you're ready."
    echo ""
fi
