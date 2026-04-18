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
header "Step 1 / 5  —  Platform"

if [[ "$(uname -s)" != "Darwin" ]]; then
    error "This installer is for macOS only. On Linux, use Docker Compose instead:"
    echo "       docker compose up -d"
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

# ─── 2. Homebrew ─────────────────────────────────────────────────────────────
header "Step 2 / 5  —  Homebrew"

if ! command -v brew &>/dev/null; then
    warn "Homebrew not found. Installing... (you may be asked for your password)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Apple Silicon
    if [[ "$ARCH" == "arm64" ]] && [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    success "Homebrew installed"
else
    success "Homebrew already installed"
fi

# ─── 3. Python ───────────────────────────────────────────────────────────────
header "Step 3 / 5  —  Python"

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
header "Step 4 / 5  —  Ollama (local AI engine)"

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
echo ""
echo -e "  ${BOLD}Best models for M1 8 GB MacBook Air — ranked by quality/size:${RESET}"
echo ""
echo -e "  ${BOLD}  1.  qwen3:4b     ${DIM}(2.5 GB — rivals models 18× its size · thinking mode · 256K ctx)${RESET}  ✓ RECOMMENDED"
echo -e "  ${DIM}  2.  phi4-mini     (2.3 GB — Microsoft Phi-4 Mini · top-tier reasoning · multilingual)${RESET}"
echo -e "  ${DIM}  3.  gemma3:4b     (2.5 GB — Google Gemma 3 · 128K ctx · 35M+ downloads · proven)${RESET}"
echo -e "  ${DIM}  4.  llama3.2:3b   (2.0 GB — Meta · lightweight classic · good all-round)${RESET}"
echo -e "  ${DIM}  5.  qwen3:1.7b    (1.4 GB — ultra-light · Qwen3 quality in tiny size)${RESET}"
echo -e "  ${DIM}  6.  Custom — type any Ollama model name${RESET}"
echo ""
echo -e "  ${DIM}Tip: Latest Ollama uses Apple MLX backend (~3× faster on M1).${RESET}"
echo -e "  ${DIM}     Run  brew upgrade ollama  before your first model pull.${RESET}"
echo ""
ask "Choose model [1]: "
read -r MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-1}"

case "$MODEL_CHOICE" in
    1|"") MODEL="qwen3:4b" ;;
    2)    MODEL="phi4-mini" ;;
    3)    MODEL="gemma3:4b" ;;
    4)    MODEL="llama3.2:3b" ;;
    5)    MODEL="qwen3:1.7b" ;;
    6)    ask "Enter model name: "; read -r MODEL ;;
    *)    MODEL="$MODEL_CHOICE" ;;  # treat raw input as model name
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
header "Step 5 / 5  —  Python dependencies & configuration"

VENV_PY="$VENV_DIR/bin/python"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating Python virtual environment at .venv/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

info "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip --quiet 2>&1 | grep -v "^$" || true

info "Installing Python dependencies (this may take a few minutes)..."
"$VENV_PY" -m pip install -r "$SCRIPT_DIR/requirements_local.txt" --quiet 2>&1 | grep -v "^$" || true
success "Python dependencies installed"

# Pre-download embedding model
info "Downloading embedding model (one-time, ~280 MB)..."
MODEL_CACHE="$DATA_DIR/models/embeddings"
mkdir -p "$MODEL_CACHE"
"$VENV_PY" - <<PYEOF
import sys
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('intfloat/multilingual-e5-base', cache_folder='${MODEL_CACHE}')
    print("  ✓  Embedding model ready")
except Exception as e:
    print(f"  ⚠  Will download on first use: {e}", file=sys.stderr)
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

# ─── Write start.sh ──────────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/start.sh" <<'STARTSCRIPT'
#!/usr/bin/env bash
# RAG System — one-command launcher
# Run this after install_mac.sh to start the system.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
ENV_FILE="$SCRIPT_DIR/.env.local"

# Load local env
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
APP_PORT="${APP_PORT:-8080}"

# Ensure Ollama is running
if ! curl -s --max-time 2 "$OLLAMA_HOST/api/tags" &>/dev/null; then
    echo "  Starting Ollama..."
    ollama serve &>/tmp/ollama_rag.log &
    sleep 3
fi

clear
echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║         RAG System               ║"
echo "  ╠══════════════════════════════════╣"
echo "  ║  1.  Interactive CLI (recommended)║"
echo "  ║  2.  Web interface               ║"
echo "  ╚══════════════════════════════════╝"
echo ""
printf "  Choose [1]: "
read -r CHOICE
CHOICE="${CHOICE:-1}"

case "$CHOICE" in
    1)
        exec "$VENV_PY" "$SCRIPT_DIR/cli.py" "$@"
        ;;
    2)
        echo ""
        echo "  Open your browser at:  http://localhost:${APP_PORT}"
        echo "  Press Ctrl+C to stop."
        echo ""
        exec "$VENV_PY" -m uvicorn app.main:app \
            --host 0.0.0.0 --port "$APP_PORT" --reload
        ;;
    *)
        echo "  Invalid choice."
        exit 1
        ;;
esac
STARTSCRIPT
chmod +x "$SCRIPT_DIR/start.sh"
success "Created start.sh"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║            Installation complete!            ║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}To start the system:${RESET}"
echo ""
echo -e "    ${BOLD}./start.sh${RESET}"
echo ""
echo -e "  ${BOLD}To add your documents:${RESET}"
echo -e "    Copy PDF or text files to  ${BOLD}./data/raw/${RESET}"
echo -e "    Then run the CLI and choose ${BOLD}\"Ingest documents\"${RESET}"
echo ""
echo -e "  ${DIM}Ollama model: ${MODEL}${RESET}"
echo -e "  ${DIM}Vector DB:    local file storage (./data/qdrant_local)${RESET}"
echo -e "  ${DIM}No Docker required.${RESET}"
echo ""
