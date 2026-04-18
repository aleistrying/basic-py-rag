#!/usr/bin/env bash
#
# RAG System — Linux Installer
# Works on Ubuntu 20.04+, Debian 11+, Fedora 38+, and derivatives.
# Requires NO Docker. Uses Ollama natively + local Qdrant file storage.
#
# Usage:
#   ./install_linux.sh
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
echo -e "${BOLD}${CYAN}║          RAG System  —  Linux Setup          ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${RESET}"
echo ""

# ─── 1. Platform check ───────────────────────────────────────────────────────
header "Step 1 / 5  —  Platform"

if [[ "$(uname -s)" != "Linux" ]]; then
    error "This installer is for Linux only. On macOS use:  ./install_mac.sh"
    exit 1
fi

ARCH=$(uname -m)
info "Linux on ${ARCH}"

# Detect package manager
if command -v apt-get &>/dev/null; then
    PKG_MANAGER="apt"
    success "Detected apt-based distro (Ubuntu / Debian)"
elif command -v dnf &>/dev/null; then
    PKG_MANAGER="dnf"
    success "Detected dnf-based distro (Fedora / RHEL / CentOS)"
elif command -v pacman &>/dev/null; then
    PKG_MANAGER="pacman"
    success "Detected pacman-based distro (Arch / Manjaro)"
else
    PKG_MANAGER="unknown"
    warn "Package manager not recognised — you may need to install curl and python3 manually."
fi

# Check for NVIDIA GPU
GPU_INFO=""
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    success "NVIDIA GPU detected: ${GPU_INFO} — Ollama will use CUDA automatically."
elif [[ -d /dev/dri ]] && ls /dev/dri/renderD* &>/dev/null 2>&1; then
    success "GPU render device found — Ollama will use it via ROCm/VAAPI where supported."
else
    info "No dedicated GPU detected — models will run on CPU (works fine for 3–7B models)."
fi

# ─── 2. System dependencies ──────────────────────────────────────────────────
header "Step 2 / 5  —  System dependencies"

install_pkg() {
    case "$PKG_MANAGER" in
        apt)    sudo apt-get install -y "$@" ;;
        dnf)    sudo dnf install -y "$@" ;;
        pacman) sudo pacman -S --noconfirm "$@" ;;
        *)      warn "Please install manually: $*"; return 0 ;;
    esac
}

NEED_INSTALL=()
for bin in curl git; do
    command -v "$bin" &>/dev/null || NEED_INSTALL+=("$bin")
done

if [[ ${#NEED_INSTALL[@]} -gt 0 ]]; then
    info "Installing: ${NEED_INSTALL[*]}"
    install_pkg "${NEED_INSTALL[@]}"
    success "System tools ready"
else
    success "curl and git already present"
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
    warn "Python 3.10+ not found. Attempting to install..."
    case "$PKG_MANAGER" in
        apt)
            sudo apt-get install -y python3.11 python3.11-venv python3-pip
            PYTHON="python3.11"
            ;;
        dnf)
            sudo dnf install -y python3.11
            PYTHON="python3.11"
            ;;
        pacman)
            sudo pacman -S --noconfirm python
            PYTHON="python3"
            ;;
        *)
            error "Could not auto-install Python. Please install Python 3.10+ and re-run."
            exit 1
            ;;
    esac
    success "Python installed"
fi

# Ensure python3-venv is available (Ubuntu/Debian split it out)
if [[ "$PKG_MANAGER" == "apt" ]]; then
    PYVER=$("$PYTHON" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PKG_VENV="python${PYVER}-venv"
    if ! "$PYTHON" -m venv --help &>/dev/null 2>&1; then
        info "Installing $PKG_VENV..."
        sudo apt-get install -y "$PKG_VENV" python3-pip || true
    fi
fi

# ─── 4. Ollama ───────────────────────────────────────────────────────────────
header "Step 4 / 5  —  Ollama (local AI engine)"

if ! command -v ollama &>/dev/null; then
    info "Installing Ollama via official install script..."
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed"
else
    OLLAMA_VER=$(ollama --version 2>/dev/null | head -1 || echo "version unknown")
    success "Ollama already installed ($OLLAMA_VER)"
fi

# Start Ollama if not already running
OLLAMA_WAS_RUNNING=false
if curl -s --max-time 3 http://localhost:11434/api/tags &>/dev/null; then
    OLLAMA_WAS_RUNNING=true
    success "Ollama server already running"
else
    info "Starting Ollama server in background..."
    ollama serve &>/tmp/ollama_rag.log &
    for i in $(seq 1 20); do
        sleep 1
        if curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
            success "Ollama server started"
            break
        fi
    done
    if ! curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
        warn "Ollama didn't respond in time — it may still be starting."
        warn "If model pull fails, run:  ollama serve   in a separate terminal."
    fi
fi

# ─── Model selection ─────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Recommended models for document research (8 GB RAM)${RESET}"
echo ""
echo -e "  ${BOLD}  1.  qwen3:4b     ${DIM}(2.5 GB — rivals models 18× its size · thinking mode · 256K ctx)${RESET}  ✓ RECOMMENDED"
echo -e "  ${DIM}  2.  phi4-mini     (2.3 GB — Microsoft Phi-4 Mini · top reasoning · multilingual)${RESET}"
echo -e "  ${DIM}  3.  gemma3:4b     (2.5 GB — Google Gemma 3 · 128K ctx · 35M+ downloads)${RESET}"
echo -e "  ${DIM}  4.  llama3.2:3b   (2.0 GB — Meta · lightweight classic · good all-round)${RESET}"
echo -e "  ${DIM}  5.  qwen3:1.7b    (1.4 GB — ultra-light · same Qwen3 quality in tiny size)${RESET}"
echo -e "  ${DIM}  6.  Custom — type any Ollama model name${RESET}"
if [[ -n "$GPU_INFO" ]]; then
    echo ""
    echo -e "  ${DIM}Tip: NVIDIA GPU detected — larger models (7b+) will work well too.${RESET}"
fi
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
    *)    MODEL="$MODEL_CHOICE" ;;
esac

info "Pulling model: ${BOLD}${MODEL}${RESET}  (downloads once, may take several minutes)..."
if ollama pull "$MODEL" 2>&1; then
    success "Model '${MODEL}' is ready"
else
    warn "Could not pull '${MODEL}'. Check your internet connection."
    warn "You can pull it later:  ollama pull ${MODEL}"
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
# .env.local — RAG System local/Linux configuration
# Generated by install_linux.sh — edit freely

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
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
ENV_FILE="$SCRIPT_DIR/.env.local"

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
echo "  ╔════════════════════════════════════╗"
echo "  ║         RAG System                 ║"
echo "  ╠════════════════════════════════════╣"
echo "  ║  1.  Interactive CLI               ║"
echo "  ║  2.  Research UI  ← for daily use  ║"
echo "  ║  3.  Full web app  (all features)  ║"
echo "  ╚════════════════════════════════════╝"
echo ""
printf "  Choose [2]: "
read -r CHOICE
CHOICE="${CHOICE:-2}"

# Try to find a working browser opener
open_browser() {
    local url="$1"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$url" &>/dev/null &
    elif command -v sensible-browser &>/dev/null; then
        sensible-browser "$url" &>/dev/null &
    elif command -v gnome-open &>/dev/null; then
        gnome-open "$url" &>/dev/null &
    fi
}

case "$CHOICE" in
    1)
        exec "$VENV_PY" "$SCRIPT_DIR/cli.py" "$@"
        ;;
    2)
        echo ""
        echo "  Starting Research UI…"
        echo "  Opening http://localhost:${APP_PORT}/research"
        echo "  Press Ctrl+C to stop."
        echo ""
        (sleep 2 && open_browser "http://localhost:${APP_PORT}/research" || true) &
        exec "$VENV_PY" -m uvicorn app.main:app \
            --host 0.0.0.0 --port "$APP_PORT"
        ;;
    3)
        echo ""
        echo "  Starting full web app at http://localhost:${APP_PORT}"
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

# Make install script executable (in case it wasn't)
chmod +x "$SCRIPT_DIR/install_linux.sh"

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
