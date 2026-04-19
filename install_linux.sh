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
header "Step 1 / 6  —  Platform"

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

# Check for NVIDIA GPU and driver health
GPU_INFO=""
NVIDIA_NEEDS_DRIVER=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "0")
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
    success "NVIDIA GPU detected: ${GPU_INFO} (driver ${DRIVER_VER})"
    # Blackwell (RTX 50-series) and recent GPUs need driver 565+
    # PyTorch CUDA libraries also require a recent driver
    if [[ "$DRIVER_MAJOR" -lt 565 ]]; then
        warn "Driver ${DRIVER_VER} is outdated — PyTorch/CUDA may not work correctly."
        NVIDIA_NEEDS_DRIVER=true
    else
        success "NVIDIA driver is up to date — Ollama will use CUDA automatically."
    fi
elif lspci 2>/dev/null | grep -qi nvidia; then
    warn "NVIDIA GPU found in hardware but nvidia-smi is not working — driver may be missing."
    GPU_INFO="NVIDIA (driver not installed)"
    NVIDIA_NEEDS_DRIVER=true
elif [[ -d /dev/dri ]] && ls /dev/dri/renderD* &>/dev/null 2>&1; then
    success "GPU render device found — Ollama will use it via ROCm/VAAPI where supported."
else
    info "No dedicated GPU detected — models will run on CPU (works fine for 3–7B models)."
fi

# ─── 2. System dependencies ──────────────────────────────────────────────────
header "Step 2 / 6  —  System dependencies"

install_pkg() {
    case "$PKG_MANAGER" in
        apt)    sudo apt-get install -y "$@" ;;
        dnf)    sudo dnf install -y "$@" ;;
        pacman) sudo pacman -S --noconfirm "$@" ;;
        *)      warn "Please install manually: $*"; return 0 ;;
    esac
}

# Refresh package index — required on fresh systems before any apt installs.
# We use || true because third-party repos may have key/404 errors that are
# irrelevant to our packages (all of which are in the main Ubuntu archive).
if [[ "$PKG_MANAGER" == "apt" ]]; then
    info "Refreshing package index (sudo required)..."
    sudo apt-get update -qq 2>&1 | grep -v "^W:\|^N:" || true
    success "Package index refreshed"
fi

# Install essential tools (curl, git, xdg-utils for browser open)
NEED_INSTALL=()
for bin in curl git; do
    command -v "$bin" &>/dev/null || NEED_INSTALL+=("$bin")
done
command -v xdg-open &>/dev/null || NEED_INSTALL+=("xdg-utils")

if [[ ${#NEED_INSTALL[@]} -gt 0 ]]; then
    info "Installing system tools: ${NEED_INSTALL[*]}"
    install_pkg "${NEED_INSTALL[@]}"
fi
success "System tools ready"

# ─── 2b. NVIDIA driver (if needed) ───────────────────────────────────────────
if [[ "$NVIDIA_NEEDS_DRIVER" == "true" && "$PKG_MANAGER" == "apt" ]]; then
    echo ""
    warn "An outdated or missing NVIDIA driver was detected."
    warn "The system needs driver 565+ for CUDA and Ollama GPU acceleration."
    echo ""
    ask "Install/upgrade to the latest recommended NVIDIA driver now? [Y/n]: "
    read -r DRIVER_CHOICE
    DRIVER_CHOICE="${DRIVER_CHOICE:-Y}"

    if [[ "$DRIVER_CHOICE" =~ ^[Yy]$ ]]; then
        info "Installing ubuntu-drivers-common tool..."
        sudo apt-get install -y ubuntu-drivers-common

        if command -v ubuntu-drivers &>/dev/null; then
            # Prefer ubuntu-drivers autoinstall — it picks the right driver for the GPU
            info "Detecting best driver for your GPU..."
            RECOMMENDED=$(ubuntu-drivers devices 2>/dev/null \
                | grep "recommended" | awk '{print $3}' | head -1 || true)

            if [[ -n "$RECOMMENDED" ]]; then
                info "Installing recommended driver: ${RECOMMENDED}"
                sudo apt-get install -y "$RECOMMENDED"
            else
                info "Running ubuntu-drivers autoinstall..."
                sudo ubuntu-drivers autoinstall || {
                    warn "ubuntu-drivers autoinstall failed; falling back to nvidia-driver-570..."
                    sudo apt-get install -y nvidia-driver-570 || \
                        warn "Driver install failed — please install manually and reboot."
                }
            fi
        else
            # ubuntu-drivers not available — install latest known driver directly
            warn "ubuntu-drivers tool not available; installing nvidia-driver-570 directly..."
            sudo apt-add-repository -y ppa:graphics-drivers/ppa 2>/dev/null || true
            sudo apt-get update -qq 2>&1 | grep -v "^W:\|^N:" || true
            sudo apt-get install -y nvidia-driver-570 || \
                warn "Driver install failed — please install manually from https://www.nvidia.com/drivers"
        fi

        success "NVIDIA driver install complete."
        warn "A system reboot is required to activate the new driver."
        warn "After rebooting, re-run  ./install_linux.sh  to complete setup."
        echo ""
        ask "Reboot now? [y/N]: "
        read -r REBOOT_NOW
        REBOOT_NOW="${REBOOT_NOW:-N}"
        if [[ "$REBOOT_NOW" =~ ^[Yy]$ ]]; then
            info "Rebooting in 5 seconds... (re-run install_linux.sh after boot)"
            sleep 5; sudo reboot
        else
            warn "Continuing without reboot — GPU won't be active until next restart."
        fi
    else
        warn "Skipping driver install. Models will run on CPU until driver is updated."
    fi
elif [[ "$NVIDIA_NEEDS_DRIVER" == "true" && "$PKG_MANAGER" == "dnf" ]]; then
    warn "Outdated NVIDIA driver detected. To install the latest driver on Fedora:"
    warn "  sudo dnf install akmod-nvidia   (requires RPM Fusion repo)"
    warn "Then reboot and re-run this installer."
fi

# ─── GPU / CPU mode selection ────────────────────────────────────────────────
USE_GPU=false
if [[ -n "$GPU_INFO" && "$NVIDIA_NEEDS_DRIVER" == "false" ]]; then
    echo ""
    info "Working GPU detected: ${GPU_INFO}"
    ask "Use GPU for AI inference? (faster) [Y/n]: "
    read -r GPU_CHOICE
    GPU_CHOICE="${GPU_CHOICE:-Y}"
    if [[ "$GPU_CHOICE" =~ ^[Yy]$ ]]; then
        USE_GPU=true
        success "GPU mode enabled — Ollama and embeddings will use CUDA"
    else
        USE_GPU=false
        info "CPU mode selected — models will run on CPU only"
    fi
else
    info "No working GPU available — running in CPU mode"
fi

# ─── 3. Docker (optional) ────────────────────────────────────────────────────
header "Step 3 / 6  —  Docker (optional)"

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
    ask "Install Docker Engine now? [Y/n]: "
    read -r DOCKER_CHOICE
    DOCKER_CHOICE="${DOCKER_CHOICE:-Y}"

    if [[ "$DOCKER_CHOICE" =~ ^[Yy]$ ]]; then
        if [[ "$PKG_MANAGER" == "apt" ]]; then
            info "Installing Docker Engine via official install script..."
            curl -fsSL https://get.docker.com | sh
            if getent group docker &>/dev/null; then
                sudo usermod -aG docker "$USER" || true
                warn "Log out and back in (or run: newgrp docker) before using Docker without sudo."
            fi
            sudo systemctl enable docker --quiet 2>/dev/null || true
            sudo systemctl start  docker 2>/dev/null || true
            DOCKER_INSTALLED=true
        elif [[ "$PKG_MANAGER" == "dnf" ]]; then
            info "Installing Docker Engine on Fedora/RHEL..."
            sudo dnf -y install dnf-plugins-core 2>/dev/null || true
            sudo dnf config-manager --add-repo \
                https://download.docker.com/linux/fedora/docker-ce.repo 2>/dev/null || true
            sudo dnf install -y docker-ce docker-ce-cli containerd.io \
                docker-buildx-plugin docker-compose-plugin
            sudo usermod -aG docker "$USER" || true
            sudo systemctl enable docker --quiet 2>/dev/null || true
            sudo systemctl start  docker 2>/dev/null || true
            DOCKER_INSTALLED=true
        elif [[ "$PKG_MANAGER" == "pacman" ]]; then
            info "Installing Docker on Arch/Manjaro..."
            sudo pacman -S --noconfirm docker docker-compose
            sudo usermod -aG docker "$USER" || true
            sudo systemctl enable docker --quiet 2>/dev/null || true
            sudo systemctl start  docker 2>/dev/null || true
            DOCKER_INSTALLED=true
        else
            warn "Auto-install not available for your package manager."
            warn "Install Docker manually:  https://docs.docker.com/engine/install/"
        fi
        [[ "$DOCKER_INSTALLED" == "true" ]] && success "Docker installed"
    else
        info "Skipping Docker — system will run in local mode (no Docker required)."
    fi
fi

# ─── 4. Python ───────────────────────────────────────────────────────────────
header "Step 4 / 6  —  Python"

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
            # Re-run the apt update guard in case we skipped it (shouldn't happen but safe)
            sudo apt-get update -qq &>/dev/null || true
            ;;
        dnf)
            sudo dnf install -y python3.11 python3-pip
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

# Ensure python3-venv and pip are available.
# Ubuntu/Debian split these into separate packages (python3.X-venv, python3-pip).
if [[ "$PKG_MANAGER" == "apt" ]]; then
    PYVER=$("$PYTHON" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PKG_VENV="python${PYVER}-venv"
    info "Ensuring Python venv + pip packages (${PKG_VENV}, python3-pip)..."
    sudo apt-get install -y "$PKG_VENV" python3-pip &>/dev/null || true
    success "Python venv support ready"
elif [[ "$PKG_MANAGER" == "dnf" ]]; then
    sudo dnf install -y python3-pip &>/dev/null || true
elif [[ "$PKG_MANAGER" == "pacman" ]]; then
    sudo pacman -S --noconfirm python-pip &>/dev/null || true
fi

# ─── 4. Ollama ───────────────────────────────────────────────────────────────
header "Step 5 / 6  —  Ollama (local AI engine)"

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

# Detect total RAM (integer GB)
RAM_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || echo "0")
RAM_GB=$(( RAM_KB / 1024 / 1024 ))
success "Detected ${RAM_GB} GB RAM"

# Detect VRAM if GPU mode is active
VRAM_GB=0
if [[ "$USE_GPU" == "true" ]] && command -v nvidia-smi &>/dev/null; then
    VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0")
    VRAM_GB=$(( ${VRAM_MIB:-0} / 1024 ))
    success "Detected ${VRAM_GB} GB GPU VRAM"
fi

# Pick the best default model — VRAM drives the suggestion in GPU mode, RAM in CPU mode
if [[ "$USE_GPU" == "true" && $VRAM_GB -gt 0 ]]; then
    _CAPACITY_LABEL="${VRAM_GB} GB VRAM"
    if   [[ $VRAM_GB -ge 18 ]]; then
        _DEFAULT_MODEL_NUM=7; _SUGGEST_MODEL="qwen3:14b"
        _SUGGEST_NOTE="(best quality for legal research — 9 GB, your ${VRAM_GB} GB VRAM handles it)"
    elif [[ $VRAM_GB -ge 8 ]]; then
        _DEFAULT_MODEL_NUM=6; _SUGGEST_MODEL="qwen3:8b"
        _SUGGEST_NOTE="(strong reasoning — 5 GB, fits your ${VRAM_GB} GB VRAM)"
    elif [[ $VRAM_GB -ge 4 ]]; then
        _DEFAULT_MODEL_NUM=1; _SUGGEST_MODEL="qwen3:4b"
        _SUGGEST_NOTE="(best fit for ${VRAM_GB} GB VRAM — thinking mode, 256K ctx)"
    else
        _DEFAULT_MODEL_NUM=5; _SUGGEST_MODEL="qwen3:1.7b"
        _SUGGEST_NOTE="(recommended for ${VRAM_GB} GB VRAM — ultra-light Qwen3 with thinking)"
    fi
else
    # CPU mode — model must fit entirely in system RAM
    _CAPACITY_LABEL="${RAM_GB} GB RAM"
    if   [[ $RAM_GB -ge 20 ]]; then
        _DEFAULT_MODEL_NUM=7; _SUGGEST_MODEL="qwen3:14b"
        _SUGGEST_NOTE="(best quality for legal research — 9 GB, your ${RAM_GB} GB handles it)"
    elif [[ $RAM_GB -ge 12 ]]; then
        _DEFAULT_MODEL_NUM=6; _SUGGEST_MODEL="qwen3:8b"
        _SUGGEST_NOTE="(significantly better reasoning than 4b — 5 GB, fits your ${RAM_GB} GB)"
    elif [[ $RAM_GB -ge 7 ]]; then
        _DEFAULT_MODEL_NUM=1; _SUGGEST_MODEL="qwen3:4b"
        _SUGGEST_NOTE="(best fit for ${RAM_GB} GB — thinking mode, 256K ctx)"
    elif [[ $RAM_GB -ge 4 ]]; then
        _DEFAULT_MODEL_NUM=1; _SUGGEST_MODEL="qwen3:4b"
        _SUGGEST_NOTE="(fits ${RAM_GB} GB — thinking mode, 256K ctx)"
    else
        _DEFAULT_MODEL_NUM=5; _SUGGEST_MODEL="qwen3:1.7b"
        _SUGGEST_NOTE="(recommended for ${RAM_GB} GB — ultra-light Qwen3 with thinking)"
    fi
fi

echo ""
echo -e "  ${BOLD}Model recommendation for your machine (${_CAPACITY_LABEL}):${RESET}"
echo -e "  ${BOLD}${GREEN}  →  ${_SUGGEST_MODEL}  ${DIM}${_SUGGEST_NOTE}${RESET}"
echo ""
echo -e "  All options (Qwen3 family uses thinking mode for legal reasoning):"
echo ""
echo -e "  ${DIM}  1.  qwen3:4b     (2.5 GB — rivals models 18× its size · thinking mode · 256K ctx)${RESET}"
echo -e "  ${DIM}  2.  phi4-mini     (2.3 GB — Microsoft Phi-4 Mini · top reasoning · multilingual)${RESET}"
echo -e "  ${DIM}  3.  gemma3:4b     (2.5 GB — Google Gemma 3 · 128K ctx · 35M+ downloads)${RESET}"
echo -e "  ${DIM}  4.  llama3.2:3b   (2.0 GB — Meta · lightweight classic · good all-round)${RESET}"
echo -e "  ${DIM}  5.  qwen3:1.7b    (1.4 GB — ultra-light · same Qwen3 quality in tiny size)${RESET}"
echo -e "  ${DIM}  6.  qwen3:8b      (5.0 GB — noticeably stronger reasoning · needs 8 GB VRAM or 12 GB RAM)${RESET}"
echo -e "  ${DIM}  7.  qwen3:14b     (9.0 GB — best open legal reasoning model · needs 16 GB VRAM or 20 GB RAM)${RESET}"
echo -e "  ${DIM}  8.  Custom — type any Ollama model name${RESET}"
if [[ "$USE_GPU" == "true" && $VRAM_GB -gt 0 ]]; then
    echo ""
    echo -e "  ${DIM}Tip: GPU mode active — suggestion based on ${VRAM_GB} GB VRAM (larger models run fully on GPU).${RESET}"
elif [[ -n "$GPU_INFO" && "$NVIDIA_NEEDS_DRIVER" == "false" ]]; then
    echo ""
    echo -e "  ${DIM}Tip: Running in CPU mode — suggestion based on ${RAM_GB} GB system RAM.${RESET}"
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

info "Pulling model: ${BOLD}${MODEL}${RESET}  (downloads once, may take several minutes)..."
if ollama pull "$MODEL" 2>&1; then
    success "Model '${MODEL}' is ready"
else
    warn "Could not pull '${MODEL}'. Check your internet connection."
    warn "You can pull it later:  ollama pull ${MODEL}"
fi

# ─── 5. Python deps ──────────────────────────────────────────────────────────
header "Step 6 / 6  —  Python dependencies & configuration"

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
    # Retry without psycopg2-binary — it needs PostgreSQL dev headers (optional package)
    warn "Retrying without optional PostgreSQL driver (psycopg2-binary)..."
    TEMP_REQ="$VENV_DIR/requirements_no_psql.txt"
    grep -v "^psycopg2" "$SCRIPT_DIR/requirements_local.txt" > "$TEMP_REQ"
    if ! "$VENV_PY" -m pip install -r "$TEMP_REQ" --progress-bar=off; then
        rm -f "$TEMP_REQ"
        error "Python package installation failed — see errors above."
        error "Common fix:  sudo apt-get install build-essential python3-dev"
        error "Then re-run: ./install_linux.sh"
        exit 1
    fi
    rm -f "$TEMP_REQ"
    warn "psycopg2-binary skipped — PostgreSQL features disabled."
    warn "To enable later:  sudo apt-get install libpq-dev && pip install psycopg2-binary"
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
    print('  \u2713  Embedding model ready')
except Exception as err:
    print('  \u26a0  Will download on first use: ' + str(err), file=sys.stderr)
PYEOF

# ─── Write .env.local ────────────────────────────────────────────────────────
if [[ "$USE_GPU" == "true" ]]; then
    _EMB_DEVICE="cuda"
    _OLLAMA_NUM_GPU="999"
else
    _EMB_DEVICE="cpu"
    _OLLAMA_NUM_GPU="0"
fi

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

# GPU / CPU mode
# EMBEDDING_DEVICE: device for the embedding model (cpu | cuda)
# OLLAMA_NUM_GPU:   0 = CPU only, 999 = use all GPUs (Ollama default when GPU is available)
EMBEDDING_DEVICE=${_EMB_DEVICE}
OLLAMA_NUM_GPU=${_OLLAMA_NUM_GPU}

# Web interface port
APP_PORT=8080
ENVEOF

success "Configuration written to .env.local"

# ─── Ensure start.sh is executable ───────────────────────────────────────────
chmod +x "$SCRIPT_DIR/start.sh"
success "start.sh is ready"

# Make install script executable (in case it wasn't)
chmod +x "$SCRIPT_DIR/install_linux.sh"

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
