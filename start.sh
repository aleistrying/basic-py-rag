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
