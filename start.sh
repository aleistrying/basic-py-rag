#!/usr/bin/env bash
# RAG System — one-command launcher
# Handles port conflicts, auto-installs, and auto-restarts automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
ENV_FILE="$SCRIPT_DIR/.env.local"

# ─── Load local overrides ─────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
APP_PORT="${APP_PORT:-8080}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
_APP_MARKER="$HOME/.local/share/research-studio/desktop_v1"

# ─── Auto-install if venv is missing ─────────────────────────────────────────
if [[ ! -f "$VENV_PY" ]]; then
    echo ""
    echo "  First run — installing dependencies automatically…"
    bash "$SCRIPT_DIR/install_linux.sh" || {
        echo "  ✗ Installation failed. Please run install_linux.sh manually."
        exit 1
    }
fi

# ─── Free the port (kill whatever is already using it) ───────────────────────
free_port() {
    local port="$1"
    local pids=""
    if command -v lsof &>/dev/null; then
        pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    elif command -v fuser &>/dev/null; then
        pids=$(fuser "${port}/tcp" 2>/dev/null | tr -s ' ' '\n' | grep -v '^$' || true)
    fi
    if [[ -n "$pids" ]]; then
        echo "  → Port $port busy — stopping previous server…"
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 3
        # Force-kill if still alive
        local remaining
        remaining=$(lsof -ti tcp:"$port" 2>/dev/null || true)
        if [[ -n "$remaining" ]]; then
            echo "$remaining" | xargs kill -KILL 2>/dev/null || true
            sleep 1
        fi
        echo "  ✓ Port $port is free"
    fi
}

# ─── Desktop shortcut installer (runs once automatically) ──────────────────────
_install_linux_desktop() {
    local apps_dir="$HOME/.local/share/applications"
    mkdir -p "$apps_dir"
    local term=""
    for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal tilix alacritty kitty xterm; do
        command -v "$t" &>/dev/null && { term="$t"; break; }
    done
    if [[ -z "$term" ]]; then
        echo "  ⚠  No terminal emulator found — app shortcut not created."; return 1
    fi
    local exec_line
    case "$term" in
        gnome-terminal) exec_line="gnome-terminal -- bash \"$SCRIPT_DIR/start.sh\"" ;;
        konsole)        exec_line="konsole -e bash \"$SCRIPT_DIR/start.sh\"" ;;
        *)              exec_line="$term -e bash \"$SCRIPT_DIR/start.sh\"" ;;
    esac
    cat > "$apps_dir/research-studio.desktop" << DESKTOPEOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Research Studio
GenericName=RAG Research Tool
Comment=AI document search — upload, index, and ask questions
Exec=$exec_line
Icon=applications-science
Terminal=false
Categories=Education;Science;Utility;
StartupNotify=true
DESKTOPEOF
    chmod +x "$apps_dir/research-studio.desktop"
    update-desktop-database "$apps_dir" 2>/dev/null || true
    if [[ -d "$HOME/Desktop" ]]; then
        cp "$apps_dir/research-studio.desktop" "$HOME/Desktop/research-studio.desktop"
        chmod +x "$HOME/Desktop/research-studio.desktop"
        gio set "$HOME/Desktop/research-studio.desktop" metadata::trusted true 2>/dev/null || true
        echo "  ✓ 'Research Studio' shortcut on Desktop + app launcher"
    else
        echo "  ✓ 'Research Studio' added to app launcher (search for it)"
    fi
}

_install_macos_app() {
    local app_dir="$HOME/Applications/Research Studio.app"
    mkdir -p "$app_dir/Contents/MacOS"
    cat > "$app_dir/Contents/MacOS/ResearchStudio" << MACBINEOF
#!/usr/bin/env bash
osascript -e 'tell application "Terminal" to activate' \\
          -e 'tell application "Terminal" to do script "cd $SCRIPT_DIR && ./start.sh"'
MACBINEOF
    chmod +x "$app_dir/Contents/MacOS/ResearchStudio"
    cat > "$app_dir/Contents/Info.plist" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>ResearchStudio</string>
  <key>CFBundleIdentifier</key><string>com.researchstudio.app</string>
  <key>CFBundleName</key><string>Research Studio</string>
  <key>CFBundleDisplayName</key><string>Research Studio</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>10.13</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLISTEOF
    if [[ -d "$HOME/Desktop" ]]; then
        printf '%s\n' "#!/usr/bin/env bash" "cd \"$SCRIPT_DIR\"" "./start.sh" \
            > "$HOME/Desktop/Research Studio.command"
        chmod +x "$HOME/Desktop/Research Studio.command"
        echo "  ✓ 'Research Studio' shortcut on Desktop"
    fi
    echo "  ✓ 'Research Studio.app' installed in ~/Applications"
    echo "    Drag it to your Dock for one-click access."
}

install_desktop_app_if_needed() {
    [[ -f "$_APP_MARKER" ]] && return
    local os; os="$(uname -s)"
    echo ""
    echo "  Installing desktop shortcut (one-time setup)…"
    if [[ "$os" == "Darwin" ]]; then
        _install_macos_app
    elif [[ "$os" == "Linux" ]]; then
        _install_linux_desktop
    fi
    mkdir -p "$(dirname "$_APP_MARKER")"
    echo "$SCRIPT_DIR" > "$_APP_MARKER"
    echo ""
}

install_desktop_app_if_needed

# ─── Ensure Ollama is running ─────────────────────────────────────────────────
if ! curl -s --max-time 2 "$OLLAMA_HOST/api/tags" &>/dev/null; then
    echo "  Starting Ollama…"
    ollama serve &>/tmp/ollama_rag.log &
    sleep 3
fi

# ─── Menu ─────────────────────────────────────────────────────────────────────
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

open_browser() {
    local url="$1"
    if command -v xdg-open &>/dev/null;        then xdg-open        "$url" &>/dev/null & return; fi
    if command -v sensible-browser &>/dev/null; then sensible-browser "$url" &>/dev/null & return; fi
    if command -v gnome-open &>/dev/null;       then gnome-open       "$url" &>/dev/null & return; fi
    if command -v open &>/dev/null;             then open             "$url" &>/dev/null & return; fi
}

# ─── Auto-restarting server runner ───────────────────────────────────────────
run_server() {
    local url="$1"
    shift  # remaining args passed to uvicorn

    cd "$SCRIPT_DIR"
    free_port "$APP_PORT"

    echo "  Opening $url"
    echo "  Press Ctrl+C to stop."
    echo ""

    # Open browser after 2 s without blocking
    (sleep 2 && open_browser "$url") &

    local restarts=0
    local _stopped=0
    trap '_stopped=1' INT TERM

    while [[ $_stopped -eq 0 ]]; do
        "$VENV_PY" -m uvicorn app.main:app \
            --host 0.0.0.0 --port "$APP_PORT" "$@" || true

        # Ctrl+C / SIGTERM → clean exit
        [[ $_stopped -eq 1 ]] && break

        restarts=$((restarts + 1))
        if [[ $restarts -ge $MAX_RESTARTS ]]; then
            echo ""
            echo "  ✗ Server crashed $MAX_RESTARTS times in a row."
            echo "    Check the error messages above, then run start.sh again."
            exit 1
        fi

        echo ""
        echo "  ⚠  Server stopped unexpectedly. Restarting… (attempt $restarts / $MAX_RESTARTS)"
        sleep 2
        free_port "$APP_PORT"
        sleep 1
    done

    echo ""
    echo "  Server stopped. Run ./start.sh to start again."
}

# ─── Launch ───────────────────────────────────────────────────────────────────
case "$CHOICE" in
    1)
        exec "$VENV_PY" "$SCRIPT_DIR/cli.py" "$@"
        ;;
    2)
        echo ""
        echo "  Starting Research UI…"
        run_server "http://localhost:${APP_PORT}/research"
        ;;
    3)
        echo ""
        echo "  Starting full web app…"
        run_server "http://localhost:${APP_PORT}" --reload
        ;;
    *)
        echo "  Invalid choice."
        exit 1
        ;;
esac
