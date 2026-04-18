#!/usr/bin/env python3
"""
RAG CLI - Beginner-friendly interactive command line tool
Reuses the same core functions as the web app (DRY).

Usage:
    python cli.py
    python cli.py --check     (status check only)
"""
import os
import sys
import subprocess
import argparse
import textwrap
import platform
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env.local if present (local/macOS mode set by install_mac.sh)
# ---------------------------------------------------------------------------
_ENV_LOCAL = Path(__file__).parent / ".env.local"
if _ENV_LOCAL.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_LOCAL, override=False)
    except ImportError:
        # dotenv not installed — parse manually (key=value, no spaces)
        for _line in _ENV_LOCAL.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Path setup – must happen before any local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional dependency imports (handled gracefully for noobs)
# ---------------------------------------------------------------------------
try:
    import questionary
    from questionary import Style as QStyle
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.spinner import Spinner
    from rich.live import Live
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# ---------------------------------------------------------------------------
# Rich helpers (fall back to plain print if Rich not installed)
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    if HAS_RICH:
        console.print(Panel(Text(title, justify="center",
                      style="bold cyan"), box=box.DOUBLE_EDGE))
    else:
        print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}\n")


def print_success(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold green]✓[/bold green] {msg}")
    else:
        print(f"[OK] {msg}")


def print_error(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold red]✗[/bold red] {msg}")
    else:
        print(f"[ERROR] {msg}")


def print_warning(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold yellow]⚠[/bold yellow]  {msg}")
    else:
        print(f"[WARN] {msg}")


def print_info(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[cyan]{msg}[/cyan]")
    else:
        print(msg)


def print_separator() -> None:
    if HAS_RICH:
        console.rule(style="dim")
    else:
        print("-" * 60)


def ask_input(prompt_text: str) -> str:
    """Single-line text input, using Rich if available."""
    if HAS_RICH:
        return Prompt.ask(f"[bold]{prompt_text}[/bold]")
    return input(f"{prompt_text}: ")


def ask_confirm(prompt_text: str, default: bool = True) -> bool:
    """Yes/No confirmation, using Rich if available."""
    if HAS_RICH:
        return Confirm.ask(f"[bold]{prompt_text}[/bold]", default=default)
    answer = input(
        f"{prompt_text} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if answer == "":
        return default
    return answer in ("y", "yes")


def _missing_dep_warning(package: str) -> None:
    print_warning(f"'{package}' is not installed. Run:  pip install {package}")


# ---------------------------------------------------------------------------
# Dependency check helpers
# ---------------------------------------------------------------------------

def _check_cli_deps() -> bool:
    """Warn the user if interactive menu dependencies are missing."""
    ok = True
    if not HAS_QUESTIONARY:
        _missing_dep_warning("questionary")
        ok = False
    if not HAS_RICH:
        _missing_dep_warning("rich")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Core app imports (lazy so the CLI is still usable even if some are missing)
# ---------------------------------------------------------------------------

def _import_rag():
    """Import RAG functions. Returns (search_knowledge_base, generate_llm_answer) or (None, None)."""
    try:
        from app.rag import search_knowledge_base, generate_llm_answer
        return search_knowledge_base, generate_llm_answer
    except Exception as exc:
        print_error(f"Could not import RAG module: {exc}")
        return None, None


def _import_ollama_utils():
    """Import Ollama utilities. Returns (check_health, generate_with_retry) or (None, None)."""
    try:
        from app.ollama_utils import check_ollama_health, ollama_generate_with_retry
        return check_ollama_health, ollama_generate_with_retry
    except Exception as exc:
        print_error(f"Could not import Ollama utils: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# STATUS CHECK
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 5):
    """stdlib-only HTTP GET. Returns (ok, response_text). No deps required."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return True, resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def _tcp_ping(host: str, port: int, timeout: int = 3) -> tuple:
    """Check if a TCP port is open. Returns (ok, info_str)."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} reachable"
    except OSError as exc:
        return False, str(exc)


def flow_status_check() -> None:
    print_header("System Status Check")

    # Determine hosts (env vars mirror docker-compose service names)
    ollama_host = os.getenv(
        "OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")

    # --- Docker containers (nice-to-have, skip gracefully) ---
    docker_rows = []
    try:
        result = subprocess.run(
            ["docker", "ps", "--format",
                "{{.Names}}\t{{.Status}}\t{{.Image}}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = line.split("\t")
                docker_rows.append(parts)
    except Exception:
        pass

    # --- Ollama – hit its REST API (no Python package needed) ---
    ollama_ok = False
    ollama_info = ""
    with _spinner("Checking Ollama..."):
        ok, body = _http_get(f"{ollama_host}/api/tags", timeout=5)
    if ok:
        import json as _json
        try:
            data = _json.loads(body)
            models = [m.get("name", "?") for m in data.get("models", [])]
            ollama_ok = True
            ollama_info = f"{len(models)} model(s): {', '.join(models) or 'none pulled yet'}"
        except Exception:
            ollama_ok = True
            ollama_info = "reachable (could not parse model list)"
    else:
        ollama_info = ollama_host + " – " + body

    # --- Qdrant – try its collections REST endpoint first, TCP as fallback ---
    qdrant_ok = False
    qdrant_info = ""
    with _spinner("Checking Qdrant..."):
        ok, body = _http_get(
            f"http://{qdrant_host}:6333/collections", timeout=5)
    if ok:
        import json as _json
        try:
            data = _json.loads(body)
            cols = [c.get("name", "?")
                    for c in data.get("result", {}).get("collections", [])]
            qdrant_ok = True
            qdrant_info = f"{len(cols)} collection(s): {', '.join(cols) or 'none'}"
        except Exception:
            qdrant_ok = True
            qdrant_info = "reachable"
    else:
        # TCP fallback
        ok2, info2 = _tcp_ping(qdrant_host, 6333)
        qdrant_ok = ok2
        qdrant_info = info2 if ok2 else f"{qdrant_host}:6333 unreachable – {body}"

    # --- PostgreSQL – TCP check (no psycopg2 required) ---
    pg_ok = False
    pg_info = ""
    with _spinner("Checking PostgreSQL..."):
        pg_ok, pg_info = _tcp_ping(pg_host, 5432)
    if not pg_ok:
        pg_info = f"{pg_host}:5432 unreachable"
    else:
        # If psycopg2 is available locally, also list tables
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname="vectordb", user="pguser", password="pgpass",
                host=pg_host, port=5432, connect_timeout=3
            )
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public';"
                )
                tables = [row[0] for row in cur.fetchall()]
            conn.close()
            pg_info = f"tables: {', '.join(tables) or 'none'}"
        except ImportError:
            pg_info = "port open (psycopg2 not in local venv – running inside Docker is fine)"
        except Exception as exc:
            pg_info = f"port open, auth error: {exc}"

    # --- Render table ---
    if HAS_RICH:
        if docker_rows:
            dt = Table(title="Docker Containers", box=box.ROUNDED)
            dt.add_column("Name", style="bold")
            dt.add_column("Status")
            dt.add_column("Image")
            for row in docker_rows:
                name = row[0] if len(row) > 0 else "?"
                status = row[1] if len(row) > 1 else "?"
                image = row[2] if len(row) > 2 else "?"
                colour = "green" if "Up" in status else "red"
                dt.add_row(name, f"[{colour}]{status}[/{colour}]", image)
            console.print(dt)

        table = Table(title="Services", box=box.ROUNDED)
        table.add_column("Service", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        def _icon(ok: bool) -> str:
            return "[bold green]ONLINE[/bold green]" if ok else "[bold red]OFFLINE[/bold red]"

        table.add_row("Ollama (LLM)", _icon(ollama_ok), ollama_info)
        table.add_row("Qdrant (vector DB)", _icon(qdrant_ok), qdrant_info)
        table.add_row("PostgreSQL (pgvector)", _icon(pg_ok), pg_info)

        console.print(table)
    else:
        print(f"Ollama  : {'OK' if ollama_ok else 'FAIL'} – {ollama_info}")
        print(f"Qdrant  : {'OK' if qdrant_ok else 'FAIL'} – {qdrant_info}")
        print(f"Postgres: {'OK' if pg_ok else 'FAIL'} – {pg_info}")

    _press_enter()


# ---------------------------------------------------------------------------
# ASK RAG
# ---------------------------------------------------------------------------

def flow_ask_rag() -> None:
    print_header("Ask the Knowledge Base (RAG)")
    print_info("This searches your documents and uses the AI to answer.\n")

    search_knowledge_base, generate_llm_answer = _import_rag()
    if search_knowledge_base is None:
        _press_enter()
        return

    query = ask_input("Your question")
    if not query.strip():
        print_warning("Empty query – cancelling.")
        _press_enter()
        return

    # Backend choice
    backend = _select_backend()
    # AI answer or raw search
    want_ai = ask_confirm(
        "Generate an AI answer? (requires Ollama to be running)", default=True)

    if want_ai:
        model = _select_model()
        with _spinner("Thinking..."):
            try:
                result = generate_llm_answer(
                    query=query, backend=backend, model=model)
            except Exception as exc:
                print_error(f"Error: {exc}")
                _press_enter()
                return
        _render_rag_answer(result)
    else:
        with _spinner("Searching..."):
            try:
                result = search_knowledge_base(query=query, backend=backend)
            except Exception as exc:
                print_error(f"Error: {exc}")
                _press_enter()
                return
        _render_search_results(result)

    _press_enter()


def _render_rag_answer(result: dict) -> None:
    ai_response = result.get("ai_response") or result.get("answer", "")
    query = result.get("query", "")
    model = result.get("model", "N/A")
    sources = result.get("sources") or result.get("results", [])

    if HAS_RICH:
        console.print(Panel(
            Text(query, style="italic"),
            title="[bold]Your Question[/bold]",
            border_style="blue"
        ))
        console.print(Panel(
            ai_response,
            title=f"[bold green]AI Answer[/bold green]  [dim](model: {model})[/dim]",
            border_style="green"
        ))
        if sources:
            _render_sources_table(sources)
    else:
        print(f"\nQuestion: {query}")
        print(f"\nAnswer ({model}):\n{ai_response}")
        if sources:
            print("\nSources:")
            for i, s in enumerate(sources, 1):
                doc = s.get("document") or s.get("path", "Unknown")
                print(f"  {i}. {doc}")

    _offer_open_document(sources)


def _render_search_results(result: dict) -> None:
    results = result.get("results", [])
    query = result.get("query", "")

    if HAS_RICH:
        console.print(f"\n[bold]Query:[/bold] {query}")
        console.print(f"[dim]{len(results)} result(s) from {result.get('backend', '?')} "
                      f"in {result.get('backend_info', {}).get('search_time_ms', '?')} ms[/dim]\n")
        _render_sources_table(results)
    else:
        print(f"\nQuery: {query} – {len(results)} result(s)")
        for i, r in enumerate(results, 1):
            print(
                f"\n  [{i}] {r.get('document', 'Unknown')}  (score: {r.get('similarity', '?')})")
            print(f"       {r.get('preview', '')}")

    _offer_open_document(results)


def _render_sources_table(sources: list) -> None:
    if not HAS_RICH:
        return
    table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Document")
    table.add_column("Score")
    table.add_column("Preview")
    for i, s in enumerate(sources, 1):
        doc_name = s.get("document") or s.get("reference", "Unknown")
        raw_path = s.get("path") or s.get("source_path") or ""
        # Make the document name a clickable file link if we have the path
        if raw_path:
            doc_cell = _file_link(raw_path, label=doc_name)
        else:
            doc_cell = f"[bold]{doc_name}[/bold]"
        score = str(s.get("similarity") or s.get("score") or "N/A")
        preview = s.get("preview") or s.get("content", "")
        preview = textwrap.shorten(preview or "", width=70, placeholder="...")
        table.add_row(str(i), doc_cell, score, preview)
    console.print(table)


# ---------------------------------------------------------------------------
# ASK LLM DIRECTLY (no RAG context)
# ---------------------------------------------------------------------------

def flow_ask_llm() -> None:
    print_header("Ask the AI Directly")
    print_info(
        "This sends your question straight to the LLM – no document search.\n")

    _, ollama_generate_with_retry = _import_ollama_utils()
    if ollama_generate_with_retry is None:
        _press_enter()
        return

    query = ask_input("Your question")
    if not query.strip():
        print_warning("Empty query – cancelling.")
        _press_enter()
        return

    model = _select_model()

    with _spinner("Generating answer..."):
        try:
            response = ollama_generate_with_retry(model=model, prompt=query)
        except Exception as exc:
            print_error(f"Error: {exc}")
            _press_enter()
            return

    answer = response.get("response", "")
    model_used = response.get("model", model)

    if HAS_RICH:
        console.print(Panel(
            Text(query, style="italic"),
            title="[bold]Your Question[/bold]",
            border_style="blue"
        ))
        console.print(Panel(
            answer,
            title=f"[bold green]Answer[/bold green]  [dim](model: {model_used})[/dim]",
            border_style="green"
        ))
    else:
        print(f"\nQuestion: {query}")
        print(f"\nAnswer ({model_used}):\n{answer}")

    _press_enter()


# ---------------------------------------------------------------------------
# BROWSE DOCUMENTS
# ---------------------------------------------------------------------------

def flow_browse_documents() -> None:
    print_header("Browse Documents")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    clean_dir = PROJECT_ROOT / "data" / "clean"

    raw_files = sorted([f for f in raw_dir.glob("**/*") if f.is_file()])
    clean_files = sorted([f for f in clean_dir.glob("**/*") if f.is_file()])

    if not raw_files and not clean_files:
        print_warning("No documents found in data/raw/ or data/clean/")
        _press_enter()
        return

    if HAS_RICH:
        table = Table(title="Available Documents", box=box.ROUNDED)
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="bold")
        table.add_column("Folder")
        table.add_column("Size")

        all_files = [(f, "raw") for f in raw_files] + [(f, "clean")
                                                       for f in clean_files]
        for i, (f, folder) in enumerate(all_files, 1):
            size = f"{f.stat().st_size // 1024} KB"
            table.add_row(str(i), f.name, folder, size)
        console.print(table)
    else:
        print("\nRaw documents:")
        for i, f in enumerate(raw_files, 1):
            print(f"  {i}. {f.name}")
        print("\nProcessed (clean) chunks:")
        for i, f in enumerate(clean_files, 1):
            print(f"  {i}. {f.name}")

    all_files_list = [(f, "raw") for f in raw_files] + \
        [(f, "clean") for f in clean_files]
    if not all_files_list:
        _press_enter()
        return

    # Let user pick a file to preview
    if ask_confirm("\nPreview a file?", default=False):
        try:
            idx_str = ask_input(f"Enter file number (1-{len(all_files_list)})")
            idx = int(idx_str) - 1
            if 0 <= idx < len(all_files_list):
                _preview_file(all_files_list[idx][0])
            else:
                print_warning("Invalid number.")
        except ValueError:
            print_warning("Please enter a valid number.")

    _press_enter()


def _preview_file(file_path: Path) -> None:
    suffix = file_path.suffix.lower()
    try:
        if suffix in (".txt", ".md", ".jsonl", ".json"):
            lines = file_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
            preview = "\n".join(lines[:50])
            if HAS_RICH:
                console.print(Panel(
                    preview,
                    title=f"[bold]{file_path.name}[/bold] [dim](first 50 lines)[/dim]",
                    border_style="cyan"
                ))
            else:
                print(f"\n--- {file_path.name} (first 50 lines) ---")
                print(preview)
        else:
            print_info(f"Preview not supported for {suffix} files.")
    except Exception as exc:
        print_error(f"Could not read file: {exc}")


def open_document(path_str: str) -> None:
    """Open a document with the OS default application."""
    # Resolve to absolute path
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    # Strip leading ./ prefixes that come from stored paths
    try:
        path = path.resolve()
    except Exception:
        pass

    if not path.exists():
        print_warning(f"File not found: {path}")
        return

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", str(path)], check=True)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(path)], check=True)
        elif system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            print_warning(f"Don't know how to open files on {system}.")
            print_info(f"File location: {path}")
            return
        print_success(f"Opened: {path.name}")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print_error(f"Could not open file: {exc}")
        print_info(f"File location: {path}")


def _file_link(path_str: str, label: str | None = None) -> str:
    """
    Return a Rich-formatted [link=file://...] if Rich is available,
    otherwise just the label.  Clickable in iTerm2, macOS Terminal ≥14,
    Warp, VS Code terminal, and most modern terminals.
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    abs_uri = path.as_uri()     # → file:///absolute/path
    display = label or path.name
    if HAS_RICH:
        return f"[link={abs_uri}][bold cyan]{display}[/bold cyan][/link]"
    return display


def _offer_open_document(results: list) -> None:
    """After showing results, let the user open a source document."""
    if not results:
        return
    # Collect unique source paths
    paths = []
    seen = set()
    for r in results:
        p = r.get("path") or r.get("source_path") or ""
        if p and p not in seen:
            seen.add(p)
            paths.append(p)
    if not paths:
        return

    if not ask_confirm("\nOpen a source document?", default=False):
        return

    if HAS_QUESTIONARY and len(paths) > 1:
        choices = [Path(p).name for p in paths] + ["Cancel"]
        chosen_name = questionary.select(
            "Which document?", choices=choices, style=CUSTOM_STYLE
        ).ask()
        if chosen_name and chosen_name != "Cancel":
            for p in paths:
                if Path(p).name == chosen_name:
                    open_document(p)
                    break
    else:
        print_info("Available documents:")
        for i, p in enumerate(paths, 1):
            print_info(f"  {i}. {Path(p).name}")
        raw = ask_input(f"Enter number (1-{len(paths)}, or 0 to cancel)")
        try:
            idx = int(raw)
            if 1 <= idx <= len(paths):
                open_document(paths[idx - 1])
        except (ValueError, IndexError):
            pass


# ---------------------------------------------------------------------------
# INGEST DOCUMENTS
# ---------------------------------------------------------------------------

def flow_ingest_documents() -> None:
    print_header("Ingest Documents to Vector Database")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_files = list(raw_dir.glob("**/*"))
    raw_files = [f for f in raw_files if f.is_file()]

    if not raw_files:
        print_warning("No files found in data/raw/ – nothing to ingest.")
        print_info(
            "Copy your PDF or text files to data/raw/ then run this again.")
        _press_enter()
        return

    print_info(f"Found {len(raw_files)} file(s) in data/raw/:")
    for f in raw_files:
        print_info(f"  • {f.name}")

    if not ask_confirm("\nProceed with ingestion? This may take several minutes.", default=True):
        print_info("Ingestion cancelled.")
        _press_enter()
        return

    # Extra options
    clear_first = ask_confirm(
        "Clear existing database collections before ingesting?", default=False
    )
    memory_safe = ask_confirm(
        "Use memory-safe mode? (slower but won't crash on low RAM)", default=True
    )

    # Build command
    cmd = [sys.executable, "scripts/main_pipeline.py"]
    if clear_first:
        cmd.append("--clear")
    if memory_safe:
        cmd.append("--memory-safe")

    print_info(f"\nRunning: {' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        # Stream output line by line so the user can see progress
        for line in process.stdout:
            line = line.rstrip()
            if HAS_RICH:
                # Colour-code based on log level keywords
                if "ERROR" in line or "error" in line:
                    console.print(f"[red]{line}[/red]")
                elif "WARNING" in line or "warning" in line or "WARN" in line:
                    console.print(f"[yellow]{line}[/yellow]")
                elif "✓" in line or "completed" in line.lower() or "success" in line.lower():
                    console.print(f"[green]{line}[/green]")
                else:
                    console.print(line)
            else:
                print(line)

        process.wait()
        if process.returncode == 0:
            print_success("Ingestion pipeline completed successfully!")
        else:
            print_error(f"Pipeline exited with code {process.returncode}")
    except (FileNotFoundError, OSError) as exc:
        print_error(f"Could not run pipeline: {exc}")

    _press_enter()


# ---------------------------------------------------------------------------
# INTERACTIVE MENU HELPERS
# ---------------------------------------------------------------------------

MENU_OPTIONS = [
    ("Ask the knowledge base a question (RAG + AI)", "rag"),
    ("Ask the AI a question directly (no documents)", "llm"),
    ("Browse / read documents", "browse"),
    ("Ingest documents into the vector database", "ingest"),
    ("Check system status (Ollama, Qdrant, PostgreSQL)", "status"),
    ("Exit", "exit"),
]

CUSTOM_STYLE = None
if HAS_QUESTIONARY:
    CUSTOM_STYLE = QStyle([
        ("qmark",        "fg:#00bcd4 bold"),
        ("question",     "bold"),
        ("answer",       "fg:#00e676 bold"),
        ("pointer",      "fg:#00bcd4 bold"),
        ("highlighted",  "fg:#00e676 bold"),
        ("selected",     "fg:#cccccc"),
        ("separator",    "fg:#555555"),
        ("instruction",  "fg:#888888"),
        ("text",         ""),
        ("disabled",     "fg:#555555 italic"),
    ])


def _select_backend() -> str:
    if HAS_QUESTIONARY:
        return questionary.select(
            "Which vector database backend?",
            choices=["qdrant", "pgvector"],
            default="qdrant",
            style=CUSTOM_STYLE,
        ).ask() or "qdrant"
    return ask_input("Backend (qdrant / pgvector) [qdrant]") or "qdrant"


def _select_model() -> str:
    models = ["gemma2:2b", "phi3:mini", "qwen2.5:3b"]
    if HAS_QUESTIONARY:
        return questionary.select(
            "Which AI model?",
            choices=models,
            default=models[0],
            style=CUSTOM_STYLE,
        ).ask() or models[0]
    return ask_input(f"Model [{models[0]}]") or models[0]


def _spinner(label: str):
    """Context manager – animated spinner if Rich available, otherwise no-op."""
    if HAS_RICH:
        return console.status(f"[cyan]{label}[/cyan]")
    else:
        class _Noop:
            def __enter__(self): print(label); return self
            def __exit__(self, *_): pass
        return _Noop()


def _press_enter() -> None:
    if HAS_RICH:
        console.print("\n[dim]Press Enter to return to menu...[/dim]", end="")
    else:
        print("\nPress Enter to continue...", end="")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# WELCOME BANNER
# ---------------------------------------------------------------------------

BANNER = r"""
  ____      _         ____  ____    _    ____
 |  _ \  __| |_   _  |  _ \/ ___|  / \  / ___|
 | |_) |/ _` | | | | | |_) | |    / _ \| |  _
 |  _ <| (_| | |_| | |  _ <| |___ / ___ \ |_| |
 |_| \_\\__,_|\__, | |_| \_\\____/_/   \_\____|
               |___/
"""


def _print_welcome() -> None:
    if HAS_RICH:
        console.print(Text(BANNER, style="bold cyan"))
        console.print(
            Panel(
                "[bold]Welcome![/bold]  This tool lets you interact with your RAG system "
                "without knowing any code.\n"
                "[dim]Use the arrow keys to navigate, Enter to select.[/dim]",
                border_style="cyan",
            )
        )
    else:
        print(BANNER)
        print("Welcome! Use numbers to navigate the menu.\n")


# ---------------------------------------------------------------------------
# QUESTIONARY MAIN LOOP
# ---------------------------------------------------------------------------

def _run_questionary_loop() -> None:
    _print_welcome()
    while True:
        if HAS_RICH:
            console.print()
        choice = questionary.select(
            "What would you like to do?",
            choices=[label for label, _ in MENU_OPTIONS],
            style=CUSTOM_STYLE,
        ).ask()

        if choice is None:  # User pressed Ctrl+C
            break

        action = next(
            (v for label, v in MENU_OPTIONS if label == choice), "exit")

        if action == "exit":
            if HAS_RICH:
                console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
            else:
                print("\nGoodbye!")
            break
        elif action == "rag":
            flow_ask_rag()
        elif action == "llm":
            flow_ask_llm()
        elif action == "browse":
            flow_browse_documents()
        elif action == "ingest":
            flow_ingest_documents()
        elif action == "status":
            flow_status_check()


# ---------------------------------------------------------------------------
# PLAIN-TEXT FALLBACK MENU (when questionary is not installed)
# ---------------------------------------------------------------------------

def _run_plain_loop() -> None:
    _print_welcome()
    while True:
        print("\nMain Menu:")
        for i, (label, _) in enumerate(MENU_OPTIONS, 1):
            print(f"  {i}. {label}")
        print()
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw) - 1
            if idx < 0 or idx >= len(MENU_OPTIONS):
                raise ValueError
        except ValueError:
            print("Please enter a valid number.")
            continue

        _, action = MENU_OPTIONS[idx]

        if action == "exit":
            print("\nGoodbye!\n")
            break
        elif action == "rag":
            flow_ask_rag()
        elif action == "llm":
            flow_ask_llm()
        elif action == "browse":
            flow_browse_documents()
        elif action == "ingest":
            flow_ingest_documents()
        elif action == "status":
            flow_status_check()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG CLI – beginner-friendly interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python cli.py          Run the interactive menu
              python cli.py --check  Just run the status check and exit
        """),
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Run system status check only (non-interactive)"
    )
    args = parser.parse_args()

    if args.check:
        flow_status_check()
        return

    if not _check_cli_deps():
        print()
        print_info(
            "Tip: install full dependencies with:  pip install questionary rich")
        print_info("Falling back to plain text menu...\n")

    try:
        if HAS_QUESTIONARY:
            _run_questionary_loop()
        else:
            _run_plain_loop()
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print("\n\n[bold cyan]Goodbye![/bold cyan]\n")
        else:
            print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
