#!/bin/bash
# Quick setup script for clean ingest pipeline dependencies

set -e

echo "🚀 Setting up Clean Ingest Pipeline Dependencies"
echo "================================================"

MODE="ocr"
if [[ "${1:-}" == "--full" ]]; then
    MODE="full"
fi

# Pick installer priority: uv pip (if in uv project) -> pip -> pip3
INSTALLER=""
if command -v uv >/dev/null 2>&1 && [[ -d ".venv" ]]; then
    INSTALLER="uv pip"
elif command -v pip >/dev/null 2>&1; then
    INSTALLER="pip"
elif command -v pip3 >/dev/null 2>&1; then
    INSTALLER="pip3"
else
    echo "❌ pip not found. Create/activate a venv first (uv venv && source .venv/bin/activate)."
    exit 1
fi

echo "🔧 Installer: ${INSTALLER}"
echo "📦 Mode: ${MODE}"

echo "📦 Installing required packages..."

# OCR/PDF processing packages (fast path, no torch)
echo "📄 Installing OCR + PDF packages..."
${INSTALLER} install pdfplumber PyMuPDF PyPDF2 pdf2image pytesseract pillow

if [[ "${MODE}" == "full" ]]; then
    # Embedding + backends (includes torch, large download)
    echo "🧠 Installing embedding + backend packages (this can take a while)..."
    ${INSTALLER} install numpy sentence-transformers transformers torch qdrant-client psycopg2-binary
fi

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "🎯 Next steps:"
echo "   1. Check OCR tools: tesseract --version && pdftoppm -v"
echo "   2. Put PDFs in data/raw"
echo "   3. OCR process: FORCE_OCR_ALL=1 python scripts/pdf_processing.py"
echo ""
echo "ℹ️  For full RAG dependencies later: ./setup_deps.sh --full"
echo ""