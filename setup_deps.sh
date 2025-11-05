#!/bin/bash
# Quick setup script for clean ingest pipeline dependencies

echo "🚀 Setting up Clean Ingest Pipeline Dependencies"
echo "================================================"

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found - please install python3-pip first"
    exit 1
fi

echo "📦 Installing required packages..."

# Core packages for the pipeline
pip3 install --user numpy sentence-transformers transformers torch

# Optional packages for backends
echo "🗄️  Installing backend packages..."
pip3 install --user qdrant-client psycopg2-binary

# PDF processing packages
echo "📄 Installing PDF processing packages..."
pip3 install --user pdfplumber PyMuPDF PyPDF2

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "🎯 Next steps:"
echo "   1. Check setup: python3 scripts/ingest_all.py --check"
echo "   2. Test pipeline: python3 test_clean_pipeline.py"  
echo "   3. Run pipeline: python3 scripts/ingest_all.py"
echo ""