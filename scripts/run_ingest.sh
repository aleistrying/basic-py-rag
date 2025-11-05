#!/bin/bash
# Wrapper script to run combined ingestion with virtual environment

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
source ../.venv/bin/activate

# Run the ingestion script
python ingest_all.py "$@"
