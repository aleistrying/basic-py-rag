# Ingestion Scripts

This folder contains scripts for ingesting documents into the vector databases.

## Quick Start

### Combined Ingestion (Recommended)

Ingest to both Qdrant and PostgreSQL at once:

```bash
# From project root
./scripts/run_ingest.sh

# Or with custom data folder
./scripts/run_ingest.sh ./path/to/documents
```

Or if you have the virtual environment activated:

```bash
python scripts/ingest_all.py
```

### Individual Ingestion

If you need to ingest to only one database:

```bash
# Qdrant only
python scripts/ingest_qdrant.py

# PostgreSQL only
python scripts/ingest_pgvector.py
```

## What the Scripts Do

### `ingest_all.py` - Combined Ingestion

1. **Resets both databases** - Drops and recreates collections/tables
2. **Processes documents** - Reads all files from `./data/raw` (or specified folder)
3. **Creates chunks** - Splits documents into token-aware chunks (220 tokens each)
4. **Generates embeddings** - Uses E5 multilingual model (768 dimensions)
5. **Extracts schedules** - Automatically detects and extracts course schedules from PDFs
6. **Ingests to both** - Uploads to Qdrant and PostgreSQL in parallel

**Progress reporting includes:**

- Step-by-step status messages
- Document processing progress
- Chunk creation count
- Embedding generation status
- Batch upload progress
- Final statistics (total chunks, time, throughput)

### Features

✅ **Automatic schedule extraction** - Detects course guide PDFs and extracts programming schedules
✅ **Database reset** - Ensures clean state on each run
✅ **Progress reporting** - Clear visual feedback on what's happening
✅ **Error handling** - Graceful failure with helpful error messages
✅ **Batch processing** - Efficient upload in 128-chunk batches
✅ **E5 embeddings** - Better Spanish language support
✅ **Metadata preservation** - Stores schedules and file paths

## Supported File Types

- `.txt` - Plain text
- `.pdf` - PDF documents (with schedule extraction)
- `.docx` - Word documents
- `.xlsx` - Excel spreadsheets
- `.pptx` - PowerPoint presentations
- `.html` / `.htm` - HTML files

## Configuration

Edit `ingest_all.py` to change:

- `QDRANT_COLLECTION` - Qdrant collection name (default: "docs_qdrant")
- `DIM` - Embedding dimensions (default: 768 for E5-base)
- `POSTGRES_CONFIG` - Database connection settings
- `BATCH` - Upload batch size (default: 128)

## Output Example

```
======================================================================
  🚀 COMBINED DATABASE INGESTION
======================================================================
Target databases: Qdrant + PostgreSQL (pgvector)
Source folder: ./data/raw

[1/4] 🔄 Resetting Qdrant database...
    ✅ Deleted existing collection: docs_qdrant
    ✅ Created fresh collection: docs_qdrant

[2/4] 🔄 Resetting PostgreSQL database...
    ✅ Dropped existing docs table
    ✅ Created fresh docs table with vector(768)
    ✅ Created cosine similarity index

[3/4] 📄 Processing documents from: ./data/raw
    ℹ️  Found 2 documents

    [1/2] Processing: curso_objetivos_clean.txt
        📊 Created 4 chunks
        🧮 Generating embeddings...
        ✅ Generated 4 embeddings

    [2/2] Processing: Guia de Curso 1CA217 2S2025 Sistemas BD Avanzadas v2.pdf
        📊 Created 4 chunks
        🧮 Generating embeddings...
        ✅ Generated 4 embeddings
        📅 Schedule data attached

    ✅ Total chunks processed: 8

[4/4] 💾 Ingesting to Qdrant...
    📦 Uploading batch 1/1... ✅
    ✅ Ingested 8 chunks to Qdrant

💾 Ingesting to PostgreSQL...
    ✅ Ingested 8 chunks to PostgreSQL

======================================================================
  ✅ INGESTION COMPLETE
======================================================================
  📊 Total chunks: 8
  ⏱️  Time elapsed: 7.41 seconds
  📈 Chunks per second: 1.08

  Databases updated:
    • Qdrant collection: docs_qdrant
    • PostgreSQL table: docs
======================================================================
```

## Troubleshooting

**Error: Connection refused**

- Make sure Docker containers are running: `docker compose up -d`

**Error: Module not found**

- Activate virtual environment: `source .venv/bin/activate`
- Or use the wrapper script: `./scripts/run_ingest.sh`

**No documents found**

- Check that files exist in `./data/raw/`
- Or specify a different folder: `./scripts/run_ingest.sh /path/to/docs`
