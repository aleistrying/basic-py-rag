# Configuration for clean ingest pipeline
import os
from pathlib import Path

# Directory structure
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"

# Create directories if they don't exist
Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

# Embedding model: multilingual E5 for Spanish support
# Alternative: "Alibaba-NLP/gte-multilingual-base"
EMBED_MODEL = "intfloat/multilingual-e5-base"

# Chunking parameters (token-aware)
CHUNK_TOKENS = 350  # ~250-300 words
CHUNK_OVERLAP = 60  # ~40-50 words overlap
MIN_CHARS = 100     # Skip very short pages

# Backend configuration
USE_QDRANT = True
USE_PGVECTOR = True

# Qdrant settings
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "course_docs_clean"
QDRANT_VECTOR_SIZE = 768  # E5-base dimensions
QDRANT_DISTANCE = "Cosine"

# PostgreSQL+pgvector settings
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "pguser"
PG_PASSWORD = "pgpass"
PG_DATABASE = "vectordb"
PG_TABLE = "docs_clean"
PG_DIM = 768

# Build DSN
PG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# Batch processing (smaller to avoid memory issues)
BATCH_SIZE = 8   # Small batches to prevent RAM overflow
MAX_CHUNKS_PER_FILE = 50  # Limit chunks per file to avoid memory issues

# E5 prefixes (critical for multilingual-e5)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
