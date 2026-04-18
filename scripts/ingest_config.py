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

# Chunking parameters (optimized for short queries like "what time are classes")
CHUNK_TOKENS = 250  # Smaller chunks ~150-200 words for better granular search
CHUNK_OVERLAP = 50  # ~30-40 words overlap to preserve context
MIN_CHARS = 50      # Allow shorter chunks to capture specific details

# Backend configuration (overridable via env vars / .env.local)
USE_QDRANT = True
USE_PGVECTOR = True

# Qdrant settings
# Local file path (set by install_mac.sh / .env.local) takes priority over URL
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH")  # e.g. ./data/qdrant_local
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")  # Docker default
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "course_docs_clean")
QDRANT_VECTOR_SIZE = 768  # E5-base dimensions
QDRANT_DISTANCE = "Cosine"

# PostgreSQL+pgvector settings
PG_HOST = "postgres"  # Use Docker service name for container networking
PG_PORT = 5432
PG_USER = "pguser"
PG_PASSWORD = "pgpass"
PG_DATABASE = "vectordb"
PG_TABLE = "docs_clean"
PG_DIM = 768

# Build DSN
PG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# Disable pgvector for local/offline mode
USE_PGVECTOR = os.getenv(
    "USE_PGVECTOR", "true").lower() not in ("false", "0", "no")
USE_QDRANT = os.getenv(
    "USE_QDRANT", "true").lower() not in ("false", "0", "no")

# Batch processing (optimized for HIGH-PERFORMANCE hardware - 24-core i7 + RTX 5060)
BATCH_SIZE = 512   # Aggressive batching for your powerful CPU
LARGE_BATCH_SIZE = 1024  # Your 24 cores can handle massive parallel processing
MAX_CHUNKS_PER_FILE = 2000  # No limits needed with your specs

# E5 prefixes (critical for multilingual-e5)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
