#!/usr/bin/env python3
"""
Complete RAG Pipeline - All-in-one processing
Handles: PDF cleaning → text extraction → table parsing → chunking → embedding → database upsert

Usage:
    python scripts/pipeline.py                     # Process all files in data/raw
    python scripts/pipeline.py --clear             # Clear databases first
    python scripts/pipeline.py --config            # Show configuration
    python scripts/pipeline.py --file path.pdf     # Process single file
"""

import os
import sys
import json
import re
import time
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple
import uuid

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"

# Embedding settings
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_DIMENSIONS = 768
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Chunking settings
CHUNK_TOKENS = 350
CHUNK_OVERLAP = 60
MIN_CHARS = 100

# Database settings
USE_QDRANT = True
USE_PGVECTOR = True
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "course_docs_clean"
PG_DSN = "postgresql://pguser:pgpass@localhost:5432/vectordb"
PG_TABLE = "docs_clean"

# Processing settings
BATCH_SIZE = 8
MAX_MEMORY_MB = 500  # Memory limit for chunks

# Create clean directory
Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# TEXT CLEANING & WATERMARK REMOVAL
# ============================================================================


def remove_watermarks(text: str) -> str:
    """Aggressively remove watermarks and PDF corruption artifacts."""

    # UTP institutional watermarks
    watermarks = [
        r"UTP-FISC-2S2025",
        r"UTP[\s\-]*FISC[\s\-]*\d+S\d+",
        r"Universidad\s+Tecnológica\s+de\s+Panamá",
        r"Facultad\s+de\s+Ingeniería\s+de\s+Sistemas\s+Computacionales",
        r"Maestría\s+en\s+Analítica\s+de\s+Datos",
        r"Sistemas\s+de\s+Bases\s+de\s+Datos\s+Avanzadas",
        r"GUIA\s+DE\s+CURSO.*SEMESTRE\s+\d+",
        r"Ing\.\s+Boris\s+Landero\s+Pág\.\s+\d+\s+de\s+\d+",
        r"SistemFas\s+Ide\s+Base\s+de\s+Datos\s+Avanzada",
    ]

    # Remove watermark patterns
    for pattern in watermarks:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove scattered letter artifacts from OCR corruption
    corrupted_patterns = [
        r"\b[A-Z]\s+[A-Z]\s+[-–]\s+[A-Z]\s+\d+\s+[A-Z]\b",  # "U C - F 2 S"
        r"\b\d+\s+[A-Z]\s+\d+\s+[A-Z]\s+\d+\s+[A-Z]\b",     # "2 S 5 P 2 S"
        r"\b[A-Z]\s+\d+\s+[A-Z]\s+\d+\s+[A-Z]\b",           # "F 2 U C S"
        r"(?:[A-Z]\s+){3,}\d+",                              # "F S C 2"
        r"(?:\d+\s+){3,}[A-Z]",                              # "2 5 0 T"
        r"\b[T]\s+[C]\s+[-]\s+[F]",                          # "T C - F"
        r"[A-Z]\s+[A-Z]\s+[A-Z]\s+[-]\s+\d+",               # Common pattern
        # Start of pages with corruption
        r"Page\s+\d+:\n[A-Z]\s+[A-Z]\s+[-]",
    ]

    for pattern in corrupted_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove lines with high corruption ratio (>70% single letters/numbers)
    lines = text.split('\n')
    clean_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue

        # Count single letters/numbers separated by spaces
        corruption_chars = len(re.findall(r'\b[A-Z0-9]\b', line))
        total_chars = len(re.findall(r'\S', line))

        if total_chars > 0:
            corruption_ratio = corruption_chars / total_chars
            if corruption_ratio > 0.7:  # Skip highly corrupted lines
                continue

        # Keep meaningful lines
        if len(line) > 10 and any(c.isalpha() for c in line):
            clean_lines.append(line)

    return '\n'.join(clean_lines)


def normalize_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""

    # Remove watermarks first
    text = remove_watermarks(text)

    # Remove control characters
    text = text.replace("\x00", " ").replace("\ufffd", "")
    text = text.replace("\u00ad", "")  # Soft hyphens

    # Fix hyphenated words across line breaks
    text = re.sub(r"-\s*\n\s*", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ============================================================================
# PDF TEXT & TABLE EXTRACTION
# ============================================================================


def extract_tables_from_pdf(pdf_path: Path) -> List[str]:
    """Extract tables from PDF and convert to text format."""
    tables_text = []

    if not pdfplumber:
        return tables_text

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables
                tables = page.extract_tables()

                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:  # Has header + data
                        table_text = f"\n=== TABLA {table_idx + 1} (Página {page_num}) ===\n"

                        # Convert table to text format
                        for row_idx, row in enumerate(table):
                            if row and any(cell for cell in row if cell):  # Non-empty row
                                # Clean cells
                                clean_row = []
                                for cell in row:
                                    if cell:
                                        clean_cell = normalize_text(str(cell))
                                        if clean_cell:
                                            clean_row.append(clean_cell)

                                if clean_row:
                                    if row_idx == 0:  # Header
                                        table_text += " | ".join(
                                            clean_row) + "\n"
                                        table_text += "-" * \
                                            (len(" | ".join(clean_row))) + "\n"
                                    else:  # Data
                                        table_text += " | ".join(
                                            clean_row) + "\n"

                        table_text += "=== FIN TABLA ===\n"
                        tables_text.append(table_text)

    except Exception as e:
        print(f"   ⚠️  Table extraction error: {e}")

    return tables_text


def extract_pdf_content(pdf_path: Path) -> Tuple[str, List[str]]:
    """Extract both regular text and tables from PDF."""
    text_content = ""
    tables_content = []

    if not pdfplumber:
        print(f"   ❌ pdfplumber not available for {pdf_path.name}")
        return text_content, tables_content

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []

            for page_num, page in enumerate(pdf.pages, 1):
                # Extract regular text
                page_text = page.extract_text() or ""
                if page_text:
                    cleaned_text = normalize_text(page_text)
                    if cleaned_text and len(cleaned_text) > MIN_CHARS:
                        pages_text.append(
                            f"=== Página {page_num} ===\n{cleaned_text}")

            text_content = "\n\n".join(pages_text)

            # Extract tables separately
            tables_content = extract_tables_from_pdf(pdf_path)

        print(
            f"   ✅ Extracted {len(pages_text)} pages + {len(tables_content)} tables")
        return text_content, tables_content

    except Exception as e:
        print(f"   ❌ PDF extraction failed: {e}")
        return "", []


def process_text_file(file_path: Path) -> str:
    """Process text/markdown files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        cleaned = normalize_text(content)
        print(f"   ✅ Extracted {len(cleaned)} characters")
        return cleaned

    except Exception as e:
        print(f"   ❌ Text processing failed: {e}")
        return ""

# ============================================================================
# CHUNKING
# ============================================================================


def simple_token_count(text: str) -> int:
    """Simple word-based token approximation."""
    tokens = re.findall(r'\w+', text)
    return len(tokens)


def chunk_text(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Memory-efficient word-based chunking."""
    if not text or not text.strip():
        return []

    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text and len(chunk_text) > 50:  # Skip very short chunks
            chunks.append(chunk_text)

        # Move with overlap but ensure progress
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, max_tokens // 2)
        start = next_start

    return chunks


def create_chunks_from_content(source_path: str, content: str, content_type: str = "text") -> List[Dict]:
    """Create chunks with metadata from content."""
    chunks = chunk_text(content)
    chunk_records = []

    for chunk_idx, chunk_content in enumerate(chunks):
        record = {
            "source_path": source_path,
            "chunk_id": chunk_idx,
            "content": chunk_content,
            "metadata": {
                "content_type": content_type,
                "source_name": Path(source_path).name,
                "char_count": len(chunk_content),
                "word_count": len(chunk_content.split()),
                "token_count": simple_token_count(chunk_content)
            }
        }
        chunk_records.append(record)

    return chunk_records

# ============================================================================
# EMBEDDING
# ============================================================================


_embedding_model = None


def get_embedding_model():
    """Load and cache embedding model."""
    global _embedding_model
    if _embedding_model is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers required")
        print(f"🤖 Loading embedding model: {EMBED_MODEL}")
        _embedding_model = SentenceTransformer(EMBED_MODEL)
    return _embedding_model


def embed_texts(texts: List[str], mode: str = "passage") -> np.ndarray:
    """Embed texts with E5 prefixes and normalization."""
    model = get_embedding_model()

    # Add E5 prefixes
    prefix = E5_QUERY_PREFIX if mode == "query" else E5_PASSAGE_PREFIX
    prefixed_texts = [prefix + text for text in texts]

    # Embed with L2 normalization for cosine similarity
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True)
    return embeddings

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================


def clear_databases():
    """Clear existing data from both databases."""
    print("🧹 Clearing databases...")

    # Clear Qdrant
    if USE_QDRANT and QdrantClient:
        try:
            client = QdrantClient(url=QDRANT_URL)
            try:
                client.delete_collection(QDRANT_COLLECTION)
                print(f"   ✅ Cleared Qdrant collection: {QDRANT_COLLECTION}")
            except:
                print(f"   ℹ️  Collection {QDRANT_COLLECTION} doesn't exist")

            # Create fresh collection
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBED_DIMENSIONS, distance=Distance.COSINE)
            )
            print(f"   ✅ Created fresh Qdrant collection")
        except Exception as e:
            print(f"   ❌ Qdrant clear failed: {e}")

    # Clear PostgreSQL
    if USE_PGVECTOR and psycopg2:
        try:
            conn = psycopg2.connect(PG_DSN)
            conn.autocommit = True
            cur = conn.cursor()

            # Drop and recreate table
            cur.execute(f'DROP TABLE IF EXISTS {PG_TABLE};')
            cur.execute(f'''
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE {PG_TABLE} (
                    id SERIAL PRIMARY KEY,
                    source_path TEXT,
                    chunk_id INTEGER,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({EMBED_DIMENSIONS})
                );
                CREATE INDEX {PG_TABLE}_emb_cosine_idx 
                ON {PG_TABLE} USING hnsw (embedding vector_cosine_ops);
            ''')

            conn.close()
            print(f"   ✅ Cleared PostgreSQL table: {PG_TABLE}")
        except Exception as e:
            print(f"   ❌ PostgreSQL clear failed: {e}")


def upsert_batch_to_qdrant(batch_records: List[Dict], embeddings: np.ndarray):
    """Upsert batch to Qdrant."""
    if not USE_QDRANT or not QdrantClient:
        return

    try:
        client = QdrantClient(url=QDRANT_URL)
        points = []

        for record, embedding in zip(batch_records, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "source_path": record["source_path"],
                    "chunk_id": record["chunk_id"],
                    "content": record["content"],
                    "metadata": record["metadata"]
                }
            )
            points.append(point)

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    except Exception as e:
        print(f"   ❌ Qdrant upsert failed: {e}")


def upsert_batch_to_pgvector(batch_records: List[Dict], embeddings: np.ndarray):
    """Upsert batch to PostgreSQL."""
    if not USE_PGVECTOR or not psycopg2:
        return

    try:
        conn = psycopg2.connect(PG_DSN)
        conn.autocommit = True
        cur = conn.cursor()

        for record, embedding in zip(batch_records, embeddings):
            cur.execute(f'''
                INSERT INTO {PG_TABLE} (source_path, chunk_id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                record["source_path"],
                record["chunk_id"],
                record["content"],
                json.dumps(record["metadata"]),
                embedding.tolist()
            ))

        conn.close()

    except Exception as e:
        print(f"   ❌ PostgreSQL upsert failed: {e}")


def process_chunk_batch(batch_records: List[Dict]):
    """Process and upsert a batch of chunks."""
    if not batch_records:
        return

    try:
        # Extract content for embedding
        contents = [record["content"] for record in batch_records]

        # Generate embeddings
        embeddings = embed_texts(contents, mode="passage")

        # Upsert to databases
        upsert_batch_to_qdrant(batch_records, embeddings)
        upsert_batch_to_pgvector(batch_records, embeddings)

        print(f"   ✅ Processed {len(batch_records)} chunks")

    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
    finally:
        # Force garbage collection to free memory
        gc.collect()

# ============================================================================
# MAIN PIPELINE
# ============================================================================


def process_single_file(file_path: Path) -> List[Dict]:
    """Process a single file and return chunk records."""
    print(f"\n📄 Processing: {file_path.name}")

    all_chunks = []

    if file_path.suffix.lower() == '.pdf':
        # Extract PDF content and tables
        text_content, tables_content = extract_pdf_content(file_path)

        # Process main text
        if text_content:
            text_chunks = create_chunks_from_content(
                str(file_path), text_content, "pdf_text")
            all_chunks.extend(text_chunks)
            print(f"   📝 Text chunks: {len(text_chunks)}")

        # Process tables separately
        for table_idx, table_content in enumerate(tables_content):
            if table_content:
                table_chunks = create_chunks_from_content(
                    str(file_path), table_content, f"pdf_table_{table_idx}")
                all_chunks.extend(table_chunks)
                print(
                    f"   📊 Table {table_idx + 1} chunks: {len(table_chunks)}")

    elif file_path.suffix.lower() in ['.txt', '.md']:
        # Process text files
        content = process_text_file(file_path)
        if content:
            text_chunks = create_chunks_from_content(
                str(file_path), content, "text")
            all_chunks.extend(text_chunks)
            print(f"   📝 Text chunks: {len(text_chunks)}")

    else:
        print(f"   ⚠️  Unsupported file type: {file_path.suffix}")

    return all_chunks


def run_pipeline(target_files: List[Path] = None, clear_first: bool = True):
    """Run the complete pipeline."""
    print("=" * 60)
    print("🚀 Complete RAG Pipeline")
    print("=" * 60)

    start_time = time.time()

    # Check dependencies
    missing_deps = []
    if USE_QDRANT and not QdrantClient:
        missing_deps.append("qdrant-client")
    if USE_PGVECTOR and not psycopg2:
        missing_deps.append("psycopg2")
    if not SentenceTransformer:
        missing_deps.append("sentence-transformers")
    if not np:
        missing_deps.append("numpy")
    if not pdfplumber:
        missing_deps.append("pdfplumber")

    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install " + " ".join(missing_deps))
        return False

    # Clear databases if requested
    if clear_first:
        clear_databases()

    # Find files to process
    if not target_files:
        raw_path = Path(RAW_DIR)
        if not raw_path.exists():
            print(f"❌ Raw directory not found: {RAW_DIR}")
            return False

        target_files = []
        for pattern in ['*.pdf', '*.txt', '*.md']:
            target_files.extend(raw_path.glob(pattern))

    if not target_files:
        print("❌ No files to process")
        return False

    print(f"📂 Processing {len(target_files)} files...")

    # Process files and collect all chunks
    all_chunks = []
    total_files = len(target_files)

    for file_idx, file_path in enumerate(target_files, 1):
        print(f"\n[{file_idx}/{total_files}] Processing: {file_path.name}")

        try:
            file_chunks = process_single_file(file_path)
            all_chunks.extend(file_chunks)
            print(f"   ✅ Generated {len(file_chunks)} chunks")

        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            continue

    if not all_chunks:
        print("❌ No chunks generated")
        return False

    print(f"\n📊 Total chunks to embed: {len(all_chunks)}")

    # Process chunks in batches to avoid memory issues
    print("\n🧠 Embedding and upserting chunks...")

    batch_num = 0
    total_processed = 0

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        batch_num += 1

        print(f"   📦 Batch {batch_num}: {len(batch)} chunks",
              end="", flush=True)

        try:
            process_chunk_batch(batch)
            total_processed += len(batch)
            print(" ✅")
        except Exception as e:
            print(f" ❌ {e}")
            continue

    elapsed = time.time() - start_time

    print(f"\n✅ Pipeline Complete!")
    print(f"📊 Total chunks processed: {total_processed}")
    print(f"⏱️  Time elapsed: {elapsed:.1f} seconds")
    print(f"📈 Throughput: {total_processed/elapsed:.1f} chunks/sec")

    return True


def show_configuration():
    """Display current configuration."""
    print("⚙️  Pipeline Configuration")
    print("=" * 40)
    print(f"📁 Raw directory: {RAW_DIR}")
    print(f"📁 Clean directory: {CLEAN_DIR}")
    print(f"🤖 Embedding model: {EMBED_MODEL}")
    print(f"📐 Dimensions: {EMBED_DIMENSIONS}")
    print(f"✂️  Chunk tokens: {CHUNK_TOKENS}")
    print(f"🔄 Chunk overlap: {CHUNK_OVERLAP}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🗄️  Qdrant: {'✅' if USE_QDRANT else '❌'} → {QDRANT_COLLECTION}")
    print(f"🗄️  PostgreSQL: {'✅' if USE_PGVECTOR else '❌'} → {PG_TABLE}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Complete RAG Pipeline")
    parser.add_argument("--config", action="store_true",
                        help="Show configuration")
    parser.add_argument("--clear", action="store_true",
                        help="Clear databases before processing")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear databases")
    parser.add_argument("--file", help="Process single file")

    args = parser.parse_args()

    if args.config:
        show_configuration()
        return

    # Determine files to process
    target_files = None
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            return
        target_files = [file_path]

    # Determine whether to clear databases
    clear_first = not args.no_clear  # Default: clear unless --no-clear
    if args.clear:
        clear_first = True

    # Run pipeline
    success = run_pipeline(target_files, clear_first)

    if not success:
        print("\n❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
