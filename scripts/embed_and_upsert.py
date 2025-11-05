"""
E5 embedder with proper prefixes and L2 normalization.
Upserts to both Qdrant and pgvector with consistent cosine similarity.
"""
import json
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Generator

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("❌ sentence-transformers not installed. Install with: pip install sentence-transformers")

from ingest_config import (
    CLEAN_DIR, EMBED_MODEL,
    USE_QDRANT, USE_PGVECTOR, BATCH_SIZE,
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE, QDRANT_DISTANCE,
    PG_DSN, PG_TABLE, PG_DIM,
    E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
)

# Load embedding model
_model = None


def get_embedding_model():
    """Load and cache the embedding model"""
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers required for embeddings")
        print(f"📥 Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        print(
            f"✅ Model loaded. Dimensions: {_model.get_sentence_embedding_dimension()}")
    return _model


def e5_embed(texts: List[str], mode: str = "passage") -> np.ndarray:
    """
    Embed texts using E5 with proper prefixes and L2 normalization.

    Args:
        texts: List of strings to embed
        mode: "query" or "passage" (determines prefix)

    Returns:
        Normalized embeddings as numpy array
    """
    model = get_embedding_model()

    # Add E5 prefixes (critical for multilingual-e5)
    prefix = E5_QUERY_PREFIX if mode == "query" else E5_PASSAGE_PREFIX
    prefixed_texts = [prefix + text for text in texts]

    # Embed with L2 normalization (important for cosine similarity)
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True)

    return embeddings


# ============================================================================
# Qdrant Setup
# ============================================================================

_qdrant_client = None


def get_qdrant_client():
    """Initialize Qdrant client and collection"""
    global _qdrant_client

    if not USE_QDRANT:
        return None

    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams, PointStruct

            print(f"🔗 Connecting to Qdrant: {QDRANT_URL}")
            _qdrant_client = QdrantClient(url=QDRANT_URL)

            # Check if collection exists, create if not
            try:
                collection_info = _qdrant_client.get_collection(
                    QDRANT_COLLECTION)
                print(f"📊 Qdrant collection '{QDRANT_COLLECTION}' exists")
                print(f"   Points: {collection_info.points_count}")
                print(
                    f"   Dimensions: {collection_info.config.params.vectors.size}")
                print(
                    f"   Distance: {collection_info.config.params.vectors.distance}")
            except:
                print(f"🆕 Creating Qdrant collection: {QDRANT_COLLECTION}")
                _qdrant_client.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print("✅ Collection created with cosine distance")

        except ImportError:
            print(
                "❌ qdrant-client not installed. Install with: pip install qdrant-client")
            return None
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            return None

    return _qdrant_client


# ============================================================================
# PostgreSQL Setup
# ============================================================================

def setup_pgvector():
    """Initialize PostgreSQL table and index"""
    if not USE_PGVECTOR:
        return False

    try:
        import psycopg2

        print(f"🔗 Connecting to PostgreSQL: {PG_DSN}")
        conn = psycopg2.connect(PG_DSN)
        conn.autocommit = True

        with conn.cursor() as cur:
            # Enable vector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create table with proper schema
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {PG_TABLE} (
                    id bigserial PRIMARY KEY,
                    source_path text NOT NULL,
                    page integer NOT NULL,
                    chunk_id integer NOT NULL,
                    content text NOT NULL,
                    metadata jsonb,
                    embedding vector({PG_DIM}) NOT NULL
                );
            """)

            # Create cosine similarity index
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {PG_TABLE}_cosine_idx 
                ON {PG_TABLE} USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 200);
            """)

            # Check existing data
            cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE};")
            count = cur.fetchone()[0]
            print(
                f"📊 PostgreSQL table '{PG_TABLE}' ready (existing rows: {count})")

        conn.close()
        return True

    except ImportError:
        print("❌ psycopg2 not installed. Install with: pip install psycopg2")
        return False
    except Exception as e:
        print(f"❌ Failed to setup PostgreSQL: {e}")
        return False


# ============================================================================
# Batch Processing
# ============================================================================

def upsert_to_qdrant(batch_records: List[Dict], embeddings: np.ndarray):
    """Upsert batch to Qdrant"""
    client = get_qdrant_client()
    if not client:
        return

    try:
        from qdrant_client.http.models import PointStruct

        points = []
        for record, embedding in zip(batch_records, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "source_path": record["source_path"],
                    "page": record["page"],
                    "chunk_id": record["chunk_id"],
                    "content": record["content"],
                    "metadata": record.get("metadata", {})
                }
            )
            points.append(point)

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"  📤 Qdrant: {len(points)} points upserted")

    except Exception as e:
        print(f"  ❌ Qdrant upsert failed: {e}")


def upsert_to_pgvector(batch_records: List[Dict], embeddings: np.ndarray):
    """Upsert batch to PostgreSQL"""
    if not USE_PGVECTOR:
        return

    try:
        import psycopg2
        import json

        conn = psycopg2.connect(PG_DSN)
        conn.autocommit = True

        with conn.cursor() as cur:
            # Prepare batch insert
            values = []
            for record, embedding in zip(batch_records, embeddings):
                values.append((
                    record["source_path"],
                    record["page"],
                    record["chunk_id"],
                    record["content"],
                    json.dumps(record.get("metadata", {})),
                    embedding.tolist()  # Convert to list for pgvector
                ))

            # Bulk insert
            cur.executemany(f"""
                INSERT INTO {PG_TABLE} 
                (source_path, page, chunk_id, content, metadata, embedding) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """, values)

            print(f"  📤 PostgreSQL: {len(values)} rows inserted")

        conn.close()

    except Exception as e:
        print(f"  ❌ PostgreSQL upsert failed: {e}")


def process_batch(batch_records: List[Dict]):
    """Process a batch of records: embed and upsert to backends (memory optimized)"""
    if not batch_records:
        return

    # Extract content for embedding
    contents = [record["content"] for record in batch_records]

    try:
        # Generate embeddings
        embeddings = e5_embed(contents, mode="passage")

        # Upsert to backends
        if USE_QDRANT:
            upsert_to_qdrant(batch_records, embeddings)

        if USE_PGVECTOR:
            upsert_to_pgvector(batch_records, embeddings)

    finally:
        # Force garbage collection after each batch to prevent memory buildup
        import gc
        gc.collect()


# ============================================================================
# Data Streaming
# ============================================================================

def stream_chunks() -> Generator[Dict, None, None]:
    """Stream chunk records from all .chunks.jsonl files"""
    clean_path = Path(CLEAN_DIR)
    chunk_files = list(clean_path.glob("*.chunks.jsonl"))

    if not chunk_files:
        print(f"❌ No chunk files found in {CLEAN_DIR}")
        print("   Run pdf_cleaner.py and chunker.py first")
        return

    print(f"📖 Reading chunks from {len(chunk_files)} files...")

    total_chunks = 0
    for chunk_file in chunk_files:
        print(f"  📄 Reading: {chunk_file.name}")

        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                file_chunks = 0
                for line in f:
                    if line.strip():
                        yield json.loads(line.strip())
                        file_chunks += 1
                        total_chunks += 1

                print(f"    📊 {file_chunks} chunks")

        except Exception as e:
            print(f"    ❌ Error reading {chunk_file.name}: {e}")
            continue

    print(f"📊 Total chunks to process: {total_chunks}")


def clear_existing_data():
    """Clear existing data from backends"""
    print("🧹 Clearing existing data...")

    # Clear Qdrant
    if USE_QDRANT:
        client = get_qdrant_client()
        if client:
            try:
                client.delete_collection(QDRANT_COLLECTION)
                print("  🗑️  Qdrant collection deleted")

                # Recreate
                from qdrant_client.http.models import Distance, VectorParams
                client.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print("  ✅ Qdrant collection recreated")
            except Exception as e:
                print(f"  ❌ Qdrant clear failed: {e}")

    # Clear PostgreSQL
    if USE_PGVECTOR:
        try:
            import psycopg2
            conn = psycopg2.connect(PG_DSN)
            conn.autocommit = True

            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {PG_TABLE};")
                print("  🗑️  PostgreSQL table dropped")

            conn.close()

            # Recreate table
            setup_pgvector()

        except Exception as e:
            print(f"  ❌ PostgreSQL clear failed: {e}")


# ============================================================================
# Main Processing
# ============================================================================

def embed_and_upsert_all(clear_first: bool = True):
    """Main function to embed all chunks and upsert to backends"""
    print("=" * 60)
    print("🚀 E5 Embedding & Upsert Pipeline")
    print("=" * 60)

    # Initialize backends
    if USE_QDRANT:
        get_qdrant_client()

    if USE_PGVECTOR:
        setup_pgvector()

    # Clear existing data if requested
    if clear_first:
        clear_existing_data()
        # Re-initialize after clearing
        if USE_QDRANT:
            get_qdrant_client()
        if USE_PGVECTOR:
            setup_pgvector()

    # Process chunks in batches
    batch = []
    total_processed = 0

    for record in stream_chunks():
        batch.append(record)

        if len(batch) >= BATCH_SIZE:
            print(
                f"\n📦 Processing batch {total_processed//BATCH_SIZE + 1} ({len(batch)} chunks)")
            process_batch(batch)
            total_processed += len(batch)
            batch = []

    # Process final partial batch
    if batch:
        print(f"\n📦 Processing final batch ({len(batch)} chunks)")
        process_batch(batch)
        total_processed += len(batch)

    print(f"\n✅ Processing complete!")
    print(f"📊 Total chunks processed: {total_processed}")
    print(f"🧠 Embedding model: {EMBED_MODEL}")
    print(f"📐 Vector dimensions: {QDRANT_VECTOR_SIZE}")
    print(f"📏 Distance metric: Cosine similarity")

    # Final stats
    if USE_QDRANT:
        client = get_qdrant_client()
        if client:
            try:
                info = client.get_collection(QDRANT_COLLECTION)
                print(f"📊 Qdrant final count: {info.points_count}")
            except:
                pass

    if USE_PGVECTOR:
        try:
            import psycopg2
            conn = psycopg2.connect(PG_DSN)
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE};")
                count = cur.fetchone()[0]
                print(f"📊 PostgreSQL final count: {count}")
            conn.close()
        except:
            pass


if __name__ == "__main__":
    embed_and_upsert_all(clear_first=True)
