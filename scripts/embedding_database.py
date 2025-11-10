#!/usr/bin/env python3
"""
Unified Embedding and Database Operations Module
Consolidates embed_and_upsert.py, embed_safe.py, and related database functionality
Supports both Qdrant and PostgreSQL with memory-efficient processing
"""

from ingest_config import (
    CLEAN_DIR, EMBED_MODEL,
    USE_QDRANT, USE_PGVECTOR, BATCH_SIZE,
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE, QDRANT_DISTANCE,
    PG_DSN, PG_TABLE, PG_DIM,
    E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
)
import json
import logging
import gc
import time
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies with fallbacks
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.error(
        "sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None
    logger.warning("qdrant-client not available")

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None
    logger.warning("psycopg2 not available")


class UnifiedEmbeddingProcessor:
    """
    Unified processor for embeddings and database operations
    Combines memory-safe processing with robust database management
    """

    def __init__(self, memory_safe_mode: bool = True, batch_size: int = None):
        self.memory_safe_mode = memory_safe_mode
        self.batch_size = batch_size or (4 if memory_safe_mode else BATCH_SIZE)
        self.model = None
        self.qdrant_client = None
        self.pg_connection = None

        logger.info(
            f"Initialized processor - Memory safe: {memory_safe_mode}, Batch size: {self.batch_size}")

    def get_embedding_model(self):
        """Load and cache the embedding model"""
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed")

            logger.info(f"🤖 Loading embedding model: {EMBED_MODEL}")
            self.model = SentenceTransformer(EMBED_MODEL)
            logger.info(
                f"✅ Model loaded (dimensions: {self.model.get_sentence_embedding_dimension()})")

        return self.model

    def embed_texts(self, texts: List[str], mode: str = "passage") -> np.ndarray:
        """
        Embed texts using E5 with proper prefixes and L2 normalization

        Args:
            texts: List of strings to embed
            mode: "query" or "passage" (determines prefix)

        Returns:
            Normalized embeddings as numpy array
        """
        model = self.get_embedding_model()

        # Add E5 prefixes (critical for multilingual-e5)
        prefix = E5_QUERY_PREFIX if mode == "query" else E5_PASSAGE_PREFIX
        prefixed_texts = [prefix + text for text in texts]

        # Embed with L2 normalization (important for cosine similarity)
        embeddings = model.encode(prefixed_texts, normalize_embeddings=True)

        return embeddings

    def setup_qdrant(self):
        """Initialize Qdrant client and collection"""
        if not USE_QDRANT:
            return None

        if self.qdrant_client is None:
            if QdrantClient is None:
                raise ImportError("qdrant-client not installed")

            logger.info(f"🔗 Connecting to Qdrant at {QDRANT_URL}")
            self.qdrant_client = QdrantClient(url=QDRANT_URL)

            # Check if collection exists, create if needed
            try:
                collection_info = self.qdrant_client.get_collection(
                    QDRANT_COLLECTION)
                logger.info(
                    f"✅ Using existing Qdrant collection: {QDRANT_COLLECTION}")
            except:
                logger.info(
                    f"📋 Creating Qdrant collection: {QDRANT_COLLECTION}")
                self.qdrant_client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(
                    f"✅ Qdrant collection created: {QDRANT_COLLECTION}")

        return self.qdrant_client

    def setup_pgvector(self):
        """Initialize PostgreSQL table and index"""
        if not USE_PGVECTOR:
            return None

        if self.pg_connection is None:
            if psycopg2 is None:
                raise ImportError("psycopg2 not installed")

            logger.info(f"🔗 Connecting to PostgreSQL")
            self.pg_connection = psycopg2.connect(PG_DSN)
            self.pg_connection.autocommit = True

            cur = self.pg_connection.cursor()

            # Create pgvector extension if not exists
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')

            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (PG_TABLE,))

            table_exists = cur.fetchone()[0]

            if not table_exists:
                logger.info(f"📋 Creating PostgreSQL table: {PG_TABLE}")
                cur.execute(f'''
                    CREATE TABLE {PG_TABLE} (
                        id SERIAL PRIMARY KEY,
                        source_path TEXT,
                        page INTEGER,
                        chunk_id INTEGER,
                        content TEXT,
                        metadata JSONB,
                        embedding vector({PG_DIM})
                    );
                ''')

                # Create index
                cur.execute(f'''
                    CREATE INDEX {PG_TABLE}_emb_cosine_idx 
                    ON {PG_TABLE} USING hnsw (embedding vector_cosine_ops);
                ''')
                logger.info(f"✅ PostgreSQL table created: {PG_TABLE}")
            else:
                logger.info(f"✅ Using existing PostgreSQL table: {PG_TABLE}")

        return self.pg_connection

    def clear_databases(self):
        """Clear existing data from both databases"""
        logger.info("🧹 Clearing existing data...")

        if USE_QDRANT:
            client = self.setup_qdrant()
            if client:
                try:
                    client.delete_collection(QDRANT_COLLECTION)
                    logger.info(
                        f"🗑️  Cleared Qdrant collection: {QDRANT_COLLECTION}")
                except:
                    logger.info(
                        f"🗑️  Qdrant collection {QDRANT_COLLECTION} didn't exist")

                # Recreate the collection after deletion
                try:
                    logger.info(
                        f"📋 Recreating Qdrant collection: {QDRANT_COLLECTION}")
                    client.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(
                            size=QDRANT_VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(
                        f"✅ Qdrant collection recreated: {QDRANT_COLLECTION}")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to recreate Qdrant collection: {e}")

        if USE_PGVECTOR:
            conn = self.setup_pgvector()
            if conn:
                cur = conn.cursor()
                try:
                    cur.execute(f'DELETE FROM {PG_TABLE};')
                    logger.info(f"🗑️  Cleared PostgreSQL table: {PG_TABLE}")
                except:
                    logger.info(
                        f"🗑️  PostgreSQL table {PG_TABLE} didn't exist")

    def upsert_batch_to_qdrant(self, batch_records: List[Dict], embeddings: np.ndarray):
        """Upsert batch to Qdrant"""
        client = self.setup_qdrant()
        if not client:
            return

        try:
            points = []
            for i, (record, embedding) in enumerate(zip(batch_records, embeddings)):
                point_id = f"{record['source_path']}_{record.get('page', 0)}_{record.get('chunk_id', i)}"
                # Create a hash-based ID to avoid collisions - ensure positive integer
                point_id = abs(hash(point_id)) % (2**31 - 1)

                point = PointStruct(
                    id=point_id,  # Now using integer directly
                    vector=embedding.tolist(),
                    payload={
                        "source_path": record["source_path"],
                        "page": record.get("page", 0),
                        "chunk_id": record.get("chunk_id", i),
                        "content": record["content"],
                        "metadata": record.get("metadata", {})
                    }
                )
                points.append(point)

            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            logger.debug(f"📤 Qdrant: {len(points)} points upserted")

        except Exception as e:
            logger.error(f"❌ Qdrant upsert failed: {e}")
            raise

    def upsert_batch_to_pgvector(self, batch_records: List[Dict], embeddings: np.ndarray):
        """Upsert batch to PostgreSQL"""
        conn = self.setup_pgvector()
        if not conn:
            return

        try:
            cur = conn.cursor()

            # Prepare batch insert
            insert_data = []
            for record, embedding in zip(batch_records, embeddings):
                # Convert page to integer for PostgreSQL compatibility
                page = record.get("page", 1)
                if isinstance(page, str):
                    # Handle page ranges like "1-25" -> 1
                    try:
                        page = int(page.split('-')[0])
                    except:
                        page = 1

                insert_data.append((
                    record["source_path"],
                    page,
                    record.get("chunk_id", 0),
                    record["content"],
                    json.dumps(record.get("metadata", {})),
                    embedding.tolist()
                ))

            # Batch insert
            insert_query = f'''
                INSERT INTO {PG_TABLE} (source_path, page, chunk_id, content, metadata, embedding)
                VALUES %s
            '''

            psycopg2.extras.execute_values(
                cur, insert_query, insert_data,
                template=None, page_size=self.batch_size
            )

            logger.debug(f"📤 PostgreSQL: {len(insert_data)} rows inserted")

        except Exception as e:
            logger.error(f"❌ PostgreSQL upsert failed: {e}")
            raise

    def process_batch(self, batch_records: List[Dict]):
        """Process a batch of records: embed and upsert to backends"""
        if not batch_records:
            return

        # Extract texts for embedding
        texts = [record["content"] for record in batch_records]

        # Generate embeddings
        logger.debug(f"🧠 Embedding batch of {len(texts)} texts...")
        embeddings = self.embed_texts(texts, mode="passage")

        # Upsert to backends
        if USE_QDRANT:
            try:
                self.upsert_batch_to_qdrant(batch_records, embeddings)
            except Exception as e:
                logger.error(f"Qdrant batch failed: {e}")

        if USE_PGVECTOR:
            try:
                self.upsert_batch_to_pgvector(batch_records, embeddings)
            except Exception as e:
                logger.error(f"PostgreSQL batch failed: {e}")

        # Memory cleanup in safe mode
        if self.memory_safe_mode:
            del embeddings
            gc.collect()

    def stream_chunks(self) -> Generator[Dict, None, None]:
        """Stream chunk records from all .chunks.jsonl files"""
        clean_path = Path(CLEAN_DIR)
        chunk_files = list(clean_path.glob("*.chunks.jsonl"))

        if not chunk_files:
            logger.warning(f"No chunk files found in {CLEAN_DIR}")
            return

        logger.info(f"📖 Reading chunks from {len(chunk_files)} files...")

        total_chunks = 0
        for chunk_file in chunk_files:
            logger.info(f"  📄 Reading: {chunk_file.name}")

            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    file_chunks = 0
                    for line in f:
                        if line.strip():
                            yield json.loads(line.strip())
                            file_chunks += 1
                            total_chunks += 1

                    logger.info(f"    📊 {file_chunks} chunks")

            except Exception as e:
                logger.error(f"Error reading {chunk_file.name}: {e}")
                continue

        logger.info(f"📊 Total chunks streamed: {total_chunks}")

    def process_all_chunks(self, clear_first: bool = True):
        """Main function to embed all chunks and upsert to backends"""
        logger.info("=" * 60)
        logger.info("🚀 Unified Embedding & Database Pipeline")
        logger.info("=" * 60)

        start_time = time.time()

        # Initialize backends
        if USE_QDRANT:
            self.setup_qdrant()
        if USE_PGVECTOR:
            self.setup_pgvector()

        # Clear existing data if requested
        if clear_first:
            self.clear_databases()
            # Re-initialize after clearing
            if USE_QDRANT:
                self.setup_qdrant()
            if USE_PGVECTOR:
                self.setup_pgvector()

        # Process chunks in batches
        batch = []
        total_processed = 0
        batch_num = 0

        try:
            for record in self.stream_chunks():
                batch.append(record)

                if len(batch) >= self.batch_size:
                    batch_num += 1
                    logger.info(
                        f"📦 Processing batch {batch_num} ({len(batch)} chunks)")

                    self.process_batch(batch)
                    total_processed += len(batch)

                    if self.memory_safe_mode and batch_num % 5 == 0:
                        logger.info(
                            f"🧹 Memory cleanup after {total_processed} chunks")
                        gc.collect()

                    batch = []

            # Process final partial batch
            if batch:
                batch_num += 1
                logger.info(
                    f"📦 Processing final batch {batch_num} ({len(batch)} chunks)")
                self.process_batch(batch)
                total_processed += len(batch)

        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return False

        # Final statistics
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 Embedding & Database Pipeline Complete!")
        logger.info(f"📊 Total chunks processed: {total_processed}")
        logger.info(f"⏱️  Processing time: {elapsed:.1f} seconds")
        logger.info(f"🔥 Throughput: {total_processed/elapsed:.1f} chunks/sec")

        # Verify final counts
        self._verify_final_counts()

        return True

    def _verify_final_counts(self):
        """Verify final database counts"""
        logger.info("\n📊 Verifying final database counts...")

        if USE_QDRANT and self.qdrant_client:
            try:
                collection_info = self.qdrant_client.get_collection(
                    QDRANT_COLLECTION)
                count = collection_info.points_count
                logger.info(f"📊 Qdrant final count: {count}")
            except Exception as e:
                logger.error(f"Failed to get Qdrant count: {e}")

        if USE_PGVECTOR and self.pg_connection:
            try:
                cur = self.pg_connection.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE};")
                count = cur.fetchone()[0]
                logger.info(f"📊 PostgreSQL final count: {count}")
            except Exception as e:
                logger.error(f"Failed to get PostgreSQL count: {e}")

    def cleanup(self):
        """Clean up resources"""
        if self.pg_connection:
            self.pg_connection.close()
            self.pg_connection = None

        if self.model:
            del self.model
            self.model = None

        gc.collect()


def embed_and_upsert_all(memory_safe: bool = True, clear_first: bool = True, batch_size: int = None):
    """
    Convenience function for backward compatibility

    Args:
        memory_safe: Use memory-safe processing (smaller batches, more GC)
        clear_first: Clear databases before processing
        batch_size: Override default batch size
    """
    processor = UnifiedEmbeddingProcessor(
        memory_safe_mode=memory_safe,
        batch_size=batch_size
    )

    try:
        success = processor.process_all_chunks(clear_first=clear_first)
        return success
    finally:
        processor.cleanup()


def safe_embed_and_upsert():
    """Backward compatibility function for memory-safe processing"""
    return embed_and_upsert_all(memory_safe=True, clear_first=True, batch_size=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Embedding & Database Pipeline")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear databases first")
    parser.add_argument("--memory-safe", action="store_true",
                        default=True, help="Use memory-safe processing")
    parser.add_argument("--batch-size", type=int, help="Override batch size")

    args = parser.parse_args()

    success = embed_and_upsert_all(
        memory_safe=args.memory_safe,
        clear_first=not args.no_clear,
        batch_size=args.batch_size
    )

    if success:
        logger.info("✅ Pipeline completed successfully")
    else:
        logger.error("❌ Pipeline failed")
        exit(1)
