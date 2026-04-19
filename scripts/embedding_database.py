#!/usr/bin/env python3
"""
Unified Embedding and Database Operations Module
Consolidates embed_and_upsert.py, embed_safe.py, and related database functionality
Supports both Qdrant and PostgreSQL with memory-efficient processing
"""

from ingest_config import (
    CLEAN_DIR, EMBED_MODEL,
    USE_QDRANT, USE_PGVECTOR, BATCH_SIZE, LARGE_BATCH_SIZE,
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE, QDRANT_DISTANCE,
    QDRANT_LOCAL_PATH,
    PG_DSN, PG_TABLE, PG_DIM,
    E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
)
import json
import logging
import gc
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
import numpy as np

# Configure logging - REDUCED noise for multi-algorithm processing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce HTTP request logging noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

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
    Supports multiple algorithm combinations simultaneously
    """

    # Define all possible algorithm combinations
    DISTANCE_METRICS = ["cosine", "euclidean", "dot_product", "manhattan"]
    INDEX_ALGORITHMS = ["hnsw", "ivfflat", "scalar_quantization", "exact"]

    def __init__(self, memory_safe_mode: bool = True, batch_size: int = None, large_docs: bool = False,
                 distance_metric: str = "cosine", index_algorithm: str = "hnsw",
                 process_all_combinations: bool = False):
        self.memory_safe_mode = memory_safe_mode
        self.distance_metric = distance_metric
        self.index_algorithm = index_algorithm
        self.process_all_combinations = process_all_combinations

        # Generate all algorithm combinations if requested
        if process_all_combinations:
            self.algorithm_combinations = [
                (dm, ia) for dm in self.DISTANCE_METRICS for ia in self.INDEX_ALGORITHMS
            ]
            logger.info(
                f"🔄 Will process ALL {len(self.algorithm_combinations)} algorithm combinations")
        else:
            self.algorithm_combinations = [(distance_metric, index_algorithm)]
            logger.info(
                f"🎯 Will process single combination: {distance_metric}_{index_algorithm}")

        # Use larger batches for better performance
        if large_docs:
            self.batch_size = batch_size or LARGE_BATCH_SIZE
        else:
            self.batch_size = batch_size or (
                BATCH_SIZE // 2 if memory_safe_mode else BATCH_SIZE)

        # Reduce memory cleanup frequency for larger batches and faster processing
        self.cleanup_interval = max(
            8, self.batch_size // 4)  # Less frequent cleanup

        self.model = None
        self.qdrant_client = None
        self.pg_connection = None

        logger.info(
            f"Initialized processor - Memory safe: {memory_safe_mode}, Batch size: {self.batch_size}, Cleanup every: {self.cleanup_interval} batches")

        for dm, ia in self.algorithm_combinations:
            logger.info(
                f"Algorithm config - Distance: {dm}, Index: {ia}, Suffix: _{dm}_{ia}")

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
        """Initialize Qdrant client and algorithm-specific collections for all combinations"""
        if not USE_QDRANT:
            return None

        if self.qdrant_client is None:
            if QdrantClient is None:
                raise ImportError("qdrant-client not installed")

            if QDRANT_LOCAL_PATH:
                import os as _os
                _os.makedirs(QDRANT_LOCAL_PATH, exist_ok=True)
                logger.info(f"🗂️  Qdrant local file mode: {QDRANT_LOCAL_PATH}")
                self.qdrant_client = QdrantClient(path=QDRANT_LOCAL_PATH)
            else:
                logger.info(f"🔗 Connecting to Qdrant at {QDRANT_URL}")
                self.qdrant_client = QdrantClient(url=QDRANT_URL)

        # Create collections for all algorithm combinations
        distance_mapping = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot_product": Distance.DOT,
            "manhattan": Distance.MANHATTAN
        }

        logger.info(
            f"📋 Setting up {len(self.algorithm_combinations)} Qdrant collections...")
        created_count = 0
        existing_count = 0

        for distance_metric, index_algorithm in self.algorithm_combinations:
            collection_suffix = f"_{distance_metric}_{index_algorithm}"
            collection_name = QDRANT_COLLECTION + collection_suffix

            try:
                collection_info = self.qdrant_client.get_collection(
                    collection_name)
                existing_count += 1
            except:
                distance = distance_mapping.get(
                    distance_metric, Distance.COSINE)
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=QDRANT_VECTOR_SIZE,
                        distance=distance
                    )
                )
                created_count += 1

        # Summary logging instead of individual collection logs
        if created_count > 0:
            logger.info(f"✅ Created {created_count} new Qdrant collections")
        if existing_count > 0:
            logger.info(
                f"♻️  Using {existing_count} existing Qdrant collections")

        return self.qdrant_client

    def setup_pgvector(self):
        """Initialize PostgreSQL algorithm-specific tables for all combinations"""
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

        # Create tables for all algorithm combinations
        distance_operator_map = {
            "cosine": "vector_cosine_ops",
            "euclidean": "vector_l2_ops",
            "dot_product": "vector_ip_ops",
            "manhattan": "vector_l1_ops"  # Note: not all PostgreSQL versions support this
        }

        logger.info(
            f"📋 Setting up {len(self.algorithm_combinations)} PostgreSQL tables...")
        created_count = 0
        existing_count = 0
        warnings_count = 0

        for distance_metric, index_algorithm in self.algorithm_combinations:
            table_suffix = f"_{distance_metric}_{index_algorithm}"
            table_name = PG_TABLE + table_suffix

            # Check if algorithm-specific table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name,))

            table_exists = cur.fetchone()[0]

            if not table_exists:
                cur.execute(f'''
                    CREATE TABLE {table_name} (
                        id SERIAL PRIMARY KEY,
                        source_path TEXT,
                        page INTEGER,
                        chunk_id INTEGER,
                        content TEXT,
                        metadata JSONB,
                        embedding vector({PG_DIM}),
                        distance_metric TEXT,
                        index_algorithm TEXT
                    );
                ''')

                # Create algorithm-appropriate index
                operator = distance_operator_map.get(
                    distance_metric, "vector_cosine_ops")

                try:
                    # Create index based on algorithm
                    if index_algorithm == "ivfflat":
                        cur.execute(f'''
                            CREATE INDEX {table_name}_emb_idx 
                            ON {table_name} USING ivfflat (embedding {operator}) WITH (lists = 100);
                        ''')
                    elif index_algorithm == "hnsw":
                        cur.execute(f'''
                            CREATE INDEX {table_name}_emb_idx 
                            ON {table_name} USING hnsw (embedding {operator});
                        ''')
                    else:
                        # For scalar_quantization and exact, use btree or default
                        cur.execute(f'''
                            CREATE INDEX {table_name}_emb_idx 
                            ON {table_name} USING hnsw (embedding {operator});
                        ''')

                    created_count += 1
                except Exception as e:
                    logger.warning(
                        f"⚠️  Index creation warning for {table_name}: {e}")
                    warnings_count += 1
                    created_count += 1
            else:
                existing_count += 1

        # Summary logging instead of individual table logs
        if created_count > 0:
            logger.info(f"✅ Created {created_count} new PostgreSQL tables")
        if existing_count > 0:
            logger.info(
                f"♻️  Using {existing_count} existing PostgreSQL tables")
        if warnings_count > 0:
            logger.info(
                f"⚠️  {warnings_count} tables had index creation warnings")

        return self.pg_connection

    def clear_databases(self):
        """Clear existing data from both databases - ALL algorithm-specific collections/tables"""
        logger.info(
            "🧹 Clearing existing data from ALL algorithm combinations...")

        if USE_QDRANT:
            client = self.setup_qdrant()
            if client:
                # Clear ALL algorithm-specific collections
                for distance_metric, index_algorithm in self.algorithm_combinations:
                    collection_suffix = f"_{distance_metric}_{index_algorithm}"
                    collection_name = QDRANT_COLLECTION + collection_suffix

                    try:
                        client.delete_collection(collection_name)
                        # Count cleared collections instead of logging each one
                    except:
                        pass  # Collection didn't exist

                # Also clear base collection if it exists
                try:
                    client.delete_collection(QDRANT_COLLECTION)
                    # Count cleared collections instead of logging each one
                except:
                    pass  # Base collection didn't exist

        if USE_PGVECTOR:
            conn = self.setup_pgvector()
            if conn:
                cur = conn.cursor()

                # Clear ALL algorithm-specific tables
                for distance_metric, index_algorithm in self.algorithm_combinations:
                    table_suffix = f"_{distance_metric}_{index_algorithm}"
                    table_name = PG_TABLE + table_suffix

                    try:
                        cur.execute(
                            f'DROP TABLE IF EXISTS {table_name} CASCADE;')
                        # Count cleared tables instead of logging each one
                    except Exception as e:
                        logger.warning(
                            f"Failed to clear table {table_name}: {e}")

                # Also clear base table if it exists
                try:
                    cur.execute(f'DROP TABLE IF EXISTS {PG_TABLE} CASCADE;')
                    # Count cleared tables instead of logging each one
                except:
                    pass  # Base table didn't exist

        logger.info("✅ All database clearing complete!")

    def upsert_batch_to_qdrant(self, batch_records: List[Dict], embeddings: np.ndarray):
        """Upsert batch to Qdrant collection(s) with algorithm-specific naming"""
        client = self.setup_qdrant()
        if not client:
            return

        # Get combinations to process
        combinations = self.algorithm_combinations if self.process_all_combinations else [
            (self.distance_metric, self.index_algorithm)]

        try:
            for distance_metric, index_algorithm in combinations:
                collection_suffix = f"_{distance_metric}_{index_algorithm}"
                collection_name = QDRANT_COLLECTION + collection_suffix

                points = []
                for i, (record, embedding) in enumerate(zip(batch_records, embeddings)):
                    point_id = f"{record['source_path']}_{record.get('page', 0)}_{record.get('chunk_id', i)}"
                    # Create a hash-based ID to avoid collisions - ensure positive integer
                    point_id = abs(hash(point_id)) % (2**31 - 1)

                    _meta = record.get("metadata", {})
                    point = PointStruct(
                        id=point_id,  # Now using integer directly
                        vector=embedding.tolist(),
                        payload={
                            "source_path": record["source_path"],
                            "page": record.get("page", 0),
                            "chunk_id": record.get("chunk_id", i),
                            "content": record["content"],
                            "metadata": _meta,
                            # Section metadata (present for structured markdown files)
                            "section_id": record.get("section_id", _meta.get("section_id", "")),
                            "section_title": record.get("section_title", _meta.get("section_title", "")),
                            "distance_metric": distance_metric,
                            "index_algorithm": index_algorithm
                        }
                    )
                    points.append(point)

                client.upsert(collection_name=collection_name, points=points)
                # Only log every 10th batch to reduce noise
                if len(self.algorithm_combinations) == 1 or distance_metric == "cosine":
                    logger.debug(
                        f"📤 Qdrant: {len(points)} points upserted to {collection_name}")

        except Exception as e:
            logger.error(f"❌ Qdrant upsert failed: {e}")
            raise

    def upsert_batch_to_pgvector(self, batch_records: List[Dict], embeddings: np.ndarray):
        """Upsert batch to PostgreSQL table(s) with algorithm-specific naming"""
        conn = self.setup_pgvector()
        if not conn:
            return

        # Get combinations to process
        combinations = self.algorithm_combinations if self.process_all_combinations else [
            (self.distance_metric, self.index_algorithm)]

        try:
            cur = conn.cursor()

            for distance_metric, index_algorithm in combinations:
                table_suffix = f"_{distance_metric}_{index_algorithm}"
                table_name = PG_TABLE + table_suffix

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
                        embedding.tolist(),
                        distance_metric,
                        index_algorithm
                    ))

                # Batch insert with algorithm metadata
                insert_query = f'''
                    INSERT INTO {table_name} (source_path, page, chunk_id, content, metadata, embedding, distance_metric, index_algorithm)
                    VALUES %s
                '''

                psycopg2.extras.execute_values(
                    cur, insert_query, insert_data,
                    template=None, page_size=self.batch_size
                )

                # Only log every 10th batch to reduce noise
                if len(self.algorithm_combinations) == 1 or distance_metric == "cosine":
                    logger.debug(
                        f"📤 PostgreSQL: {len(insert_data)} rows inserted into {table_name}")

        except Exception as e:
            logger.error(f"❌ PostgreSQL upsert failed: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ PostgreSQL upsert failed: {e}")
            raise

    def process_batch(self, batch_records: List[Dict]):
        """Process a batch of records: embed and upsert to backends with parallel DB operations"""
        if not batch_records:
            return

        # Extract texts for embedding — prepend section context when available
        # so that "S06 — Renvoi tarification carbone" is present in every chunk
        # from that section even if the body text doesn't repeat the title.
        texts = []
        for record in batch_records:
            meta = record.get("metadata", {}) or {}
            sid = meta.get("section_id") or record.get("section_id", "")
            stitle = meta.get("section_title") or record.get(
                "section_title", "")
            prefix = f"[{sid}: {stitle}] " if sid and stitle else ""
            texts.append(prefix + record["content"])

        # Generate embeddings (GPU-accelerated)
        logger.debug(f"🧠 Embedding batch of {len(texts)} texts...")
        embeddings = self.embed_texts(texts, mode="passage")

        # PARALLEL database upserts using threading for I/O operations
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            # Submit both database operations simultaneously
            if USE_QDRANT:
                future_qdrant = executor.submit(
                    self._safe_upsert_qdrant, batch_records, embeddings)
                futures.append(("Qdrant", future_qdrant))

            if USE_PGVECTOR:
                future_postgres = executor.submit(
                    self._safe_upsert_postgres, batch_records, embeddings)
                futures.append(("PostgreSQL", future_postgres))

            # Wait for completion and handle errors
            for db_name, future in futures:
                try:
                    future.result()  # Wait for completion
                except Exception as e:
                    logger.error(f"❌ {db_name} upsert failed: {e}")
                    # Continue with other database - don't fail entire batch

        # Memory cleanup in safe mode
        if self.memory_safe_mode:
            del embeddings

    def _safe_upsert_qdrant(self, batch_records: List[Dict], embeddings: np.ndarray):
        """Thread-safe Qdrant upsert wrapper"""
        try:
            self.upsert_batch_to_qdrant(batch_records, embeddings)
        except Exception as e:
            logger.error(f"❌ Qdrant upsert failed: {e}")
            raise

    def _safe_upsert_postgres(self, batch_records: List[Dict], embeddings: np.ndarray):
        """Thread-safe PostgreSQL upsert wrapper"""
        try:
            self.upsert_batch_to_pgvector(batch_records, embeddings)
        except Exception as e:
            logger.error(f"❌ PostgreSQL upsert failed: {e}")
            raise
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
                        f"📦 Processing batch {batch_num} ({len(batch)} chunks, {total_processed} processed so far)")

                    self.process_batch(batch)
                    total_processed += len(batch)

                    # Less frequent memory cleanup for better performance
                    if self.memory_safe_mode and batch_num % self.cleanup_interval == 0:
                        logger.info(
                            f"🧹 Memory cleanup after {total_processed:,} chunks")
                        gc.collect()

                    batch = []

            # Process final partial batch
            if batch:
                batch_num += 1
                logger.info(
                    f"📦 Processing final batch {batch_num} ({len(batch)} chunks, {total_processed + len(batch)} total)")
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

    def process_chunks_file(self, chunks_file_path: Path, clear_first: bool = False):
        """Process a single chunks file and add to vector databases"""
        logger.info(
            f"🔄 Processing single chunks file: {chunks_file_path.name}")

        start_time = time.time()

        # Initialize backends
        if USE_QDRANT:
            self.setup_qdrant()
        if USE_PGVECTOR:
            self.setup_pgvector()

        # Clear existing data if requested (usually False for uploads)
        if clear_first:
            self.clear_databases()
            if USE_QDRANT:
                self.setup_qdrant()
            if USE_PGVECTOR:
                self.setup_pgvector()

        # Process chunks in batches
        batch = []
        total_processed = 0
        batch_num = 0

        try:
            # Stream chunks from the single file
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            batch.append(record)

                            if len(batch) >= self.batch_size:
                                batch_num += 1
                                logger.info(
                                    f"📦 Processing batch {batch_num} ({len(batch)} chunks)")

                                self.process_batch(batch)
                                total_processed += len(batch)
                                batch = []

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Skipping invalid JSON line in {chunks_file_path.name}")
                            continue

                # Process final partial batch
                if batch:
                    batch_num += 1
                    logger.info(
                        f"📦 Processing final batch {batch_num} ({len(batch)} chunks)")
                    self.process_batch(batch)
                    total_processed += len(batch)

            elapsed = time.time() - start_time
            logger.info(f"✅ Single file processing complete!")
            logger.info(f"📊 Total chunks processed: {total_processed}")
            logger.info(f"⏱️  Processing time: {elapsed:.1f} seconds")

            return True

        except Exception as e:
            logger.error(f"❌ Single file processing failed: {e}")
            return False

    def _verify_final_counts(self):
        """Verify final database counts"""
        logger.info("\n📊 Verifying final database counts...")

        if USE_QDRANT and self.qdrant_client:
            try:
                # Check the first algorithm configuration as a representative sample
                distance_metric, index_algorithm = self.algorithm_combinations[0]
                sample_collection = f"{QDRANT_COLLECTION}_{distance_metric}_{index_algorithm}"
                collection_info = self.qdrant_client.get_collection(
                    sample_collection)
                count = collection_info.points_count
                logger.info(
                    f"📊 Qdrant sample collection ({sample_collection}): {count} points")
                logger.info(
                    f"📊 Total Qdrant collections created: {len(self.algorithm_combinations)}")
            except Exception as e:
                logger.error(f"Failed to get Qdrant count: {e}")

        if USE_PGVECTOR and self.pg_connection:
            try:
                # Check the first algorithm configuration as a representative sample
                distance_metric, index_algorithm = self.algorithm_combinations[0]
                sample_table = f"{PG_TABLE}_{distance_metric}_{index_algorithm}"
                cur = self.pg_connection.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {sample_table};")
                count = cur.fetchone()[0]
                logger.info(
                    f"📊 PostgreSQL sample table ({sample_table}): {count} rows")
                logger.info(
                    f"📊 Total PostgreSQL tables created: {len(self.algorithm_combinations)}")
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


def embed_and_upsert_all(memory_safe: bool = True, clear_first: bool = True, batch_size: int = None, large_docs: bool = False, process_all_combinations: bool = False):
    """
    Main function to embed and upsert all chunks - OPTIMIZED VERSION

    Args:
        memory_safe: Use memory-safe processing (smaller batches, more GC)
        clear_first: Clear databases before processing
        batch_size: Override default batch size
        large_docs: Use larger batch sizes for big documents (recommended for medical textbooks)
        process_all_combinations: Process all 16 algorithm combinations simultaneously
    """
    processor = UnifiedEmbeddingProcessor(
        memory_safe_mode=memory_safe,
        batch_size=batch_size,
        large_docs=large_docs,
        process_all_combinations=process_all_combinations
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
