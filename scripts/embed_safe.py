#!/usr/bin/env python3
"""
Memory-safe embedding and upsert - processes files one by one with tiny batches.
Prevents OOM crashes on systems with limited RAM.
"""
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ingest_config import (
    CLEAN_DIR, EMBED_MODEL, BATCH_SIZE, MAX_CHUNKS_PER_FILE,
    USE_QDRANT, USE_PGVECTOR,
    QDRANT_URL, QDRANT_COLLECTION, QDRANT_VECTOR_SIZE, QDRANT_DISTANCE,
    PG_DSN, PG_TABLE,
    E5_PASSAGE_PREFIX
)


def safe_embed_and_upsert():
    """Memory-safe version - processes one file at a time with tiny batches"""
    print("🧠 Memory-Safe Embedding & Upsert")
    print("=" * 50)

    # Load embedding model once
    if SentenceTransformer is None:
        print("❌ sentence-transformers not installed")
        return False

    print(f"🤖 Loading embedding model: {EMBED_MODEL}")
    try:
        model = SentenceTransformer(EMBED_MODEL)
        print(
            f"✅ Model loaded (dimensions: {model.get_sentence_embedding_dimension()})")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Setup backends
    qdrant_client = None
    pg_conn = None

    if USE_QDRANT:
        print(f"🔗 Setting up Qdrant...")
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            qdrant_client = QdrantClient(url=QDRANT_URL)

            # Recreate collection for clean start
            try:
                qdrant_client.delete_collection(QDRANT_COLLECTION)
            except:
                pass

            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=QDRANT_VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Qdrant collection '{QDRANT_COLLECTION}' ready")

        except Exception as e:
            print(f"❌ Qdrant setup failed: {e}")
            if USE_QDRANT:
                return False

    if USE_PGVECTOR:
        print(f"🔗 Setting up PostgreSQL...")
        try:
            import psycopg2

            pg_conn = psycopg2.connect(PG_DSN)
            pg_conn.autocommit = True
            cur = pg_conn.cursor()

            # Recreate table for clean start
            cur.execute(f'DROP TABLE IF EXISTS {PG_TABLE};')
            cur.execute(f'''
                CREATE TABLE {PG_TABLE} (
                    id SERIAL PRIMARY KEY,
                    source_path TEXT,
                    page INTEGER,
                    chunk_id INTEGER,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(768)
                );
            ''')

            # Create index
            cur.execute(f'''
                CREATE INDEX {PG_TABLE}_emb_cosine_idx 
                ON {PG_TABLE} USING hnsw (embedding vector_cosine_ops);
            ''')

            print(f"✅ PostgreSQL table '{PG_TABLE}' ready")

        except Exception as e:
            print(f"❌ PostgreSQL setup failed: {e}")
            if USE_PGVECTOR:
                return False

    # Find chunk files
    chunk_files = list(Path(CLEAN_DIR).glob("*.chunks.jsonl"))
    if not chunk_files:
        print("❌ No chunk files found")
        return False

    print(f"📄 Processing {len(chunk_files)} chunk files...")

    total_processed = 0

    # Process each file separately to save memory
    for file_idx, chunk_file in enumerate(chunk_files, 1):
        print(
            f"\n📄 [{file_idx}/{len(chunk_files)}] Processing: {chunk_file.name}")

        try:
            # Read chunks from this file only
            file_chunks = []
            with open(chunk_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > MAX_CHUNKS_PER_FILE:
                        print(
                            f"⚠️  Limiting to {MAX_CHUNKS_PER_FILE} chunks per file (memory safety)")
                        break

                    try:
                        chunk_data = json.loads(line.strip())
                        if chunk_data.get('content', '').strip():
                            file_chunks.append(chunk_data)
                    except:
                        continue

            if not file_chunks:
                print(f"   ⚠️  No valid chunks found")
                continue

            print(f"   📊 Found {len(file_chunks)} chunks")

            # Process in tiny batches to avoid memory issues
            file_processed = 0
            batch_num = 0

            for i in range(0, len(file_chunks), BATCH_SIZE):
                batch = file_chunks[i:i + BATCH_SIZE]
                batch_num += 1

                print(
                    f"   🔄 Batch {batch_num}: {len(batch)} chunks", end="", flush=True)

                try:
                    # Extract texts for embedding
                    texts = [E5_PASSAGE_PREFIX + chunk['content']
                             for chunk in batch]

                    # Generate embeddings
                    embeddings = model.encode(
                        texts, normalize_embeddings=True, show_progress_bar=False)

                    # Upsert to backends
                    for chunk_data, embedding in zip(batch, embeddings):

                        # Qdrant
                        if qdrant_client:
                            try:
                                from qdrant_client.http.models import PointStruct
                                import uuid

                                point = PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=embedding.tolist(),
                                    payload={
                                        "source_path": chunk_data.get("source_path", ""),
                                        "page": chunk_data.get("page", 0),
                                        "chunk_id": chunk_data.get("chunk_id", 0),
                                        "content": chunk_data["content"],
                                        "metadata": chunk_data.get("metadata", {})
                                    }
                                )

                                qdrant_client.upsert(
                                    collection_name=QDRANT_COLLECTION,
                                    points=[point]
                                )

                            except Exception as e:
                                print(f"\n   ❌ Qdrant upsert error: {e}")

                        # PostgreSQL
                        if pg_conn:
                            try:
                                cur = pg_conn.cursor()
                                cur.execute(f'''
                                    INSERT INTO {PG_TABLE} 
                                    (source_path, page, chunk_id, content, metadata, embedding)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                ''', (
                                    chunk_data.get("source_path", ""),
                                    chunk_data.get("page", 0),
                                    chunk_data.get("chunk_id", 0),
                                    chunk_data["content"],
                                    json.dumps(chunk_data.get("metadata", {})),
                                    embedding.tolist()
                                ))

                            except Exception as e:
                                print(f"\n   ❌ PostgreSQL upsert error: {e}")

                    file_processed += len(batch)
                    print(f" ✅")

                    # Force garbage collection after each batch
                    del embeddings, texts, batch
                    gc.collect()

                except Exception as e:
                    print(f"\n   ❌ Batch {batch_num} failed: {e}")
                    continue

            print(f"   ✅ File complete: {file_processed} chunks processed")
            total_processed += file_processed

            # Clear file chunks from memory
            del file_chunks
            gc.collect()

        except Exception as e:
            print(f"   ❌ File processing failed: {e}")
            continue

    # Cleanup
    if pg_conn:
        pg_conn.close()

    print(f"\n🎉 Memory-safe processing complete!")
    print(f"📊 Total chunks processed: {total_processed}")

    return True


if __name__ == "__main__":
    safe_embed_and_upsert()
