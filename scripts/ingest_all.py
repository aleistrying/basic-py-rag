#!/usr/bin/env python3
"""
Combined ingestion script for both Qdrant and pgvector databases
Provides detailed progress reporting and error handling
"""

import sys
import time
from pathlib import Path

# Import both ingestion modules
import uuid
import json
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Import utilities
from utils import read_texts, chunk_text, embed_e5

# Configuration
QDRANT_COLLECTION = "docs_qdrant"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DIM = 768  # E5-base dimensions

POSTGRES_CONFIG = {
    "dbname": "vectordb",
    "user": "pguser",
    "password": "pgpass",
    "host": "localhost",
    "port": 5432
}


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num, total_steps, description):
    """Print a step indicator"""
    print(f"\n[{step_num}/{total_steps}] {description}")


def reset_qdrant():
    """Reset Qdrant collection"""
    print_step(1, 4, "🔄 Resetting Qdrant database...")

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Delete existing collection
        try:
            client.delete_collection(collection_name=QDRANT_COLLECTION)
            print(f"    ✅ Deleted existing collection: {QDRANT_COLLECTION}")
        except Exception as e:
            print(f"    ℹ️  Collection doesn't exist yet (OK for first run)")

        # Create fresh collection
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
        )
        print(f"    ✅ Created fresh collection: {QDRANT_COLLECTION}")

        return client
    except Exception as e:
        print(f"    ❌ Error resetting Qdrant: {e}")
        raise


def reset_postgres():
    """Reset PostgreSQL database"""
    print_step(2, 4, "🔄 Resetting PostgreSQL database...")

    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        # Drop and recreate table
        cur.execute('DROP TABLE IF EXISTS docs;')
        print("    ✅ Dropped existing docs table")

        cur.execute('''
        CREATE TABLE docs (
            id SERIAL PRIMARY KEY,
            doc_id TEXT,
            chunk_id TEXT,
            content TEXT,
            metadata JSONB,
            embedding vector(768)
        );
        ''')
        print("    ✅ Created fresh docs table with vector(768)")

        # Create cosine similarity index
        cur.execute('''
        CREATE INDEX IF NOT EXISTS idx_docs_emb_cosine
        ON docs USING hnsw (embedding vector_cosine_ops);
        ''')
        print("    ✅ Created cosine similarity index")

        return conn, cur
    except Exception as e:
        print(f"    ❌ Error resetting PostgreSQL: {e}")
        raise


def process_documents(folder):
    """Process documents and create chunks with embeddings"""
    print_step(3, 4, f"📄 Processing documents from: {folder}")

    # Read all documents
    items = read_texts(folder)
    print(f"    ℹ️  Found {len(items)} documents")

    all_chunks = []
    total_chunks = 0

    for idx, item in enumerate(items, 1):
        doc_id = str(uuid.uuid4())
        doc_name = Path(item["path"]).name

        print(f"\n    [{idx}/{len(items)}] Processing: {doc_name}")

        # Create chunks
        chunks = chunk_text(item["text"]) or [item["text"]]
        print(f"        📊 Created {len(chunks)} chunks")

        # Generate embeddings with progress
        print(f"        🧮 Generating embeddings...")
        embs = embed_e5(chunks, is_query=False)
        print(f"        ✅ Generated {len(embs)} embeddings")

        # Store chunk data
        for chunk_idx, (chunk, emb) in enumerate(zip(chunks, embs)):
            chunk_data = {
                'doc_id': doc_id,
                'chunk_id': f"{chunk_idx}",
                'chunk': chunk,
                'embedding': emb,
                'path': item["path"],
                'metadata': {}
            }

            # Add schedule if present
            if "schedule" in item:
                chunk_data['metadata']['schedule'] = item["schedule"]
                if chunk_idx == 0:
                    print(f"        📅 Schedule data attached")

            all_chunks.append(chunk_data)
            total_chunks += 1

    print(f"\n    ✅ Total chunks processed: {total_chunks}")
    return all_chunks


def ingest_to_qdrant(client, chunks):
    """Ingest chunks to Qdrant"""
    print_step(4, 4, f"💾 Ingesting to Qdrant...")

    points = []
    for chunk_data in chunks:
        payload = {
            "doc_id": chunk_data['doc_id'],
            "chunk_id": chunk_data['chunk_id'],
            "text": chunk_data['chunk'],
            "path": chunk_data['path'],
        }

        # Add schedule if present
        if chunk_data['metadata'].get('schedule'):
            payload["schedule"] = chunk_data['metadata']['schedule']

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk_data['embedding'],
            payload=payload,
        ))

    # Upsert in batches
    BATCH = 128
    batches = (len(points) + BATCH - 1) // BATCH

    for i in range(0, len(points), BATCH):
        batch_num = (i // BATCH) + 1
        print(
            f"    📦 Uploading batch {batch_num}/{batches}...", end='', flush=True)
        client.upsert(collection_name=QDRANT_COLLECTION,
                      points=points[i:i+BATCH])
        print(" ✅")

    print(f"    ✅ Ingested {len(points)} chunks to Qdrant")


def ingest_to_postgres(conn, cur, chunks):
    """Ingest chunks to PostgreSQL"""
    print(f"\n💾 Ingesting to PostgreSQL...")

    for idx, chunk_data in enumerate(chunks, 1):
        if idx % 50 == 0 or idx == 1:
            print(
                f"    📦 Processing chunk {idx}/{len(chunks)}...", end='\r', flush=True)

        # Prepare metadata
        metadata = {"path": chunk_data["path"]}
        if chunk_data['metadata'].get('schedule'):
            metadata["schedule"] = chunk_data['metadata']['schedule']

        cur.execute(
            """
            INSERT INTO docs (doc_id, chunk_id, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                chunk_data['doc_id'],
                chunk_data['chunk_id'],
                chunk_data['chunk'],
                json.dumps(metadata),
                chunk_data['embedding']
            )
        )

    print(f"    ✅ Ingested {len(chunks)} chunks to PostgreSQL" + " " * 20)


def main():
    """Main ingestion workflow"""
    print_header("🚀 COMBINED DATABASE INGESTION")
    print("Target databases: Qdrant + PostgreSQL (pgvector)")

    start_time = time.time()

    # Get folder from command line or use default
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
    print(f"Source folder: {folder}")

    try:
        # Step 1: Reset Qdrant
        qdrant_client = reset_qdrant()

        # Step 2: Reset PostgreSQL
        pg_conn, pg_cur = reset_postgres()

        # Step 3: Process documents
        chunks = process_documents(folder)

        # Step 4: Ingest to both databases
        ingest_to_qdrant(qdrant_client, chunks)
        ingest_to_postgres(pg_conn, pg_cur, chunks)

        # Close connections
        pg_conn.close()

        # Summary
        elapsed = time.time() - start_time
        print_header("✅ INGESTION COMPLETE")
        print(f"  📊 Total chunks: {len(chunks)}")
        print(f"  ⏱️  Time elapsed: {elapsed:.2f} seconds")
        print(f"  📈 Chunks per second: {len(chunks)/elapsed:.2f}")
        print("\n  Databases updated:")
        print(f"    • Qdrant collection: {QDRANT_COLLECTION}")
        print(f"    • PostgreSQL table: docs")
        print("=" * 70 + "\n")

    except Exception as e:
        print_header("❌ INGESTION FAILED")
        print(f"  Error: {e}")
        print("=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
