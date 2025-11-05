import os
import uuid
import json
import sys
import psycopg2
from utils import read_texts, chunk_text, embed_e5

conn = psycopg2.connect(
    dbname="vectordb", user="pguser", password="pgpass", host="localhost", port=5432
)
conn.autocommit = True
cur = conn.cursor()

# Reset database: Clear existing data and recreate schema
print("🔄 Resetting PostgreSQL database...")

# Drop and recreate table with correct dimensions (768 for e5-base)
cur.execute('DROP TABLE IF EXISTS docs;')
print("✅ Dropped existing docs table")

# Enable pgvector extension
cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
print("✅ Enabled pgvector extension")

cur.execute('''
CREATE TABLE docs (
    id SERIAL PRIMARY KEY,
    doc_id TEXT,
    chunk_id TEXT,
    content TEXT,
    metadata JSONB,
    embedding vector(768)  -- E5-base has 768 dimensions
);
''')
print("✅ Created fresh docs table with vector(768)")

# Create cosine similarity index (IVFFlat)
cur.execute('''
CREATE INDEX IF NOT EXISTS docs_embedding_cos_ivfflat
ON docs USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);
''')
print("✅ Created cosine similarity index (IVFFlat)")

# Get folder from command line or use default
folder = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
items = read_texts(folder)

total_chunks = 0
for item in items:
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(item["text"]) or [item["text"]]
    # Use E5 embeddings with passage prefix
    embs = embed_e5(chunks, is_query=False)
    for idx, (chunk, emb) in enumerate(zip(chunks, embs)):
        # Prepare metadata
        metadata = {"path": item["path"]}
        if "schedule" in item:
            metadata["schedule"] = item["schedule"]

        cur.execute(
            """ 
            INSERT INTO docs (doc_id, chunk_id, content, metadata, embedding) 
            VALUES (%s, %s, %s, %s, %s) 
            """,
            (doc_id, f"{idx}", chunk, json.dumps(metadata), emb)
        )
        total_chunks += 1

print(f"✅ Ingested {total_chunks} chunks to PostgreSQL")
print(f"✅ Using multilingual-e5-base (768 dimensions) with cosine similarity")
