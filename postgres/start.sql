-- Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 768 dimensions for multilingual-e5-base
-- (e5-large is also 768; old MiniLM was 384)
CREATE TABLE IF NOT EXISTS docs (
  id bigserial PRIMARY KEY,
  doc_id text NOT NULL,
  chunk_id text NOT NULL,
  content text NOT NULL,
  metadata jsonb,
  embedding vector(768) NOT NULL
);

-- Create cosine similarity index for ANN search
-- Using IVFFlat with cosine distance for better Spanish query matching
CREATE INDEX IF NOT EXISTS docs_embedding_cos_ivfflat
ON docs USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 200);

-- Alternative: HNSW index (if pgvector >= 0.7 and you prefer HNSW)
-- CREATE INDEX IF NOT EXISTS docs_embedding_cos_hnsw
-- ON docs USING hnsw (embedding vector_cosine_ops); 
