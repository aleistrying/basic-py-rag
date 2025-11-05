# 🐘 PostgreSQL + pgvector

## ⚡ Quick Setup

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table (our schema uses vector(768) for E5 embeddings)
CREATE TABLE docs (
    id SERIAL PRIMARY KEY,
    doc_id TEXT,
    chunk_id TEXT,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(768)
);

-- Create index for cosine similarity (our default)
CREATE INDEX docs_cosine_idx ON docs USING hnsw (embedding vector_cosine_ops);
```

## 🔍 Vector Search Queries

### Generate Working Search Commands

```bash
# Generate SQL with real embeddings
python generate_search_examples.py "bases de datos vectoriales" 3
```

### Connect to Database

```bash
# Using psql directly
PGPASSWORD=pgpass psql -h localhost -U pguser -d vectordb

# Or with docker
docker exec -it pgvector_db psql -U pguser -d vectordb
```

### Cosine Similarity Search (Working Example)

```bash
# From command line - generate first:
python generate_search_examples.py "bases de datos NoSQL" 3

# Or manually with a saved embedding:
PGPASSWORD=pgpass psql -h localhost -U pguser -d vectordb -c "
SELECT
    LEFT(content, 100) as preview,
    metadata->>'path' as document,
    1 - (embedding <=> '[0.024, 0.038, ...]'::vector) as similarity
FROM docs
ORDER BY embedding <=> '[0.024, 0.038, ...]'::vector
LIMIT 3;
"
```

### Quick Test with Existing Data

```sql
-- Get a sample vector from the database and search with it
WITH sample_vector AS (
    SELECT embedding FROM docs LIMIT 1
)
SELECT
    LEFT(content, 80) as preview,
    metadata->>'path' as document,
    1 - (embedding <=> (SELECT embedding FROM sample_vector)) as similarity
FROM docs
ORDER BY embedding <=> (SELECT embedding FROM sample_vector)
LIMIT 5;
```

### With Metadata Filter

```sql
-- Search only in PDF documents
WITH query AS (
    SELECT '[0.024, 0.038, ...]'::vector(768) AS v
)
SELECT
    LEFT(content, 100) as preview,
    metadata->>'path' as document,
    1 - (embedding <=> query.v) as similarity
FROM docs, query
WHERE metadata->>'path' LIKE '%.pdf'
ORDER BY embedding <=> query.v
LIMIT 3;

-- Search documents with schedule data
SELECT
    LEFT(content, 100) as preview,
    metadata->>'path' as document,
    metadata ? 'schedule' as has_schedule,
    1 - (embedding <=> '[...]'::vector(768)) as similarity
FROM docs
WHERE metadata ? 'schedule'
ORDER BY embedding <=> '[...]'::vector(768)
LIMIT 2;
```

## 🔧 Essential Operations

### Check Index Usage

```sql
EXPLAIN ANALYZE
SELECT doc_id FROM docs
ORDER BY embedding <=> '[0.01,...]'::vector(768)
LIMIT 5;
-- Look for "Index Scan using docs_cosine_idx"
```

### Browse Data

```sql
-- See all documents
SELECT
    id,
    doc_id,
    chunk_id,
    LEFT(content, 50) as preview,
    metadata->>'path' as document
FROM docs
LIMIT 10;

-- Count chunks per document
SELECT
    metadata->>'path' as document,
    COUNT(*) as chunks
FROM docs
GROUP BY metadata->>'path';

-- View schedule data
SELECT
    metadata->>'path' as document,
    LEFT(metadata->>'schedule', 200) as schedule_preview
FROM docs
WHERE metadata ? 'schedule'
LIMIT 1;
```

### Update/Delete

```sql
-- Update vector
UPDATE docs
SET embedding = '[...]'::vector(768)
WHERE doc_id = 'doc1';

-- Delete document chunks
DELETE FROM docs WHERE doc_id = 'doc1';

-- Delete by path
DELETE FROM docs WHERE metadata->>'path' LIKE '%.txt';
```

## 📋 Key Notes

**Operators & Indices:**

- **Cosine Distance**: `<=>` operator + `vector_cosine_ops` index (our default) ✅
- **L2 Distance**: `<->` operator + `vector_l2_ops` index
- **Inner Product**: `<#>` operator + `vector_ip_ops` index
- **Dimension**: Must match model (768 for E5-base)

**Schema:**

- `content` - Document text
- `metadata` - JSONB with `path` and optional `schedule`
- `embedding` - 768-dimensional E5 vector

**Performance Tips:**

- Use HNSW index for fast approximate search
- Adjust `hnsw.ef_search` for accuracy vs speed tradeoff
- Use metadata filters after vector search for best performance
