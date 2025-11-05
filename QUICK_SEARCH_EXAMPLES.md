# 🔍 Quick Search Examples

Working examples for manual vector search in both databases.

## 🎯 Qdrant - Quick Search

### 1. Extract a test vector from your data:

```bash
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/scroll" \
  -H 'Content-Type: application/json' \
  -d '{"limit": 1, "with_payload": true, "with_vectors": true}' \
  | jq -r '.result.points[0].vector' > test_vector.json

echo "✅ Saved test vector ($(wc -l < test_vector.json) dimensions)"
```

### 2. Search with that vector:

```bash
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"vector\": $(cat test_vector.json),
    \"limit\": 3,
    \"with_payload\": true
  }" | jq '.result[] | {
    score,
    document: (.payload.path | split("/")[-1]),
    preview: (.payload.text | .[0:100])
  }'
```

**Expected output:**

```json
{
  "score": 1,
  "document": "curso_objetivos_clean.txt",
  "preview": "Sexta Edición, 2014. RAMAKRISHNAN; GERKE..."
}
```

### 3. Search with a filter (PDF files only):

```bash
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"vector\": $(cat test_vector.json),
    \"limit\": 2,
    \"with_payload\": true,
    \"filter\": {
      \"must\": [{
        \"key\": \"path\",
        \"match\": {\"text\": \".pdf\"}
      }]
    }
  }" | jq '.result[] | {
    score,
    document: (.payload.path | split("/")[-1])
  }'
```

## 🐘 PostgreSQL - Quick Search

### 1. Connect to database:

```bash
PGPASSWORD=pgpass psql -h localhost -U pguser -d vectordb
```

### 2. Search using a vector from the database:

```sql
-- Simple search
WITH sample_vector AS (
    SELECT embedding FROM docs LIMIT 1
)
SELECT
    LEFT(content, 70) as preview,
    metadata->>'path' as path,
    ROUND((1 - (embedding <=> (SELECT embedding FROM sample_vector)))::numeric, 3) as similarity
FROM docs
ORDER BY embedding <=> (SELECT embedding FROM sample_vector)
LIMIT 3;
```

### 3. Count documents and chunks:

```sql
-- Total chunks
SELECT COUNT(*) as total_chunks FROM docs;

-- Chunks per document
SELECT
    metadata->>'path' as document,
    COUNT(*) as chunks
FROM docs
GROUP BY metadata->>'path';
```

### 4. View schedule data:

```sql
SELECT
    metadata->>'path' as document,
    LEFT(metadata->>'schedule', 150) as schedule_preview
FROM docs
WHERE metadata ? 'schedule'
LIMIT 1;
```

### 5. Search with metadata filter:

```sql
-- Search only in PDF documents
WITH sample_vector AS (SELECT embedding FROM docs LIMIT 1)
SELECT
    LEFT(content, 60) as preview,
    metadata->>'path' as path,
    ROUND((1 - (embedding <=> (SELECT embedding FROM sample_vector)))::numeric, 3) as sim
FROM docs
WHERE metadata->>'path' LIKE '%.pdf'
ORDER BY embedding <=> (SELECT embedding FROM sample_vector)
LIMIT 2;
```

## 🎯 Generate Custom Searches

Use the helper script to generate searches with real embeddings:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Generate search commands
python generate_search_examples.py "bases de datos vectoriales" 3
python generate_search_examples.py "NoSQL MongoDB" 2
python generate_search_examples.py "proyecto SQL en la nube" 3
```

This will output both Qdrant curl commands and PostgreSQL SQL queries with real embeddings!

## 📊 Quick Stats

### Qdrant:

```bash
# Collection info
curl -s http://localhost:6333/collections/docs_qdrant | jq '.result'

# Count points
curl -s http://localhost:6333/collections/docs_qdrant | jq '.result.points_count'
```

### PostgreSQL:

```sql
-- Table stats
SELECT
    COUNT(*) as total_chunks,
    COUNT(DISTINCT doc_id) as total_documents,
    pg_size_pretty(pg_total_relation_size('docs')) as table_size
FROM docs;
```

## ✅ Verification

After running searches, you should see:

- ✅ Scores between 0 and 1 (higher is more similar)
- ✅ Document paths from `data/raw/`
- ✅ Text previews showing actual content
- ✅ Fast response times (< 100ms for small datasets)
