# 🎯 Qdrant - Vector Database

## ⚡ Quick Commands

### List Collections

```bash
curl -s http://localhost:6333/collections | jq
```

### Vector Search

**Generate a working search command:**

```bash
# First, generate search commands with a real embedding
python generate_search_examples.py "bases de datos vectoriales" 3
```

**Manual search example:**

```bash
# Note: Use the generate_search_examples.py script above for real embeddings
# The vector must be a 768-dimensional array from E5 embeddings

curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.024, 0.038, 0.0004, ...768 values total...],
    "limit": 3
  }' | jq '.result[] | {score, path:.payload.path, preview:.payload.text[:100]}'
```

**Quick test with actual data:**

```bash
# Get the first point's vector and search with it
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/scroll" \
  -H 'Content-Type: application/json' \
  -d '{"limit": 1, "with_payload": true, "with_vectors": true}' \
  | jq -r '.result.points[0].vector' > test_vector.json

# Use it for search (with_payload: true is important!)
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"vector\": $(cat test_vector.json),
    \"limit\": 3,
    \"with_payload\": true
  }" | jq '.result[] | {score, path:.payload.path, text:(.payload.text | .[0:120])}'
```

### Search with Metadata Filter

```bash
# Search only in PDF documents
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"vector\": $(cat test_vector.json),
    \"limit\": 3,
    \"filter\": {
      \"must\": [{
        \"key\": \"path\",
        \"match\": {\"text\": \".pdf\"}
      }]
    }
  }" | jq '.result[] | {score, path:.payload.path}'

# Search only documents with schedule data
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/search" \
  -H 'Content-Type: application/json' \
  -d "{
    \"vector\": $(cat test_vector.json),
    \"limit\": 2,
    \"filter\": {
      \"must\": [{
        \"key\": \"schedule\",
        \"match\": {\"any\": [\"PROGRAMACIÓN\"]}
      }]
    }
  }" | jq '.result[] | {score, path:.payload.path, has_schedule:(.payload.schedule != null)}'
```

## 🔧 Data Management

### Browse Documents

```bash
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/scroll" \
  -H 'Content-Type: application/json' \
  -d '{"limit": 5, "with_payload": true, "with_vectors": false}' | jq
```

### Insert Points

```bash
curl -s -X PUT "http://localhost:6333/collections/docs_qdrant/points" \
  -H 'Content-Type: application/json' \
  -d '{
    "points": [{
      "id": "test-point-1",
      "vector": [0.1, 0.2, 0.3, /* ...768 values total... */],
      "payload": {
        "path": "test_doc.txt",
        "text": "This is the content of the document",
        "doc_id": "test-doc-id",
        "chunk_id": "0"
      }
    }]
  }'
```

**Note:** Our schema uses `"text"` for content (not `"content"`).

### Delete Points

```bash
curl -s -X POST "http://localhost:6333/collections/docs_qdrant/points/delete" \
  -H 'Content-Type: application/json' \
  -d '{"points": [1, 2, 3]}'
```

## 📋 Key Notes

**Field Names:**

- `text` - Document content (768-dim E5 embeddings)
- `path` - File path
- `doc_id` - Document identifier
- `chunk_id` - Chunk number within document
- `schedule` - Optional schedule table data (for course guides)

**Distance Metrics:**

- **COSINE**: Best for normalized embeddings (our default) ✅
- **DOT**: Dot product
- **EUCLID**: L2 distance

**Best Practices:**

- Dimension must match exactly (768 for E5-base)
- Use filters to narrow search space
- Qdrant auto-optimizes indices
- Use `generate_search_examples.py` to create real search commands
