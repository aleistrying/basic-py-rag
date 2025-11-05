from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
COLLECTION = "docs_qdrant"


def search_qdrant(query_emb, k=5, where=None):
    res = client.search(collection_name=COLLECTION,
                        query_vector=query_emb, limit=k, query_filter=where)
    return [
        {
            "score": r.score,
            # Try 'text' first, fallback to 'content'
            "content": r.payload.get("text", r.payload.get("content", "")),
            "path": r.payload.get("path"),
            "doc_id": r.payload.get("doc_id"),
            "chunk_id": r.payload.get("chunk_id"),
        }
        for r in res
    ]
