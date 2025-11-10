from qdrant_client import QdrantClient
import os

# Use Docker service name when running in container, localhost otherwise
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
client = QdrantClient(host=QDRANT_HOST, port=6333)

# Default collection names (try clean pipeline first, fallback to legacy)
CLEAN_COLLECTION = "course_docs_clean"
# LEGACY_COLLECTION = "docs_qdrant"


def get_available_collection():
    """Determine which collection to use"""
    try:
        # Try clean pipeline collection first
        client.get_collection(CLEAN_COLLECTION)
        return CLEAN_COLLECTION
    except:
        return CLEAN_COLLECTION


def search_qdrant(query_emb, k=5, where=None):
    """
    Search Qdrant using the best available collection.
    Supports both clean pipeline and legacy formats.

    Args:
        query_emb: Query embedding vector
        k: Number of results to return
        where: Filter dict, supports:
            - document_type: str (e.g., "pdf", "txt", "md")
            - section: str (e.g., "objetivos", "cronograma", "evaluacion")
            - topic: str (e.g., "nosql", "vectorial", "sql")
            - page: int (for PDFs)
            - contains: str (text must contain this string)
    """
    collection = get_available_collection()

    try:
        res = client.search(
            collection_name=collection,
            query_vector=query_emb,
            limit=k,
            query_filter=where
        )

        results = []
        for r in res:
            # Handle both clean and legacy payload formats
            content = r.payload.get("content", "") or r.payload.get("text", "")
            path = r.payload.get("source_path", r.payload.get("path", ""))

            # Clean pipeline uses 'page' and 'chunk_id', legacy uses 'doc_id'
            chunk_id = r.payload.get(
                "chunk_id", r.payload.get("doc_id", "unknown"))

            results.append({
                "score": r.score,
                "content": content,
                "path": path,
                "chunk_id": chunk_id,
                "doc_id": r.payload.get("doc_id", ""),  # Legacy compatibility
                "page": r.payload.get("page", None),    # Clean pipeline info
                # Clean pipeline metadata
                "metadata": r.payload.get("metadata", {}),
            })

        return results

    except Exception as e:
        print(f"Qdrant search failed: {e}")
        return []
