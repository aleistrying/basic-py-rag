from qdrant_client import QdrantClient
import os
import logging
import threading

logger = logging.getLogger(__name__)

# Default collection names (try clean pipeline first, fallback to legacy)
CLEAN_COLLECTION = "course_docs_clean"

# Lazy-initialised client — avoids crashing at import time when Qdrant is
# not yet reachable (e.g. container still starting, or local/offline mode).
_client: QdrantClient | None = None
_client_lock = threading.Lock()


def _get_client() -> QdrantClient:
    """Return the shared Qdrant client, creating it on first call (thread-safe)."""
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        # Re-check inside the lock to avoid double-init race
        if _client is not None:
            return _client

        local_path = os.getenv("QDRANT_LOCAL_PATH")
        if local_path:
            # Local file-based storage — no server needed (great for macOS offline use)
            os.makedirs(local_path, exist_ok=True)
            logger.info(f"Qdrant: using local file storage at {local_path}")
            _client = QdrantClient(path=local_path)
        else:
            host = os.getenv("QDRANT_HOST", "localhost")
            logger.info(f"Qdrant: connecting to {host}:6333")
            _client = QdrantClient(host=host, port=6333)

    return _client


def close_client() -> None:
    """Close and reset the shared Qdrant client.

    Call this before spawning a subprocess that opens the same local-file
    storage path, otherwise both processes will hold conflicting locks.
    The client is lazily re-created on the next call to _get_client().
    """
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None
            logger.info(
                "Qdrant: client connection closed (will reopen on next request)")


def get_available_collection() -> str:
    """Determine which collection to use - prefer algorithm-specific collections."""
    client = _get_client()
    default_collection = f"{CLEAN_COLLECTION}_cosine_hnsw"

    try:
        client.get_collection(default_collection)
        return default_collection
    except Exception:
        pass

    try:
        collections = client.get_collections()
        for collection in collections.collections:
            if collection.name.startswith(CLEAN_COLLECTION + "_"):
                return collection.name
    except Exception as exc:
        logger.warning(f"Qdrant get_collections failed: {exc}")

    return CLEAN_COLLECTION


def search_qdrant(query_emb, k=5, where=None, collection_suffix=None):
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
    client = _get_client()
    collection = get_available_collection()

    # Use algorithm-specific collection if suffix provided
    if collection_suffix:
        algorithm_collection = f"{CLEAN_COLLECTION}_{collection_suffix}"
        try:
            client.get_collection(algorithm_collection)
            collection = algorithm_collection
        except Exception:
            pass  # Fall back to default collection

    try:
        from qdrant_client.models import QueryRequest
        res = client.query_points(
            collection_name=collection,
            query=query_emb,
            limit=k,
            query_filter=where
        ).points

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
