"""
Query helper that ensures proper E5 query prefixes for consistent retrieval.
Use this for embedding queries before searching Qdrant/pgvector.
"""
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ingest_config import EMBED_MODEL, E5_QUERY_PREFIX

# Model cache
_model = None


def get_model():
    """Load and cache the embedding model"""
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query with proper E5 prefix and normalization.

    Args:
        query: Search query string

    Returns:
        Normalized query embedding (768 dimensions)
    """
    model = get_model()

    # Add query prefix (critical for E5)
    prefixed_query = E5_QUERY_PREFIX + query

    # Embed with L2 normalization for cosine similarity
    embedding = model.encode([prefixed_query], normalize_embeddings=True)[0]

    return embedding


def embed_queries(queries: List[str]) -> np.ndarray:
    """
    Embed multiple queries with proper E5 prefixes.

    Args:
        queries: List of query strings

    Returns:
        Array of normalized embeddings
    """
    model = get_model()

    # Add query prefixes
    prefixed_queries = [E5_QUERY_PREFIX + q for q in queries]

    # Embed with L2 normalization
    embeddings = model.encode(prefixed_queries, normalize_embeddings=True)

    return embeddings


def test_query_embedding():
    """Test the query embedding function"""
    test_queries = [
        "¿Cuáles son las nubes que usaremos?",
        "fechas de entrega",
        "objetivos del curso",
        "¿Qué evaluaciones hay?"
    ]

    print("🧪 Testing query embeddings...")
    print(f"📐 Model: {EMBED_MODEL}")

    for query in test_queries:
        try:
            embedding = embed_query(query)
            print(
                f"✅ '{query}' → {embedding.shape} dims, norm: {np.linalg.norm(embedding):.3f}")
        except Exception as e:
            print(f"❌ '{query}' → Error: {e}")


if __name__ == "__main__":
    test_query_embedding()
