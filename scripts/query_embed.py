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

from ingest_config import EMBED_MODEL, E5_QUERY_PREFIX, E5_PASSAGE_PREFIX

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


def expand_query(query: str) -> str:
    """
    Expand Spanish queries with synonyms and common variations
    to improve retrieval on short queries.

    Args:
        query: Original search query

    Returns:
        Expanded query with relevant synonyms
    """
    expansion_dict = {
        "nube": ["cloud", "clouds", "plataforma en la nube", "Azure", "AWS", "MongoDB Atlas"],
        "nubes": ["cloud", "clouds", "plataformas en la nube", "Azure", "AWS", "MongoDB Atlas"],
        "entrega": ["fecha de entrega", "fecha límite", "deadline"],
        "evaluación": ["puntaje", "porcentaje", "evaluaciones", "nota"],
        "evaluaciones": ["puntaje", "porcentaje", "evaluación", "nota"],
        "examen": ["evaluación", "prueba", "test"],
        "horario": ["cronograma", "calendario", "fechas"],
        "profesor": ["docente", "instructor", "maestro"],
        "clase": ["curso", "materia", "asignatura"],
        "hora": ["tiempo", "horario", "schedule"],
    }

    ql = query.lower()
    extra_terms = []

    for key, synonyms in expansion_dict.items():
        if key in ql:
            extra_terms.extend(synonyms)

    if extra_terms:
        # Add top 2-3 most relevant synonyms to avoid query bloat
        unique_terms = list(set(extra_terms))[:3]
        return query + " " + " ".join(unique_terms)

    return query


def embed_e5(texts: Union[List[str], str], is_query: bool = False) -> Union[np.ndarray, List[float]]:
    """
    E5 embeddings with proper prefixes for better Spanish support.

    Args:
        texts: Single string or list of strings to embed
        is_query: True for query mode (adds query prefix), False for passage mode

    Returns:
        Normalized embeddings (768 dimensions for e5-base)
        - List[float] if single text provided
        - np.ndarray if list of texts provided
    """
    model = get_model()

    # Handle single string input
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    # Add appropriate prefix
    prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
    prefixed = [prefix + t.strip() for t in texts]

    # Encode with normalization for cosine similarity
    vecs = model.encode(prefixed, normalize_embeddings=True)

    # Return format matching input
    if single_input:
        return vecs[0].tolist()
    return vecs.tolist()


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
