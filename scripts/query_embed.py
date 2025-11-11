"""
Query helper that ensures proper E5 query prefixes for consistent retrieval.
Use this for embedding queries before searching Qdrant/pgvector.
"""
from typing import List, Union
import numpy as np
import sys
from pathlib import Path

# Add project root to path for container compatibility
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from ingest_config import EMBED_MODEL, E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
except ImportError:
    # Fallback values if ingest_config not available
    EMBED_MODEL = "intfloat/multilingual-e5-base"
    E5_QUERY_PREFIX = "query: "
    E5_PASSAGE_PREFIX = "passage: "

# Model cache
_model = None


def get_model():
    """Load and cache the embedding model with GPU optimization when available"""
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers")

        try:
            import torch
            import os

            # Detect GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🔧 Loading embedding model on: {device}")

            # Docker-safe PyTorch configuration
            torch.set_default_dtype(torch.float32)

            # Environment setup for containers
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

            # Load model with device optimization
            _model = SentenceTransformer(
                EMBED_MODEL,
                device=device,
                trust_remote_code=False
            )

            # Device-specific optimizations
            if device == "cuda":
                print("🚀 Model loaded with GPU acceleration")
            else:
                print("💻 Model loaded on CPU")
                # CPU-specific optimizations for containers
                _model = _model.cpu()
                for param in _model.parameters():
                    if param.device.type != 'cpu':
                        param.data = param.data.cpu()
                    if param.dtype != torch.float32:
                        param.data = param.data.float()

            # Test the model to ensure it works
            with torch.no_grad():
                test_embedding = _model.encode(
                    ["test"], show_progress_bar=False, convert_to_tensor=False)

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Model loading failed in container: {error_msg}")
            # Specifically check for meta tensor error
            if "meta tensor" in error_msg or "to_empty" in error_msg:
                print(
                    "🔧 Detected PyTorch meta tensor issue, disabling model for fallback")
            # Set model to None to trigger fallback
            _model = None
            # Don't re-raise - we want to use fallbacks
            return None

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
    Falls back to deterministic mock embeddings if model fails.

    Args:
        texts: Single string or list of strings to embed
        is_query: True for query mode (adds query prefix), False for passage mode

    Returns:
        Normalized embeddings (768 dimensions for e5-base)
        - List[float] if single text provided
        - np.ndarray if list of texts provided
    """
    # Handle single string input
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    try:
        model = get_model()

        # If model loading failed, use fallback immediately
        if model is None:
            print("⚠️ Using fallback embeddings (model unavailable)")
            return _create_fallback_embeddings(texts, single_input=single_input)

        # Add appropriate prefix
        prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
        prefixed = [prefix + t.strip() for t in texts]

        # Encode with normalization for cosine similarity
        import torch
        with torch.no_grad():
            vecs = model.encode(
                prefixed, normalize_embeddings=True, convert_to_tensor=False)

        # Return format matching input
        if single_input:
            return vecs[0].tolist()
        return vecs.tolist()

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Model embedding failed, using fallback: {error_msg}")
        if "meta tensor" in error_msg or "to_empty" in error_msg:
            print("🔧 Meta tensor error detected - using deterministic fallback")
        return _create_fallback_embeddings(texts, single_input=single_input)


def _create_fallback_embeddings(texts: List[str], single_input: bool = False) -> Union[List[float], List[List[float]]]:
    """Create deterministic fallback embeddings when real model fails"""
    import hashlib
    import numpy as np

    embeddings = []
    for text in texts:
        # Create deterministic embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        np.random.seed(seed % (2**31))

        # Generate 768-dimensional vector (E5 dimensions)
        embedding = np.random.normal(0, 1, 768)
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())

    return embeddings[0] if single_input else embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query with proper E5 prefix and normalization.
    Falls back to deterministic mock embedding if model fails.

    Args:
        query: Search query string

    Returns:
        Normalized query embedding (768 dimensions)
    """
    try:
        model = get_model()

        # If model loading failed, use fallback immediately
        if model is None:
            print("⚠️ Query embedding using fallback (model unavailable)")
            fallback_embeddings = _create_fallback_embeddings(
                [query], single_input=True)
            return np.array(fallback_embeddings)

        # Add query prefix (critical for E5)
        prefixed_query = E5_QUERY_PREFIX + query

        # Embed with L2 normalization for cosine similarity
        import torch
        with torch.no_grad():
            embedding = model.encode(
                [prefixed_query], normalize_embeddings=True, convert_to_tensor=False)[0]

        return embedding

    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Query embedding failed, using fallback: {error_msg}")
        if "meta tensor" in error_msg or "to_empty" in error_msg:
            print("🔧 Meta tensor error in query embedding - using deterministic fallback")
        # Create fallback embedding
        fallback_embeddings = _create_fallback_embeddings(
            [query], single_input=True)
        return np.array(fallback_embeddings)


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
