"""
Query helper that ensures proper E5 query prefixes for consistent retrieval.
Use this for embedding queries before searching Qdrant/pgvector.
"""
import re as _re
import unicodedata
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

            # Force CPU mode for embedding model to avoid GPU memory competition with Ollama
            device = "cpu"

            # Clear any GPU cache before loading
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    if not hasattr(get_model, '_already_printed'):
                        print("🧹 Cleared GPU cache to preserve memory for Ollama")
                except Exception:
                    pass

            # Only print during model loading, not during normal usage
            if not hasattr(get_model, '_already_printed'):
                print(
                    f"🔧 Loading embedding model on: {device} (Preserving GPU for Ollama)")
                get_model._already_printed = True

            # Docker-safe PyTorch configuration
            torch.set_default_dtype(torch.float32)

            # Environment setup for containers
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

            # Load model with device optimization - keep on CPU to avoid GPU memory conflict
            _model = SentenceTransformer(
                EMBED_MODEL,
                device="cpu",
                trust_remote_code=False
            )

            if device == "cpu":
                if not hasattr(get_model, '_already_printed'):
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


def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace — for matching only, never for embedding."""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return _re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Case packs: triggered by a specific case / statute mention.
# Each pack returns 2-3 focused variant queries (not one bloated query).
# ---------------------------------------------------------------------------
_CASE_PACKS: list[tuple[list[str], list[str]]] = [
    # Carbon pricing reference (FR + EN triggers)
    (
        ["tarification du carbone", "tarification carbone", "carbon pricing",
         "ggppa", "ltpges", "greenhouse gas pollution pricing"],
        [
            "{q} Renvoi tarification carbone LTPGES",
            "{q} double aspect fédéralisme coopératif",
            "{q} POGG intérêt national",
        ],
    ),
    # Trans Mountain / paramountcy / Burnaby — keep isolated from carbon
    (
        ["trans mountain", "burnaby", "city of burnaby", "trans-mountain"],
        [
            "{q} prépondérance fédérale paramountcy",
            "{q} entreprise fédérale réglementation municipale",
        ],
    ),
    # Secession reference
    (
        ["sécession", "secession", "renvoi sécession", "québec"],
        [
            "{q} Renvoi sécession Québec principes constitutionnels",
            "{q} fédéralisme démocratie constitutionnalisme état de droit",
        ],
    ),
    # Senate reference
    (
        ["sénat", "senate reform", "réforme du sénat"],
        [
            "{q} Renvoi réforme du Sénat",
            "{q} fédéralisme procédure de modification",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Doctrine buckets: mutually exclusive — expanding "chevauchement" must NOT
# also inject "applicabilité" or "prépondérance", which belong to different
# doctrines and pollute retrieval.
# ---------------------------------------------------------------------------
_DOCTRINE_BUCKETS: list[tuple[list[str], list[str]]] = [
    # Validity / classification
    (
        ["caract", "validit", "pith", "substance", "classification"],
        ["caractère véritable", "pith and substance", "validité constitutionnelle",
         "objet véritable"],
    ),
    # Double aspect / overlap (chevauchement)
    (
        ["chevauchement", "double aspect", "coexistence", "concurren"],
        ["double aspect", "fédéralisme coopératif", "coexistence"],
    ),
    # Paramountcy / operability (prépondérance)
    (
        ["preponderan", "prepondéran", "paramountcy", "operabilit", "operabilit",
         "conflit", "conflict", "incompatibilit"],
        ["prépondérance fédérale", "paramountcy",
            "opérabilité", "conflit de lois"],
    ),
    # Interjurisdictional immunity / applicability
    (
        ["applicabilit", "immunite", "immunity", "interjuridiction",
         "entreprise fedérale", "federal undertaking"],
        ["immunité interjuridictionnelle", "interjurisdictional immunity",
         "entreprise fédérale", "applicabilité"],
    ),
    # POGG / national concern
    (
        ["pogg", "pobg", "interet national", "national concern", "pouvoir residuaire",
         "residual power"],
        ["POGG", "POBG", "intérêt national", "national concern",
         "pouvoir résiduaire"],
    ),
    # Cooperative federalism / division of powers (generic)
    (
        ["federalism", "federalisme", "partage", "division of powers",
         "repartition", "pouvoir federal", "pouvoir provincial"],
        ["fédéralisme coopératif", "partage des compétences",
         "pouvoirs fédéraux", "pouvoirs provinciaux"],
    ),
    # Criminal law head of power
    (
        ["droit penal", "criminal law",
            "91(27)", "prohibition", "sanction penale"],
        ["droit pénal", "criminal law", "91(27)", "objectif prohibitif"],
    ),
    # Renvoi / advisory opinion (generic)
    (
        ["renvoi", "reference", "avis consultatif", "advisory opinion"],
        ["renvoi avis consultatif", "procédure de renvoi"],
    ),
]

# ---------------------------------------------------------------------------
# Bilingual bridging: single-term synonyms (FR ↔ EN), no doctrine mixing.
# Only used for the primary/fallback single-string path.
# ---------------------------------------------------------------------------
_BRIDGE: dict[str, str] = {
    "federalism": "fédéralisme",
    "fédéralisme": "federalism",
    "paramountcy": "prépondérance fédérale",
    "prépondérance": "paramountcy",
    "immunity": "immunité interjuridictionnelle",
    "immunité": "interjurisdictional immunity",
    "validity": "validité constitutionnelle",
    "validité": "constitutional validity",
    "pith and substance": "caractère véritable",
    "caractère véritable": "pith and substance",
    "national concern": "intérêt national",
    "intérêt national": "national concern",
    # Spanish course terms
    "nube": "cloud platform",
    "entrega": "deadline fecha de entrega",
    "evaluación": "nota puntaje",
    "horario": "cronograma calendario",
    "profesor": "docente instructor",
}


def build_queries(query: str, max_variants: int = 4) -> list[str]:
    """
    Build a ranked list of focused retrieval queries from a single user query.

    Strategy:
    1. Check case packs first — if the query names a specific case/statute,
       return 2-3 targeted variants specific to that case.
    2. Check doctrine buckets — expand within the matching doctrine only,
       keeping doctrines isolated from each other.
    3. Fall back to bilingual bridging for generic vocabulary.

    Always returns the original query as the first element.
    Never returns more than `max_variants` queries total.
    """
    qn = _normalize(query)
    queries: list[str] = [query]

    # ── 1. Case packs (highest priority, most precise) ──────────────────────
    for triggers, templates in _CASE_PACKS:
        if any(_normalize(t) in qn for t in triggers):
            for tmpl in templates:
                v = tmpl.format(q=query)
                if v not in queries:
                    queries.append(v)
            break  # only one case pack — stop after first match

    if len(queries) > 1:
        return queries[:max_variants]

    # ── 2. Doctrine buckets (medium priority, doctrine-isolated) ─────────────
    matched_bucket: list[str] | None = None
    for triggers, terms in _DOCTRINE_BUCKETS:
        if any(t in qn for t in triggers):
            matched_bucket = terms
            break  # only one bucket — first match wins

    if matched_bucket:
        # Build one variant that appends the doctrine-specific terms
        doctrine_suffix = " ".join(matched_bucket[:3])
        v = query + " " + doctrine_suffix
        if v not in queries:
            queries.append(v)
        # Add a second variant with the remaining terms if any
        if len(matched_bucket) > 3:
            v2 = query + " " + " ".join(matched_bucket[3:])
            if v2 not in queries:
                queries.append(v2)

    if len(queries) > 1:
        return queries[:max_variants]

    # ── 3. Bilingual bridge (fallback) ───────────────────────────────────────
    extras: list[str] = []
    for key, synonym in _BRIDGE.items():
        if _normalize(key) in qn:
            extras.append(synonym)
    if extras:
        v = query + " " + " ".join(dict.fromkeys(extras))
        queries.append(v)

    return queries[:max_variants]


def expand_query(query: str) -> str:
    """
    Return a single expanded query string for backward-compatible callers.

    Uses the same doctrine-aware logic as build_queries() but collapses the
    result to one string (joining only the first variant's extra terms, not
    all variants concatenated).

    For multi-query search, prefer build_queries() directly.
    """
    variants = build_queries(query, max_variants=2)
    if len(variants) == 1:
        return query
    # The second variant already has the right suffix appended; use it as-is
    # so we don't double-append.  But we only want the *extra* terms, not the
    # full variant (which repeats the original query).  Trim the original prefix.
    second = variants[1]
    if second.startswith(query):
        extra = second[len(query):].strip()
        # Keep at most 5 extra tokens to avoid bloating the embedding
        extra_tokens = extra.split()[:5]
        return query + (" " + " ".join(extra_tokens) if extra_tokens else "")
    return second


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
