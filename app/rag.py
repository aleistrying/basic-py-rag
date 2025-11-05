from typing import List
import logging
from scripts.utils import embed_e5, embed_texts, expand_query  # Import query expansion

# Also import clean pipeline embedder
try:
    from scripts.query_embed import embed_query as embed_query_clean
except ImportError:
    embed_query_clean = None

from app.qdrant_backend import search_qdrant
from app.pgvector_backend import search_pgvector

try:
    from app.rerank import mmr, build_context as build_context_enhanced
except ImportError:
    mmr = None
    build_context_enhanced = None

try:
    import ollama
except ImportError:
    ollama = None

BACKENDS = {"qdrant": search_qdrant, "pgvector": search_pgvector}

# Improved prompt: extractive but tolerant (allows paraphrasing)
PROMPT_TEMPLATE = """Eres un asistente académico. Responde SOLO con información contenida de forma explícita en los fragmentos.
Si la respuesta no aparece literal o inequívocamente, responde EXACTAMENTE: "No está disponible.".
Incluye citas (path:chunk) al final de CADA oración afirmativa.

Pregunta: {q}

Fragmentos (cada línea es un fragmento):
{sources}

Reglas:
- Parafrasea ligeramente si hace falta, pero no inventes nombres/fechas/datos.
- Si hay varios fragmentos, combina de forma concisa con varias citas.
- Si la pregunta es ambigua, responde "No está disponible.".
Respuesta:
"""


def build_context(hits: List[dict]) -> str:
    lines = []
    for h in hits:
        lines.append(f"- ({h['path']}:{h['chunk_id']}) {h['content']}")
    return "\n".join(lines)

# Nota: Para la demo se puede imprimir el contexto; la generación con LLM es opcional


def rag_answer(query: str, backend: str = "qdrant", k: int = 5):
    # Validate backend name
    if backend not in BACKENDS:
        available_backends = list(BACKENDS.keys())
        raise ValueError(
            f"Backend '{backend}' not supported. Available backends: {available_backends}")

    # Apply query expansion for better Spanish query matching
    expanded_query = expand_query(query)

    # Use clean pipeline embedder if available, fallback to legacy
    if embed_query_clean is not None:
        try:
            emb = embed_query_clean(expanded_query)
        except Exception as e:
            print(f"Clean embedder failed, using fallback: {e}")
            emb = embed_e5([expanded_query], is_query=True)[0]
    else:
        emb = embed_e5([expanded_query], is_query=True)[0]

    hits = BACKENDS[backend](emb, k=k)

    # Create a presentation-friendly response
    results = []
    for hit in hits:
        content = hit.get('content', '') or ''  # Handle None content
        results.append({
            "document": hit['path'].replace('./data/raw/', ''),
            "similarity": f"{hit['score']:.3f}",
            "preview": content[:120] + "..." if len(content) > 120 else content
        })

    return {
        "query": query,
        "expanded_query": expanded_query if expanded_query != query else None,
        "backend": backend.upper(),
        "total_results": len(results),
        "results": results
    }


def generate_llm_answer(query: str, backend: str = "qdrant", k: int = 5, model: str = "gemma2:2b"):
    """Generate AI response using RAG + LLM with improved reranking and query expansion"""
    if ollama is None:
        raise ImportError(
            "Ollama package not installed. Install with: pip install ollama")

    # Validate backend name
    if backend not in BACKENDS:
        available_backends = list(BACKENDS.keys())
        raise ValueError(
            f"Backend '{backend}' not supported. Available backends: {available_backends}")

    # Apply query expansion for better retrieval
    expanded_query = expand_query(query)

    # Use clean pipeline embedder if available, fallback to legacy
    if embed_query_clean is not None:
        try:
            emb = embed_query_clean(expanded_query)
        except Exception as e:
            print(f"Clean embedder failed, using fallback: {e}")
            emb = embed_e5([expanded_query], is_query=True)[0]
    else:
        emb = embed_e5([expanded_query], is_query=True)[0]

    # Get more results for reranking (12-16 results for better coverage)
    hits = BACKENDS[backend](emb, k=max(k*3, 12))

    # Prepare candidates for reranking
    candidates = []
    for hit in hits:
        candidates.append({
            'content': hit['content'],
            'sim': min(max(hit['score'], 0.0), 1.0),  # Normalize score to 0-1
            'path': hit['path'],
            'chunk_id': hit.get('chunk_id', hit.get('page', 'unknown'))
        })

    # Apply MMR reranking if available
    if mmr and len(candidates) > k:
        reranked = mmr(query, candidates, lambda_=0.7, top_k=k)
    else:
        reranked = candidates[:k]

    # Create results list
    results = []
    for item in reranked:
        content = item.get('content', '') or ''  # Handle None content
        results.append({
            "document": item['path'].replace('./data/raw/', ''),
            "similarity": f"{item['sim']:.3f}",
            "preview": content[:120] + "..." if len(content) > 120 else content
        })

    # Build context for LLM using enhanced method if available
    if build_context_enhanced:
        sources = build_context_enhanced(reranked, max_tokens=1200)
    else:
        sources = build_context(reranked)
    prompt = PROMPT_TEMPLATE.format(q=query, sources=sources)

    try:
        # Configure Ollama client
        client = ollama.Client(host='http://localhost:11434')

        # Call Ollama API with stricter settings
        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.0,  # Zero temperature for deterministic answers
                "repeat_penalty": 1.0,
                "num_predict": 256,
                "top_p": 0.9
            }
        )

        # Extract the generated text
        generated_text = response.get('response', '').strip()

        return {
            "query": query,
            "expanded_query": expanded_query if expanded_query != query else None,
            "backend": backend.upper(),
            "model": model,
            "ai_response": generated_text,
            "total_results": len(results),
            "sources": results,
            "prompt_used": prompt
        }

    except Exception as e:
        logging.error(f"Ollama API error: {e}")
        # Fallback to RAG-only response if LLM fails
        return {
            "query": query,
            "expanded_query": expanded_query if expanded_query != query else None,
            "backend": backend.upper(),
            "model": model,
            "ai_response": f"❌ LLM Error: {str(e)}. Here are the related documents:",
            "total_results": len(results),
            "sources": results,
            "error": str(e),
            "fallback_mode": True
        }
