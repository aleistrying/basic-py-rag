from typing import List
import logging
import os

# Import clean pipeline embedder and query utilities (consolidated)
try:
    from scripts.query_embed import embed_query as embed_query_clean, expand_query, embed_e5
except ImportError:
    print("Warning: query_embed module not available")
    print("Install: pip install sentence-transformers")
    embed_e5 = None
    def expand_query(x): return x
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

# Improved prompt: concise, focused answers without excessive citations
PROMPT_TEMPLATE = """Eres un asistente académico. Responde de forma directa y concisa usando SOLO la información de los fragmentos.

Pregunta: {q}

Fragmentos relevantes:
{sources}

Instrucciones:
- Da una respuesta clara y directa
- Usa fechas, números y nombres exactos de los fragmentos
- Si la información no está en los fragmentos, responde: "No está disponible."
- Solo menciona el documento si es relevante para entender la respuesta
- NO incluyas referencias bibliográficas completas ni autores irrelevantes
- Responde en español, de forma natural y concisa

Respuesta:
"""


def build_context(hits: List[dict]) -> str:
    lines = []
    for h in hits:
        # Extract page and metadata information
        page_info = ""
        if h.get('page'):
            page_info = f":página{h['page']}"
        elif h.get('chunk_id'):
            page_info = f":{h['chunk_id']}"

        # Extract document name (cleaner format)
        doc_name = h['path'].replace('./data/raw/', '').replace('.pdf', '')

        # Add quality and extraction info for better context
        metadata = h.get('metadata', {})
        quality_info = ""
        if metadata.get('quality_score', 0) > 0.7:
            quality_info = " [Alta calidad]"
        elif metadata.get('extractor') == 'ocr':
            quality_info = " [Extraído con OCR]"

        lines.append(f"- ({doc_name}{page_info}{quality_info}) {h['content']}")
    return "\n".join(lines)

# Nota: Para la demo se puede imprimir el contexto; la generación con LLM es opcional


def rag_answer(query: str, backend: str = "qdrant", k: int = 5, filters: dict = None):
    # Validate backend name
    if backend not in BACKENDS:
        available_backends = list(BACKENDS.keys())
        raise ValueError(
            f"Backend '{backend}' not supported. Available backends: {available_backends}")

    # Apply query expansion for better Spanish query matching
    expanded_query = expand_query(query)

    # Use clean pipeline embedder if available, fallback to e5 directly
    if embed_query_clean is not None:
        try:
            emb = embed_query_clean(expanded_query)
        except Exception as e:
            print(f"Clean embedder failed, using fallback: {e}")
            emb = embed_e5([expanded_query], is_query=True)[0]
    else:
        emb = embed_e5([expanded_query], is_query=True)[0]

    # Pass filters to the backend search function
    if backend == "qdrant":
        hits = BACKENDS[backend](emb, k=k, where=filters)
    else:  # pgvector
        hits = BACKENDS[backend](emb, k=k, where=filters)

    # Create a presentation-friendly response
    results = []
    for hit in hits:
        content = hit.get('content', '') or ''  # Handle None content

        # Enhanced document information
        doc_name = hit['path'].replace('./data/raw/', '').replace('.pdf', '')

        # Build enhanced reference with page/section info
        reference_parts = [doc_name]
        if hit.get('page'):
            reference_parts.append(f"página {hit['page']}")
        elif hit.get('chunk_id'):
            reference_parts.append(f"sección {hit['chunk_id']}")

        # Look for chapter information in content
        chapter_info = ""
        section_info = ""
        if 'CAPÍTULO' in content[:200]:
            import re
            chapter_match = re.search(r'CAPÍTULO\s+(\d+)', content[:200])
            if chapter_match:
                chapter_info = f", Capítulo {chapter_match.group(1)}"

        # Look for section headers
        if any(keyword in content[:150] for keyword in ['OBJETIVO', 'CRONOGRAMA', 'EVALUACIÓN', 'METODOLOGÍA']):
            import re
            section_match = re.search(
                r'(OBJETIVO[S]?|CRONOGRAMA|EVALUACIÓN|METODOLOGÍA)[:\s]', content[:150])
            if section_match:
                section_info = f", {section_match.group(1).capitalize()}"

        reference = " - ".join(reference_parts) + chapter_info + section_info

        # Extract metadata for better context
        metadata = hit.get('metadata', {})
        quality_score = metadata.get('quality_score', 0)
        extractor_used = metadata.get('extractor', 'unknown')

        results.append({
            "document": hit['path'].replace('./data/raw/', ''),
            "reference": reference,
            "page": hit.get('page'),
            "chapter": chapter_info.replace(", ", "") if chapter_info else None,
            "section": section_info.replace(", ", "") if section_info else None,
            "similarity": f"{hit['score']:.3f}",
            "quality": f"{quality_score:.2f}" if quality_score > 0 else "N/A",
            "extractor": extractor_used,
            "preview": content[:120] + "..." if len(content) > 120 else content
        })

    return {
        "query": query,
        "expanded_query": expanded_query if expanded_query != query else None,
        "backend": backend.upper(),
        "filters_applied": filters or {},
        "total_results": len(results),
        "results": results
    }


def generate_llm_answer(query: str, backend: str = "qdrant", k: int = 5, model: str = "gemma2:2b", filters: dict = None):
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

    # Use clean pipeline embedder if available, fallback to e5 directly
    if embed_query_clean is not None:
        try:
            emb = embed_query_clean(expanded_query)
        except Exception as e:
            print(f"Clean embedder failed, using fallback: {e}")
            emb = embed_e5([expanded_query], is_query=True)[0]
    else:
        emb = embed_e5([expanded_query], is_query=True)[0]

    # Get more results for reranking (12-16 results for better coverage)
    if backend == "qdrant":
        hits = BACKENDS[backend](emb, k=max(k*3, 12), where=filters)
    else:  # pgvector
        hits = BACKENDS[backend](emb, k=max(k*3, 12), where=filters)

    # Prepare candidates for reranking
    candidates = []
    for hit in hits:
        candidates.append({
            'content': hit['content'],
            'sim': min(max(hit['score'], 0.0), 1.0),  # Normalize score to 0-1
            'path': hit['path'],
            'chunk_id': hit.get('chunk_id', hit.get('page', 'unknown')),
            'page': hit.get('page'),
            'metadata': hit.get('metadata', {})
        })

    # Apply MMR reranking if available
    if mmr and len(candidates) > k:
        reranked = mmr(query, candidates, lambda_=0.7, top_k=k)
    else:
        reranked = candidates[:k]

    # Create enhanced results list with page/section references
    results = []
    for item in reranked:
        content = item.get('content', '') or ''  # Handle None content

        # Enhanced document information
        doc_name = item['path'].replace('./data/raw/', '').replace('.pdf', '')

        # Build enhanced reference with page/section info
        reference_parts = [doc_name]
        if item.get('page'):
            reference_parts.append(f"página {item['page']}")
        elif item.get('chunk_id'):
            reference_parts.append(f"sección {item['chunk_id']}")

        # Look for chapter and topic information in content
        chapter_info = ""
        topic_info = ""
        section_info = ""

        # Extract chapter information
        if 'CAPÍTULO' in content[:200]:
            import re
            chapter_match = re.search(r'CAPÍTULO\s+(\d+)', content[:200])
            if chapter_match:
                chapter_info = f"Cap. {chapter_match.group(1)}"

        # Extract topic/section information from common patterns
        content_lower = content[:300].lower()
        if 'vacuna' in content_lower or 'inmunización' in content_lower:
            topic_info = "Vacunación"
        elif 'diabetes' in content_lower:
            topic_info = "Diabetes"
        elif 'hipertensión' in content_lower or 'presión arterial' in content_lower:
            topic_info = "Hipertensión"
        elif 'parto' in content_lower or 'trabajo de parto' in content_lower:
            topic_info = "Parto"
        elif 'embarazo' in content_lower or 'gestación' in content_lower:
            topic_info = "Embarazo"
        elif 'feto' in content_lower or 'fetal' in content_lower:
            topic_info = "Desarrollo Fetal"
        elif 'complicación' in content_lower:
            topic_info = "Complicaciones"

        # Build enhanced reference
        ref_parts = []
        if item.get('page'):
            ref_parts.append(f"p.{item['page']}")
        if chapter_info:
            ref_parts.append(chapter_info)
        if topic_info:
            ref_parts.append(topic_info)

        enhanced_reference = " | ".join(
            ref_parts) if ref_parts else "Sin contexto específico"

        results.append({
            "document": item['path'].replace('./data/raw/', '').replace('.pdf', ''),
            "reference": enhanced_reference,
            "page": item.get('page'),
            "chapter": chapter_info if chapter_info else None,
            "topic": topic_info if topic_info else None,
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
        # Configure Ollama client with Docker service name support
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        # Call Ollama API with stricter settings
        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "repeat_penalty": 1.0,
                "temperature": 0.5,
                "num_predict": 256,
                "top_p": 0.9
            }
        )

        # Extract the generated text
        generated_text = response.get('response', '').strip()

        # Build enhanced source attribution footer
        source_references = []
        for i, result in enumerate(results[:3], 1):  # Show top 3 sources
            ref_parts = []
            if result.get('page'):
                ref_parts.append(f"p.{result['page']}")
            if result.get('section'):
                ref_parts.append(result['section'])
            if result.get('chapter'):
                ref_parts.append(result['chapter'])

            ref_suffix = f" ({', '.join(ref_parts)})" if ref_parts else ""
            source_references.append(f"{i}. {result['document']}{ref_suffix}")

        # Add enhanced footer to the response
        if source_references:
            sources_text = "\n".join(source_references)
            enhanced_response = f"{generated_text}\n\n\n<b>Fuentes consultadas:</b>\n{sources_text}"
        else:
            enhanced_response = generated_text

        return {
            "query": query,
            "expanded_query": expanded_query if expanded_query != query else None,
            "backend": backend.upper(),
            "model": model,
            "filters_applied": filters or {},
            "ai_response": enhanced_response,
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
            "filters_applied": filters or {},
            "ai_response": f"❌ LLM Error: {str(e)}. Here are the related documents:",
            "total_results": len(results),
            "sources": results,
            "error": str(e),
            "fallback_mode": True
        }
