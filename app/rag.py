from app.pgvector_backend import search_pgvector
from app.qdrant_backend import search_qdrant
from typing import List
from pathlib import Path
import logging
import os

# Initialize logger
logger = logging.getLogger(__name__)

# Import clean pipeline embedder and query utilities (consolidated)
try:
    from scripts.query_embed import embed_query as embed_query_clean, expand_query, embed_e5
except ImportError:
    print("Warning: query_embed module not available")
    print("Install: pip install sentence-transformers")
    embed_e5 = None
    def expand_query(x): return x
    embed_query_clean = None


try:
    from app.rerank import mmr, build_context as build_context_enhanced
except ImportError:
    mmr = None
    build_context_enhanced = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    from app.ollama_utils import ollama_generate_with_retry, check_ollama_health
except ImportError:
    ollama_generate_with_retry = None
    check_ollama_health = None

BACKENDS = {"qdrant": search_qdrant, "pgvector": search_pgvector}

# ---------------------------------------------------------------------------
# System prompt — user-editable, stored in data/system_prompt.txt
# ---------------------------------------------------------------------------

# Path to the user-editable system prompt file (relative to repo root)
_SYSTEM_PROMPT_FILE = Path(__file__).parent.parent / \
    "data" / "system_prompt.txt"

# Default prompt used when no custom file exists
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert research assistant specializing in legal and academic analysis. "
    "The user will ask a question and you will be given relevant document excerpts. "
    "Your job is to build a well-reasoned, complete answer using those excerpts as evidence.\n\n"
    "Rules:\n"
    "- Start directly with your answer — no preamble, no 'based on the documents'\n"
    "- Synthesize the information into a coherent argument or explanation, don't just quote\n"
    "- Use specific names, case citations, article numbers, dates, and legal principles from the excerpts\n"
    "- For legal/constitutional questions: identify the legal test, the competing powers or rights, and how the courts resolved the tension\n"
    "- For factual questions: be direct and precise, 2–4 sentences\n"
    "- For analytical/argumentative questions: develop the argument fully, 4–10 sentences, structured logically\n"
    "- If the excerpts are insufficient to fully answer, say what can be answered and what is missing\n"
    "- DO NOT list sources or references — the UI already shows them\n"
    "- DO NOT start with 'Fuentes consultadas' or any source list\n"
    "- Respond in the same language as the question (English, French, or Spanish)"
)


def load_system_prompt() -> str:
    """Return the active system prompt (custom file, or built-in default)."""
    try:
        if _SYSTEM_PROMPT_FILE.exists():
            text = _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception as exc:
        logger.warning(f"Could not read system prompt file: {exc}")
    return DEFAULT_SYSTEM_PROMPT


def save_system_prompt(text: str) -> None:
    """Persist a custom system prompt to disk."""
    _SYSTEM_PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SYSTEM_PROMPT_FILE.write_text(text.strip(), encoding="utf-8")


# Fixed RAG structure — wraps the editable system prompt around context
RAG_TEMPLATE = """{system_prompt}

Question: {q}

Relevant excerpts:
{sources}

Answer (direct, no source list, same language as the question):
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


def extract_document_metadata(filename: str) -> dict:
    """Extract structured metadata from document filename"""
    import re
    metadata = {
        'course': None,
        'theme': None,
        'topic': None,
        'year': None,
        'version': None
    }

    # Extract course information
    if 'BD Avanzadas' in filename or 'SBDA' in filename:
        metadata['course'] = 'Sistemas de Base de Datos Avanzadas'
    elif 'Obstetricia' in filename:
        metadata['course'] = 'Williams Obstetricia'
    elif 'Guia de Curso' in filename:
        metadata['course'] = 'Guía de Curso'

    # Extract theme/topic numbers
    theme_match = re.search(r'Tema\s*(\d+)', filename, re.IGNORECASE)
    if theme_match:
        metadata['theme'] = f"Tema {theme_match.group(1)}"

    # Extract specific topics
    if 'Introduccion' in filename:
        metadata['topic'] = 'Introducción'
    elif 'Modelos' in filename:
        metadata['topic'] = 'Modelos de Sistemas'
    elif 'Presentacion' in filename:
        metadata['topic'] = 'Presentación'
    elif 'Guia de Curso' in filename:
        metadata['topic'] = 'Guía del Curso'

    # Extract year
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        metadata['year'] = year_match.group(1)

    # Extract version
    version_match = re.search(r'[vV](\d+)', filename)
    if version_match:
        metadata['version'] = version_match.group(1)

    return metadata


def extract_content_metadata(content: str) -> dict:
    """Extract metadata from content text"""
    import re
    metadata = {
        'chapter': None,
        'section': None
    }

    # Extract chapter information
    if 'CAPÍTULO' in content[:200]:
        chapter_match = re.search(r'CAPÍTULO\s+(\d+)', content[:200])
        if chapter_match:
            metadata['chapter'] = chapter_match.group(1)
    elif 'TEMA' in content[:100]:
        tema_match = re.search(r'TEMA\s+(\d+)', content[:100])
        if tema_match:
            metadata['chapter'] = tema_match.group(1)

    # Extract section information from common patterns
    content_upper = content[:200].upper()
    if 'OBJETIVO' in content_upper:
        metadata['section'] = 'Objetivos'
    elif 'INTRODUCCIÓN' in content_upper or 'INTRODUCCION' in content_upper:
        metadata['section'] = 'Introducción'
    elif 'CRONOGRAMA' in content_upper:
        metadata['section'] = 'Cronograma'
    elif 'EVALUACIÓN' in content_upper or 'EVALUACION' in content_upper:
        metadata['section'] = 'Evaluación'
    elif 'METODOLOGÍA' in content_upper or 'METODOLOGIA' in content_upper:
        metadata['section'] = 'Metodología'
    elif 'IMPLEMENTACIÓN' in content_upper or 'IMPLEMENTACION' in content_upper:
        metadata['section'] = 'Implementación'
    elif 'MODELOS' in content_upper:
        metadata['section'] = 'Modelos'
    elif 'LABORATORIO' in content_upper:
        metadata['section'] = 'Laboratorio'

    return metadata


def search_knowledge_base(query: str, backend: str = "qdrant", k: int = 5, filters: dict = None,
                          distance_metric: str = "cosine", index_algorithm: str = "hnsw", collection_suffix: str = None) -> dict:
    import time
    start_time = time.time()

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

    # Pass filters and collection_suffix to the backend search function
    if backend == "qdrant":
        hits = BACKENDS[backend](
            emb, k=k, where=filters, collection_suffix=collection_suffix)
    else:  # pgvector
        hits = BACKENDS[backend](
            emb, k=k, where=filters, collection_suffix=collection_suffix)

    # Calculate search time
    search_time_ms = round((time.time() - start_time) * 1000, 1)

    # Deduplicate hits by (path, chunk_id) before building results.
    # Re-ingest runs can create duplicate Qdrant points for the same chunk.
    _seen: set = set()
    deduped_hits = []
    for hit in hits:
        _k = f"{hit.get('path', hit.get('document', ''))}::{hit.get('chunk_id', hit.get('page', ''))}"
        if _k not in _seen:
            _seen.add(_k)
            deduped_hits.append(hit)
    hits = deduped_hits

    # Create a presentation-friendly response
    results = []
    for hit in hits:
        content = hit.get('content', '') or ''  # Handle None content

        # Enhanced document information - safely get path and fallback to metadata
        path = hit.get('path', hit.get('source_path', ''))
        metadata = hit.get('metadata', {})

        # Try to get clean document name from multiple sources
        if metadata.get('source_name'):
            doc_name = metadata['source_name'].replace('.pdf', '')
        elif path:
            doc_name = path.replace(
                './data/raw/', '').replace('data/raw/', '').replace('.pdf', '')
        else:
            doc_name = 'Documento de curso'

        # Clean the document name for the "document" field - just the filename
        clean_document_name = doc_name.replace(
            '.pdf', '') if doc_name else 'Documento de curso'

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

        # Extract additional metadata for better context
        quality_score = metadata.get('quality_score', 0)
        extractor_used = metadata.get('extractor', 'unknown')

        results.append({
            "document": clean_document_name,
            "reference": reference,
            "page": hit.get('page'),
            "chunk_id": hit.get('chunk_id'),
            "chapter": chapter_info.replace(", ", "") if chapter_info else None,
            "section": section_info.replace(", ", "") if section_info else None,
            "similarity": f"{hit['score']:.3f}",
            "quality": f"{quality_score:.2f}" if quality_score > 0 else "N/A",
            "extractor": extractor_used,
            "preview": content[:120] + "..." if len(content) > 120 else content,
            # Add the original fields from backend for template compatibility
            "path": path,
            "content": content,
            "score": hit.get('score')
        })

    return {
        "query": query,
        "expanded_query": expanded_query if expanded_query != query else None,
        "backend": backend.upper(),
        "backend_info": {
            "search_time_ms": search_time_ms,
            "distance_metric": distance_metric,
            "index_algorithm": index_algorithm,
            "collection_suffix": collection_suffix
        },
        "filters_applied": filters or {},
        "total_results": len(results),
        "results": results
    }


def generate_llm_answer(query: str, backend: str = "qdrant", k: int = 5, model: str = "gemma2:2b", filters: dict = None,
                        distance_metric: str = "cosine", index_algorithm: str = "hnsw", collection_suffix: str = None):
    """Generate AI response using RAG + LLM with improved reranking and query expansion"""
    import time
    start_time = time.time()

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
        hits = BACKENDS[backend](emb, k=max(
            k*3, 12), where=filters, collection_suffix=collection_suffix)
    else:  # pgvector
        hits = BACKENDS[backend](emb, k=max(
            k*3, 12), where=filters, collection_suffix=collection_suffix)

    # Prepare candidates for reranking, deduplicating by (path, chunk_id) so that
    # chunks ingested multiple times don't flood the top-k results.
    seen_keys: set = set()
    candidates = []
    for hit in hits:
        _cid = hit.get('chunk_id', hit.get('page', ''))
        _path = hit.get('path', hit.get('document', ''))
        _key = f"{_path}::{_cid}"
        if _key in seen_keys:
            continue
        seen_keys.add(_key)
        candidates.append({
            'content': hit['content'],
            'sim': min(max(hit['score'], 0.0), 1.0),  # Normalize score to 0-1
            'path': hit['path'],
            'document': hit.get('document', ''),  # Include document name
            'chunk_id': _cid,
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

        # Enhanced document information with rich metadata extraction
        import re

        # Get clean document name - prefer 'document' field, fallback to path processing
        if 'document' in item and item['document']:
            clean_doc_name = item['document']
        elif 'path' in item:
            # Clean document path properly
            raw_path = item['path']
            doc_path = raw_path.replace(
                './data/raw/', '').replace('data/raw/', '').replace('.pdf', '')
            clean_doc_name = doc_path
        else:
            clean_doc_name = "Documento de curso"

        metadata = item.get('metadata', {})

        # Also try to get clean name from metadata if available
        if metadata.get('source_name'):
            clean_doc_name = metadata['source_name'].replace('.pdf', '')
        # Extract document metadata from filename
        doc_metadata = extract_document_metadata(clean_doc_name)

        # Look for chapter and topic information in content
        content_metadata = extract_content_metadata(content)

        # Build detailed reference for "Fuentes consultadas" (clean, no file path)
        ref_parts = []
        if doc_metadata.get('theme'):
            ref_parts.append(doc_metadata['theme'])
        if doc_metadata.get('topic'):
            ref_parts.append(doc_metadata['topic'])
        if item.get('page'):
            ref_parts.append(f"p.{item['page']}")
        if content_metadata.get('chapter'):
            ref_parts.append(f"Cap. {content_metadata['chapter']}")
        if content_metadata.get('section'):
            ref_parts.append(content_metadata['section'])

        # Use clean document name with page reference if available
        if clean_doc_name and clean_doc_name != "Documento" and item.get('page'):
            clean_reference = f"{clean_doc_name} (página {item['page']})"
        elif ref_parts:
            clean_reference = " - ".join(ref_parts)
        else:
            clean_reference = f"Documento (página {item.get('page', '?')})"

        # Build enhanced reference for detailed sources section
        detailed_parts = []
        if doc_metadata.get('course'):
            detailed_parts.append(f"Curso: {doc_metadata['course']}")
        if doc_metadata.get('theme'):
            detailed_parts.append(doc_metadata['theme'])
        if doc_metadata.get('topic'):
            detailed_parts.append(doc_metadata['topic'])
        if doc_metadata.get('year'):
            detailed_parts.append(f"Año: {doc_metadata['year']}")
        if doc_metadata.get('version'):
            detailed_parts.append(f"v{doc_metadata['version']}")

        enhanced_reference = " | ".join(
            detailed_parts) if detailed_parts else clean_reference

        results.append({
            "document": clean_doc_name,
            "reference": clean_reference,  # Clean reference without file path
            "detailed_reference": enhanced_reference,  # Full reference with all metadata
            "page": item.get('page'),
            "chapter": content_metadata.get('chapter'),
            "section": content_metadata.get('section'),
            "theme": doc_metadata.get('theme'),
            "topic": doc_metadata.get('topic'),
            "course": doc_metadata.get('course'),
            "year": doc_metadata.get('year'),
            "version": doc_metadata.get('version'),
            "extractor": metadata.get('extractor', 'unknown'),
            "word_count": metadata.get('word_count', 0),
            "quality_score": metadata.get('quality_score', 0),
            "similarity": f"{item['sim']:.3f}",
            "preview": content[:120] + "..." if len(content) > 120 else content
        })

    # Build context for LLM using enhanced method if available
    if build_context_enhanced:
        sources = build_context_enhanced(reranked, max_tokens=1200)
    else:
        sources = build_context(reranked)
    prompt = RAG_TEMPLATE.format(
        system_prompt=load_system_prompt(), q=query, sources=sources)

    try:
        # Use resilient Ollama generation with automatic retry and fallback
        if ollama_generate_with_retry:
            response = ollama_generate_with_retry(
                model=model,
                prompt=prompt,
                max_retries=3,
                auto_fallback=True,  # Automatically try smaller models on GPU memory errors
                auto_restart=True,   # Attempt container restart on first failure
                options={
                    "repeat_penalty": 1.0,
                    "temperature": 0.5,
                    "num_predict": 2048,
                    "top_p": 0.9
                }
            )

            # Extract metadata from resilient wrapper
            metadata = response.get('_metadata', {})
            if metadata.get('fallback_used'):
                logger.info(
                    f"⚠️  Used fallback model: {metadata['model_used']} (original: {model})")
        else:
            # Fallback to direct Ollama call if utils not available
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)
            response = client.generate(
                model=model,
                prompt=prompt,
                options={
                    "repeat_penalty": 1.0,
                    "temperature": 0.5,
                    "num_predict": 2048,
                    "top_p": 0.9
                }
            )
            metadata = {}

        # Extract the generated text — strip any model-generated source lists
        # (the UI renders sources from live data; the LLM must not add its own)
        import re as _re
        raw_response = response.get('response', '') if hasattr(
            response, 'get') else getattr(response, 'response', '')
        enhanced_response = (raw_response or '').strip()
        # Strip thinking blocks (<think>...</think>) produced by reasoning models like qwen3
        enhanced_response = _re.sub(
            r'<think>[\s\S]*?</think>\s*', '', enhanced_response).strip()
        # Only strip "Fuentes / Sources / References" when they appear at the START of a line
        enhanced_response = _re.sub(
            r'(?m)^\*{0,2}(?:Fuentes consultadas|Fuentes|Sources?|Références?|References?)\*{0,2}:?\s*$[\s\S]*',
            '', enhanced_response, flags=_re.IGNORECASE
        ).rstrip()

        # Calculate total processing time
        total_time_ms = round((time.time() - start_time) * 1000, 1)

        # Build response with model metadata
        response_data = {
            "query": query,
            "expanded_query": expanded_query if expanded_query != query else None,
            "backend": backend.upper(),
            "backend_info": {
                "search_time_ms": total_time_ms,
                "distance_metric": distance_metric,
                "index_algorithm": index_algorithm,
                "collection_suffix": collection_suffix
            },
            "model": metadata.get('model_used', model) if metadata else model,
            "filters_applied": filters or {},
            "ai_response": enhanced_response,
            "answer": enhanced_response,  # Add answer field for consistency
            "total_results": len(results),
            "sources": results,
            "results": results,  # Add results field for consistency
            "prompt_used": prompt
        }

        # Add model fallback info if applicable
        if metadata:
            if metadata.get('fallback_used'):
                response_data["model_fallback_info"] = {
                    "original_model": model,
                    "used_model": metadata['model_used'],
                    "models_tried": metadata.get('models_tried', []),
                    "attempts": metadata.get('attempts', 1)
                }

        return response_data

    except Exception as e:
        try:
            logger.error(f"Ollama API error: {e}")
        except NameError:
            # Fallback if logger is not available in this context
            import logging
            logging.error(f"Ollama API error: {e}")

        # Calculate time even for error case
        total_time_ms = round((time.time() - start_time) * 1000, 1)

        # Fallback to RAG-only response if LLM fails
        return {
            "query": query,
            "expanded_query": expanded_query if expanded_query != query else None,
            "backend": backend.upper(),
            "backend_info": {
                "search_time_ms": total_time_ms,
                "distance_metric": distance_metric,
                "index_algorithm": index_algorithm,
                "collection_suffix": collection_suffix
            },
            "model": model,
            "filters_applied": filters or {},
            "ai_response": f"❌ LLM Error: {str(e)}. Here are the related documents:",
            "total_results": len(results),
            "sources": results,
            "error": str(e),
            "fallback_mode": True
        }
