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

    # Prepare candidates for reranking
    candidates = []
    for hit in hits:
        candidates.append({
            'content': hit['content'],
            'sim': min(max(hit['score'], 0.0), 1.0),  # Normalize score to 0-1
            'path': hit['path'],
            'document': hit.get('document', ''),  # Include document name
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

        # Build enhanced source attribution footer with full metadata
        source_references = []
        for i, result in enumerate(results[:3], 1):  # Show top 3 sources
            # Extract clean document name - prefer 'document' field, fallback to path processing
            if 'document' in result and result['document']:
                doc_name = result['document']
            else:
                # Build comprehensive reference from available metadata
                path = result.get('path', result.get('source_path', ''))

                # Extract clean document name from path
                doc_name = path
                if 'data/raw/' in doc_name:
                    doc_name = doc_name.split('data/raw/')[-1]
                elif 'data/clean/' in doc_name:
                    doc_name = doc_name.split('data/clean/')[-1]

                # Clean up file extensions and normalize name
                doc_name = doc_name.replace('.pdf', '').replace(
                    '.jsonl', '').replace('.chunks', '')
                doc_name = doc_name.replace('.txt', '').replace(
                    '.yaml', '').replace('.yml', '')

                # Handle empty or missing document names
                if not doc_name or doc_name.strip() == '':
                    doc_name = "Documento de curso"

            # Add page information if available
            page_info = ""
            if result.get('page'):
                page_info = f" (página {result['page']})"
            elif result.get('chunk_id'):
                page_info = f" (sección {result['chunk_id']})"

            # Add any chapter or additional metadata
            metadata = result.get('metadata', {})
            chapter_info = ""
            if metadata.get('chapter'):
                chapter_info = f", {metadata['chapter']}"
            elif metadata.get('section'):
                chapter_info = f", {metadata['section']}"

            # Create full reference with document name included
            full_ref = f"{doc_name}{page_info}{chapter_info}"
            source_references.append(f"{i}. {full_ref}")

        # Add enhanced footer to the response - using markdown formatting
        if source_references:
            sources_text = "\n".join(source_references)
            enhanced_response = f"{generated_text}\n\n\n**Fuentes consultadas:**\n{sources_text}"
        else:
            enhanced_response = generated_text

        # Calculate total processing time
        total_time_ms = round((time.time() - start_time) * 1000, 1)

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
            "ai_response": enhanced_response,
            "answer": enhanced_response,  # Add answer field for consistency
            "total_results": len(results),
            "sources": results,
            "results": results,  # Add results field for consistency
            "prompt_used": prompt
        }

    except Exception as e:
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
