"""
Advanced RAG Techniques Module

This module implements several state-of-the-art RAG enhancements:
1. Contextual Query Rephrasing (Multi-Query)
2. Query Decomposition
3. Hypothetical Document Embeddings (HyDE)
4. Hybrid Search (Keyword + Semantic) with RRF
5. Multi-Round (Iterative) Retrieval

All techniques can be used independently or combined.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Import existing infrastructure
try:
    import ollama
except ImportError:
    ollama = None

try:
    from scripts.query_embed import embed_query as embed_query_clean, expand_query, embed_e5
except ImportError:
    embed_query_clean = None
    embed_e5 = None
    def expand_query(x): return x

from app.qdrant_backend import search_qdrant
from app.pgvector_backend import search_pgvector
from app.rerank import mmr, build_context

try:
    from app.ollama_utils import ollama_generate_with_retry, check_ollama_health
except ImportError:
    ollama_generate_with_retry = None
    check_ollama_health = None

logger = logging.getLogger(__name__)

BACKENDS = {"qdrant": search_qdrant, "pgvector": search_pgvector}

# Improved prompt template for AI answer generation - STRICT fact-only version
PROMPT_TEMPLATE = """Eres un asistente académico especializado. Responde ÚNICAMENTE basándote en los fragmentos proporcionados.

Pregunta del usuario: {q}

Fragmentos de documentos disponibles:
{sources}

INSTRUCCIONES ESTRICTAS:
- SOLO usa información que aparezca textualmente en los fragmentos
- Si la información no está en los fragmentos, responde: "La información solicitada no está disponible en los documentos consultados."
- NO inventes, asumas o extrapoles información
- USA datos exactos: fechas, números, nombres tal como aparecen en los fragmentos
- Cita el documento fuente cuando sea relevante para la comprensión
- Mantén el idioma español académico y profesional
- Sé específico y preciso con los detalles disponibles

Respuesta basada en los fragmentos:
"""


def generate_ai_answer_from_results(
    query: str,
    results: List[Dict],
    model: str = "phi3:mini"
) -> str:
    """Generate AI answer from search results using LLM with strict fact-only guidelines."""
    if not ollama or not results:
        return "No se pudo generar una respuesta AI. Servicio de IA no disponible o sin resultados."

    # Build context from results
    context = build_context(results, max_tokens=1500)
    if not context.strip():
        return "No se encontró contexto suficiente para generar una respuesta basada en los documentos."

    # Generate answer using LLM with strict guidelines
    prompt = PROMPT_TEMPLATE.format(q=query, sources=context)

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Lower temperature for more factual responses
                "num_predict": 400,  # Increased length for detailed responses
                "top_p": 0.8,        # More focused sampling
                "repeat_penalty": 1.1
            }
        )

        answer = response['response'].strip()

        # Ensure response is in Spanish if it appears to be in English
        if len(answer) > 10 and answer.count(' ') > 2:
            # Simple heuristic to check for English responses
            english_indicators = ['the', 'and', 'or',
                                  'is', 'are', 'was', 'were', 'this', 'that']
            spanish_indicators = ['el', 'la', 'y', 'o', 'es',
                                  'son', 'fue', 'era', 'esto', 'eso', 'los', 'las']

            words = answer.lower().split()[:20]  # Check first 20 words
            english_count = sum(
                1 for word in words if word in english_indicators)
            spanish_count = sum(
                1 for word in words if word in spanish_indicators)

            if english_count > spanish_count and english_count > 2:
                return "La respuesta debe generarse en español. Los documentos contienen información relevante pero la respuesta se generó en inglés. Por favor, reintente la consulta."

        return answer

    except Exception as e:
        logger.error(f"Error generating AI answer: {e}")
        return f"Error al generar respuesta AI: {str(e)}"


# ============================================================================
# 1. CONTEXTUAL QUERY REPHRASING (Multi-Query)
# ============================================================================

def generate_query_variations(
    query: str,
    model: str = "phi3:mini",
    num_variations: int = 3
) -> List[str]:
    """
    Generate multiple rephrased versions of the query using LLM.

    Args:
        query: Original user query
        model: LLM model to use for rephrasing
        num_variations: Number of query variations to generate

    Returns:
        List of query variations (including original)
    """
    if ollama is None:
        logger.warning("Ollama not available, returning original query only")
        return [query]

    prompt = f"""Eres un experto en reformular preguntas. Genera {num_variations} versiones alternativas de la siguiente pregunta, cada una usando diferente vocabulario o estructura, pero manteniendo el mismo significado.

Pregunta original: {query}

Instrucciones:
- Mantén el idioma español
- Usa sinónimos y diferentes formas de expresar la misma idea
- Mantén el contexto académico
- Responde solo con las {num_variations} preguntas reformuladas, una por línea
- NO incluyas números, viñetas o explicaciones adicionales

Preguntas reformuladas:
"""

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 200,
                "top_p": 0.9
            }
        )

        variations_text = response.get('response', '').strip()

        # Parse variations (split by newline, clean up)
        variations = []
        for line in variations_text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Remove numbered prefixes (1., 2., 3., etc.)
            import re
            line_clean = re.sub(r'^\d+\.\s*', '', line)

            # Skip lines that start with bullets or are too short
            if line_clean and not line_clean.startswith(('-', '*', '•')):
                variations.append(line_clean)

        # Limit to requested number of variations
        variations = variations[:num_variations]

        # Always include original query
        all_queries = [query] + variations
        return all_queries

    except Exception as e:
        logger.error(f"Query rephrasing failed: {e}")
        return [query]


def reciprocal_rank_fusion(
    results_list: List[List[Dict]],
    k: int = 60
) -> List[Dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF Formula: score = Σ(1 / (k + rank_i))

    Args:
        results_list: List of result lists from different queries
        k: Constant to smooth ranking (typically 60)

    Returns:
        Fused and re-ranked results
    """
    # Track scores for each unique document
    doc_scores = defaultdict(float)
    doc_data = {}  # Store full document data

    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            # Unique key: prefer path, fall back to document name, then content prefix
            _id_path = doc.get('path', '') or doc.get(
                'document', '') or doc.get('reference', '')
            _id_chunk = doc.get('chunk_id', doc.get('page', '')) or str(rank)
            doc_key = f"{_id_path}::{_id_chunk}"

            # Calculate RRF score
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_key] += rrf_score

            # Store document data (first occurrence or update with better score)
            if doc_key not in doc_data or doc.get('score', 0) > doc_data[doc_key].get('score', 0):
                doc_data[doc_key] = doc

    # Sort by RRF score
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Return documents with updated scores
    fused_results = []
    for doc_key, rrf_score in ranked_docs:
        doc = doc_data[doc_key].copy()
        doc['rrf_score'] = rrf_score
        doc['fusion_method'] = 'RRF'
        fused_results.append(doc)

    return fused_results


def multi_query_search(
    query: str,
    backend: str = "qdrant",
    k: int = 5,
    num_variations: int = 3,
    model: str = "phi3:mini",
    filters: Optional[Dict] = None,
    collection_suffix: str = None
) -> Dict[str, Any]:
    """
    Perform multi-query search with query rephrasing and RRF fusion.
    Enhanced with comprehensive timing and debugging information.

    Args:
        query: Original user query
        backend: Vector database backend
        k: Number of results per query
        num_variations: Number of query variations to generate
        model: LLM model for query rephrasing
        filters: Optional metadata filters

    Returns:
        Enhanced search results with RRF scores and detailed timing
    """
    import time
    start_time = time.time()

    # Step 1: Generate query variations (with timing)
    variation_start = time.time()
    queries = generate_query_variations(query, model, num_variations)
    variation_time = round((time.time() - variation_start) * 1000, 1)

    # Step 2: Perform search for each query variation (with timing)
    search_start = time.time()
    all_results = []
    individual_search_times = []

    for i, q in enumerate(queries):
        individual_start = time.time()

        # Embed and search
        if embed_query_clean:
            emb = embed_query_clean(q)
        else:
            emb = embed_e5([q], is_query=True)[0]

        results = BACKENDS[backend](
            emb, k=k, where=filters, collection_suffix=collection_suffix)
        all_results.append({
            "query": q,
            "query_number": i + 1,
            "is_original": i == 0,
            "results": results,
            "results_count": len(results)
        })

        individual_time = round((time.time() - individual_start) * 1000, 1)
        individual_search_times.append(individual_time)

    search_time = round((time.time() - search_start) * 1000, 1)

    # Step 3: Apply RRF fusion (with timing)
    fusion_start = time.time()
    result_lists = [item["results"] for item in all_results]
    fused_results = reciprocal_rank_fusion(result_lists, k=60)
    fusion_time = round((time.time() - fusion_start) * 1000, 1)

    # Step 4: Generate AI answer from fused results (with timing)
    answer_start = time.time()
    ai_answer = generate_ai_answer_from_results(
        query, fused_results[:k], model)
    answer_time = round((time.time() - answer_start) * 1000, 1)

    # Calculate total time
    total_time = round((time.time() - start_time) * 1000, 1)

    return {
        "query": query,
        "method": "Multi-Query + RRF",
        "model": model,
        "backend": backend.upper(),

        # Core results
        "ai_response": ai_answer,
        "sources": fused_results[:k],
        "results": fused_results[:k],
        "total_results": len(fused_results[:k]),

        # Query variation details
        "query_variations": queries[1:],  # Exclude original
        "total_queries": len(queries),
        "variation_details": all_results,

        # Comprehensive timing breakdown
        "timing": {
            "total_time_ms": total_time,
            "variation_generation_ms": variation_time,
            "search_time_ms": search_time,
            "fusion_time_ms": fusion_time,
            "answer_generation_ms": answer_time,
            "individual_search_times_ms": individual_search_times
        },

        # Process steps for debugging/learning
        "process_steps": [
            {
                "step": 1,
                "name": "Generación de Variaciones",
                "description": f"Se generaron {num_variations} reformulaciones de la consulta usando {model}",
                "time_ms": variation_time,
                "output": f"Consultas totales: {len(queries)} (1 original + {len(queries)-1} variaciones)"
            },
            {
                "step": 2,
                "name": "Búsqueda Vectorial Múltiple",
                "description": f"Cada consulta fue embebida y buscada independientemente en {backend}",
                "time_ms": search_time,
                "output": f"{len(queries)} búsquedas × {k} resultados máx. = {sum(item['results_count'] for item in all_results)} fragmentos totales"
            },
            {
                "step": 3,
                "name": "Fusión RRF",
                "description": "Los resultados se fusionaron usando Reciprocal Rank Fusion con k=60",
                "time_ms": fusion_time,
                "output": f"Fragmentos únicos después de fusión: {len(fused_results)}"
            },
            {
                "step": 4,
                "name": "Generación de Respuesta",
                "description": f"Se generó respuesta final usando {model} con los mejores {k} fragmentos",
                "time_ms": answer_time,
                "output": f"Respuesta generada basada en {len(fused_results[:k])} fuentes"
            }
        ],

        # Technical metadata
        "technical_details": {
            "rrf_k_parameter": 60,
            "fusion_method": "Reciprocal Rank Fusion",
            "embedding_model": "multilingual-e5-base",
            "filters_applied": filters or {},
            "temperature_used": 0.7,
            "variations_requested": num_variations,
            "variations_generated": len(queries) - 1
        }
    }


# ============================================================================
# 2. QUERY DECOMPOSITION
# ============================================================================

def decompose_query(
    query: str,
    model: str = "phi3:mini"
) -> List[str]:
    """
    Decompose complex query into simpler sub-questions.

    Args:
        query: Complex user query
        model: LLM model for decomposition

    Returns:
        List of atomic sub-questions
    """
    if ollama is None:
        logger.warning("Ollama not available, returning original query")
        return [query]

    prompt = f"""Eres un experto en análisis de preguntas complejas. Descompón la siguiente pregunta en sub-preguntas más simples y específicas que se puedan responder de forma independiente.

Pregunta compleja: {query}

Instrucciones:
- Identifica si la pregunta es compleja y requiere descomposición
- Si es simple, responde "SIMPLE" y repite la pregunta
- Si es compleja, genera entre 2-4 sub-preguntas específicas
- Cada sub-pregunta debe ser independiente y clara
- Mantén el idioma español y contexto académico
- Responde solo con las sub-preguntas, una por línea
- NO incluyas números, viñetas o explicaciones

Sub-preguntas:
"""

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 250
            }
        )

        decomposition = response.get('response', '').strip()

        # Check if query is simple
        if decomposition.startswith("SIMPLE"):
            return [query]

        # Parse sub-questions
        sub_questions = [
            line.strip()
            for line in decomposition.split('\n')
            if line.strip() and not line.strip().startswith(('-', '*', '•'))
        ]

        return sub_questions if sub_questions else [query]

    except Exception as e:
        logger.error(f"Query decomposition failed: {e}")
        return [query]


def decomposed_search(
    query: str,
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    filters: Optional[Dict] = None,
    synthesize: bool = True,
    collection_suffix: str = None
) -> Dict[str, Any]:
    """
    Perform search using query decomposition strategy with comprehensive debugging info.

    Args:
        query: Complex user query
        backend: Vector database backend
        k: Number of results per sub-question
        model: LLM model for decomposition and synthesis
        filters: Optional metadata filters
        synthesize: Whether to synthesize final answer from sub-results

    Returns:
        Comprehensive results with synthesis and detailed timing
    """
    import time
    start_time = time.time()

    # Step 1: Decompose query (with timing)
    decomposition_start = time.time()
    sub_questions = decompose_query(query, model)
    decomposition_time = round((time.time() - decomposition_start) * 1000, 1)

    # Step 2: Search for each sub-question (with timing)
    search_start = time.time()
    sub_results = []
    all_documents = []

    for i, sub_q in enumerate(sub_questions):
        sub_search_start = time.time()

        # Embed and search
        if embed_query_clean:
            emb = embed_query_clean(sub_q)
        else:
            emb = embed_e5([sub_q], is_query=True)[0]

        results = BACKENDS[backend](
            emb, k=k, where=filters, collection_suffix=collection_suffix)
        sub_search_time = round((time.time() - sub_search_start) * 1000, 1)

        sub_results.append({
            "sub_question": sub_q,
            "sub_question_number": i + 1,
            "results": results[:3],  # Top 3 per sub-question for display
            "total_found": len(results),
            "search_time_ms": sub_search_time
        })
        all_documents.extend(results)

    search_time = round((time.time() - search_start) * 1000, 1)

    # Step 3: Deduplicate and rerank (with timing)
    rerank_start = time.time()
    unique_docs = {
        f"{doc.get('path', '')}:{doc.get('chunk_id', doc.get('page', ''))}": doc
        for doc in all_documents
    }

    # Apply MMR reranking if available
    if mmr:
        candidates = [
            {
                'content': doc['content'],
                'sim': doc.get('score', 0.5),
                'path': doc['path'],
                'chunk_id': doc.get('chunk_id', ''),
                'page': doc.get('page'),
                'metadata': doc.get('metadata', {})
            }
            for doc in unique_docs.values()
        ]
        final_results = mmr(query, candidates, lambda_=0.7, top_k=k)
    else:
        final_results = list(unique_docs.values())[:k]

    rerank_time = round((time.time() - rerank_start) * 1000, 1)

    # Step 4: Synthesize final answer if requested (with timing)
    synthesis_start = time.time()
    synthesized_answer = None
    synthesis_time = 0

    if synthesize and ollama:
        try:
            context = build_context(final_results, max_tokens=1500)
            synthesis_prompt = f"""Responde la siguiente pregunta compleja usando ÚNICAMENTE la información de los fragmentos proporcionados.

Pregunta original: {query}

Sub-preguntas analizadas:
{chr(10).join(f"- {sq}" for sq in sub_questions)}

Fragmentos relevantes encontrados:
{context}

INSTRUCCIONES ESTRICTAS:
- SOLO usa información que aparezca textualmente en los fragmentos
- Integra información de todos los sub-temas relevantes disponibles
- Usa fechas, números y nombres exactos de los fragmentos
- Si alguna parte no se puede responder con los fragmentos, indica claramente: "Esta información no está disponible en los documentos consultados"
- Mantén un tono académico y profesional en español
- NO inventes o asumas información no presente en los fragmentos

Respuesta integral basada en los fragmentos:
"""

            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)

            response = client.generate(
                model=model,
                prompt=synthesis_prompt,
                options={
                    "temperature": 0.1,  # Lower for factual synthesis
                    "num_predict": 500,
                    "top_p": 0.8,
                    "repeat_penalty": 1.1
                }
            )

            synthesized_answer = response.get('response', '').strip()

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            synthesized_answer = f"Error en la síntesis: {str(e)}"

    synthesis_time = round((time.time() - synthesis_start) * 1000, 1)

    # Use synthesized answer or fallback to generated answer
    ai_answer = synthesized_answer or generate_ai_answer_from_results(
        query, final_results, model)

    # Calculate total time
    total_time = round((time.time() - start_time) * 1000, 1)

    return {
        "query": query,
        "method": "Query Decomposition",
        "model": model,
        "backend": backend.upper(),

        # Core results
        "ai_response": ai_answer,
        "sources": final_results,
        "results": final_results,
        "total_results": len(final_results),

        # Detailed debugging information
        "sub_questions": sub_questions,
        "sub_results": sub_results,
        "total_unique_documents": len(unique_docs),
        "decomposition_method": "LLM-based query analysis" if len(sub_questions) > 1 else "Simple query (no decomposition needed)",

        # Comprehensive timing breakdown
        "timing": {
            "total_time_ms": total_time,
            "decomposition_time_ms": decomposition_time,
            "search_time_ms": search_time,
            "rerank_time_ms": rerank_time,
            "synthesis_time_ms": synthesis_time
        },

        # Process steps for debugging/learning
        "process_steps": [
            {
                "step": 1,
                "name": "Descomposición de Consulta",
                "description": f"Se analizó la consulta y se dividió en {len(sub_questions)} sub-pregunta(s)",
                "time_ms": decomposition_time,
                "output": f"Sub-preguntas generadas: {len(sub_questions)}"
            },
            {
                "step": 2,
                "name": "Búsqueda Individual",
                "description": f"Se ejecutaron {len(sub_questions)} búsquedas vectoriales independientes",
                "time_ms": search_time,
                "output": f"Total de fragmentos encontrados: {len(all_documents)}"
            },
            {
                "step": 3,
                "name": "Deduplicación y Re-ranking",
                "description": f"Se aplicó MMR para seleccionar los mejores {k} fragmentos únicos",
                "time_ms": rerank_time,
                "output": f"Documentos únicos: {len(unique_docs)} → Seleccionados: {len(final_results)}"
            },
            {
                "step": 4,
                "name": "Síntesis de Respuesta",
                "description": f"Se generó respuesta integral usando {model}",
                "time_ms": synthesis_time,
                "output": f"Respuesta sintetizada: {'Sí' if synthesized_answer else 'Fallback usado'}"
            }
        ],

        # Technical metadata
        "technical_details": {
            "query_complexity": "Compleja" if len(sub_questions) > 1 else "Simple",
            "filters_applied": filters or {},
            "mmr_applied": bool(mmr),
            "synthesis_enabled": synthesize,
            "documents_per_subquery": k,
            "embedding_model": "multilingual-e5-base"
        }
    }


# ============================================================================
# 3. HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE)
# ============================================================================

def generate_hypothetical_document(
    query: str,
    model: str = "phi3:mini"
) -> str:
    """
    Generate a hypothetical answer document for HyDE approach.

    Args:
        query: User query
        model: LLM model to generate hypothetical answer

    Returns:
        Hypothetical document text
    """
    if ollama is None:
        logger.warning("Ollama not available, returning query as-is")
        return query

    prompt = f"""Eres un experto académico. Escribe un párrafo de respuesta hipotética para la siguiente pregunta, como si estuviera en un documento académico o guía de curso.

Pregunta: {query}

Instrucciones:
- Escribe 2-3 párrafos con información técnica y específica
- Usa terminología académica apropiada
- Incluye detalles técnicos, nombres de conceptos, y ejemplos
- Escribe en estilo formal académico, como en un libro de texto
- NO menciones que es una respuesta hipotética
- Usa español académico

Respuesta hipotética:
"""

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 300
            }
        )

        hypothetical_doc = response.get('response', '').strip()
        return hypothetical_doc if hypothetical_doc else query

    except Exception as e:
        logger.error(f"Hypothetical document generation failed: {e}")
        return query


def hyde_search(
    query: str,
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    filters: Optional[Dict] = None,
    generate_final_answer: bool = True,
    collection_suffix: str = None
) -> Dict[str, Any]:
    """
    Perform search using Hypothetical Document Embeddings (HyDE).
    Enhanced with comprehensive timing and debugging information.

    Args:
        query: User query
        backend: Vector database backend
        k: Number of results to return
        model: LLM model for generating hypothetical document
        filters: Optional metadata filters
        generate_final_answer: Whether to generate final answer with real docs

    Returns:
        Search results using HyDE approach with detailed timing
    """
    import time
    start_time = time.time()

    # Step 1: Generate hypothetical document (with timing)
    hypothesis_start = time.time()
    hypothetical_doc = generate_hypothetical_document(query, model)
    hypothesis_time = round((time.time() - hypothesis_start) * 1000, 1)

    # Step 2: Embed the hypothetical document (with timing)
    embedding_start = time.time()
    if embed_query_clean:
        # Note: We embed as document, not as query
        hyp_embedding = embed_e5([hypothetical_doc], is_query=False)[0]
    else:
        hyp_embedding = embed_e5([hypothetical_doc], is_query=False)[0]
    embedding_time = round((time.time() - embedding_start) * 1000, 1)

    # Step 3: Search using hypothetical document embedding (with timing)
    search_start = time.time()
    results = BACKENDS[backend](
        hyp_embedding, k=k, where=filters, collection_suffix=collection_suffix)
    search_time = round((time.time() - search_start) * 1000, 1)

    # Step 4: Generate final answer with real documents (with timing)
    answer_start = time.time()
    final_answer = None
    if generate_final_answer and ollama and results:
        try:
            context = build_context(results, max_tokens=1500)
            answer_prompt = f"""Responde la siguiente pregunta usando ÚNICAMENTE la información de los fragmentos de documentos reales proporcionados.

Pregunta: {query}

Fragmentos de documentos encontrados:
{context}

INSTRUCCIONES ESTRICTAS:
- SOLO usa información que aparezca textualmente en los fragmentos
- Sé específico con fechas, números y nombres tal como aparecen
- Si la información no está disponible en los fragmentos, indica: "Esta información no está disponible en los documentos consultados"
- Mantén un tono académico profesional en español
- NO inventes o extrapoles información

Respuesta basada en fragmentos reales:
"""

            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)

            response = client.generate(
                model=model,
                prompt=answer_prompt,
                options={
                    "temperature": 0.1,  # Very low for factual responses
                    "num_predict": 400,
                    "top_p": 0.8,
                    "repeat_penalty": 1.1
                }
            )

            final_answer = response.get('response', '').strip()

        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            final_answer = f"Error al generar respuesta final: {str(e)}"

    answer_time = round((time.time() - answer_start) * 1000, 1)

    # Use generated answer or fallback
    ai_answer = final_answer or generate_ai_answer_from_results(
        query, results, model)

    # Calculate total time
    total_time = round((time.time() - start_time) * 1000, 1)

    return {
        "query": query,
        "method": "HyDE (Hypothetical Document Embeddings)",
        "model": model,
        "backend": backend.upper(),

        # Core results
        "ai_response": ai_answer,
        "sources": results,
        "results": results,
        "total_results": len(results),

        # HyDE-specific details
        "hypothetical_document": hypothetical_doc,
        "hypothetical_doc_length": len(hypothetical_doc.split()),
        "hypothetical_answer": hypothetical_doc,  # For template compatibility

        # Comprehensive timing breakdown
        "timing": {
            "total_time_ms": total_time,
            "hypothesis_generation_ms": hypothesis_time,
            "embedding_time_ms": embedding_time,
            "search_time_ms": search_time,
            "answer_generation_ms": answer_time
        },

        # Process steps for debugging/learning
        "process_steps": [
            {
                "step": 1,
                "name": "Generación de Documento Hipotético",
                "description": f"Se generó una respuesta hipotética detallada usando {model}",
                "time_ms": hypothesis_time,
                "output": f"Documento hipotético: {len(hypothetical_doc.split())} palabras"
            },
            {
                "step": 2,
                "name": "Embedding del Documento Hipotético",
                "description": "El documento hipotético fue convertido a vector de 768 dimensiones",
                "time_ms": embedding_time,
                "output": "Vector generado como documento (no como consulta)"
            },
            {
                "step": 3,
                "name": "Búsqueda Semántica",
                "description": f"Se buscaron documentos similares al contenido hipotético en {backend}",
                "time_ms": search_time,
                "output": f"Fragmentos encontrados: {len(results)}"
            },
            {
                "step": 4,
                "name": "Generación de Respuesta Real",
                "description": f"Se generó respuesta final usando documentos reales con {model}",
                "time_ms": answer_time,
                "output": f"Respuesta basada en {len(results)} fragmentos reales"
            }
        ],

        # Technical metadata
        "technical_details": {
            "hyde_approach": "Document-to-document embedding similarity",
            "embedding_model": "multilingual-e5-base",
            "embedding_type": "document_embedding (not query_embedding)",
            "hypothetical_temperature": 0.7,
            "answer_temperature": 0.1,
            "filters_applied": filters or {},
            "final_answer_generated": bool(final_answer)
        }
    }


# ============================================================================
# 4. HYBRID SEARCH (Keyword + Semantic) with RRF
# ============================================================================

def bm25_search(
    query: str,
    documents: List[Dict],
    k: int = 10
) -> List[Tuple[Dict, float]]:
    """
    Simple BM25 implementation for keyword search.

    Args:
        query: Search query
        documents: List of documents to search
        k: Number of results

    Returns:
        List of (document, score) tuples
    """
    from collections import Counter
    import math

    # Parameters
    k1 = 1.5
    b = 0.75

    # Tokenize
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    query_tokens = tokenize(query)

    # Calculate document frequencies
    doc_freqs = Counter()
    doc_lengths = []
    tokenized_docs = []

    for doc in documents:
        content = doc.get('content', '')
        tokens = tokenize(content)
        tokenized_docs.append(tokens)
        doc_lengths.append(len(tokens))
        doc_freqs.update(set(tokens))

    avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
    N = len(documents)

    # Calculate BM25 scores
    scores = []
    for idx, (doc, tokens) in enumerate(zip(documents, tokenized_docs)):
        score = 0.0
        doc_length = doc_lengths[idx]
        token_freqs = Counter(tokens)

        for term in query_tokens:
            if term in token_freqs:
                tf = token_freqs[term]
                df = doc_freqs[term]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                norm = (tf * (k1 + 1)) / (tf + k1 *
                                          (1 - b + b * doc_length / avg_doc_length))
                score += idf * norm

        scores.append((doc, score))

    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def hybrid_search(
    query: str,
    backend: str = "qdrant",
    k: int = 5,
    semantic_weight: float = 0.7,
    filters: Optional[Dict] = None,
    use_rrf: bool = True,
    collection_suffix: str = None
) -> Dict[str, Any]:
    """
    Perform hybrid search combining semantic (vector) and keyword (BM25) search.
    Enhanced with comprehensive timing and debugging information.

    Args:
        query: User query
        backend: Vector database backend
        k: Number of final results
        semantic_weight: Weight for semantic search (0-1), keyword = 1 - semantic_weight
        filters: Optional metadata filters
        use_rrf: Whether to use RRF fusion (vs weighted score)

    Returns:
        Hybrid search results with detailed timing and process information
    """
    import time
    start_time = time.time()

    # Step 1: Perform semantic search (with timing)
    semantic_start = time.time()
    if embed_query_clean:
        emb = embed_query_clean(query)
    else:
        emb = embed_e5([query], is_query=True)[0]

    semantic_results = BACKENDS[backend](
        emb, k=k*3, where=filters, collection_suffix=collection_suffix)
    semantic_time = round((time.time() - semantic_start) * 1000, 1)

    # Step 2: Perform keyword search on the same set (with timing)
    keyword_start = time.time()
    keyword_results = bm25_search(query, semantic_results, k=k*3)
    keyword_results_dict = {
        f"{doc.get('path', '')}:{doc.get('chunk_id', doc.get('page', ''))}": score
        for doc, score in keyword_results
    }
    keyword_time = round((time.time() - keyword_start) * 1000, 1)

    # Step 3: Fusion process (with timing)
    fusion_start = time.time()

    if use_rrf:
        # Use RRF to combine rankings
        fused_results = reciprocal_rank_fusion(
            [semantic_results[:k*2], [doc for doc, _ in keyword_results[:k*2]]])
        final_results = fused_results[:k]
        fusion_method_used = "Reciprocal Rank Fusion (RRF)"
    else:
        # Use weighted score combination
        combined_scores = {}

        # Normalize and combine scores
        max_sem_score = max([r.get('score', 0)
                            for r in semantic_results]) if semantic_results else 1
        max_kw_score = max([s for _, s in keyword_results]
                           ) if keyword_results else 1

        for doc in semantic_results:
            doc_key = f"{doc.get('path', '')}:{doc.get('chunk_id', doc.get('page', ''))}"
            sem_score = doc.get('score', 0) / max_sem_score
            kw_score = keyword_results_dict.get(doc_key, 0) / max_kw_score

            combined_score = semantic_weight * \
                sem_score + (1 - semantic_weight) * kw_score
            combined_scores[doc_key] = (doc, combined_score)

        # Sort by combined score
        sorted_results = sorted(combined_scores.values(),
                                key=lambda x: x[1], reverse=True)
        final_results = [doc for doc, _ in sorted_results[:k]]
        fusion_method_used = f"Weighted Combination ({semantic_weight:.1f} semantic + {1-semantic_weight:.1f} keyword)"

    fusion_time = round((time.time() - fusion_start) * 1000, 1)

    # Step 4: Generate AI answer from hybrid results (with timing)
    answer_start = time.time()
    ai_answer = generate_ai_answer_from_results(
        query, final_results, "phi3:mini")
    answer_time = round((time.time() - answer_start) * 1000, 1)

    # Calculate total time
    total_time = round((time.time() - start_time) * 1000, 1)

    return {
        "query": query,
        "method": "Hybrid Search (Semantic + Keyword)",
        "model": "phi3:mini",
        "backend": backend.upper(),

        # Core results
        "ai_response": ai_answer,
        "sources": final_results,
        "results": final_results,
        "total_results": len(final_results),

        # Hybrid search specific details
        "fusion_method": fusion_method_used,
        "semantic_weight": semantic_weight,
        "keyword_weight": round(1 - semantic_weight, 1),
        "use_rrf": use_rrf,

        # Search result breakdowns
        "semantic_results_count": len(semantic_results),
        "keyword_results_count": len(keyword_results),
        "semantic_max_score": max([r.get('score', 0) for r in semantic_results]) if semantic_results else 0,
        "keyword_max_score": max([s for _, s in keyword_results]) if keyword_results else 0,

        # Comprehensive timing breakdown
        "timing": {
            "total_time_ms": total_time,
            "semantic_search_ms": semantic_time,
            "keyword_search_ms": keyword_time,
            "fusion_ms": fusion_time,
            "answer_generation_ms": answer_time
        },

        # Process steps for debugging/learning
        "process_steps": [
            {
                "step": 1,
                "name": "Búsqueda Semántica Vectorial",
                "description": f"Búsqueda vectorial usando embeddings para capturar significado semántico en {backend}",
                "time_ms": semantic_time,
                "output": f"Fragmentos encontrados: {len(semantic_results)} (max score: {max([r.get('score', 0) for r in semantic_results]) if semantic_results else 0:.3f})"
            },
            {
                "step": 2,
                "name": "Búsqueda Léxica BM25",
                "description": "Búsqueda BM25 para coincidencias exactas de palabras clave y términos específicos",
                "time_ms": keyword_time,
                "output": f"Fragmentos re-puntuados: {len(keyword_results)} (max BM25 score: {max([s for _, s in keyword_results]) if keyword_results else 0:.3f})"
            },
            {
                "step": 3,
                "name": "Fusión Híbrida",
                "description": f"Combinación de puntuaciones usando {fusion_method_used}",
                "time_ms": fusion_time,
                "output": f"Método: {fusion_method_used} → {len(final_results)} resultados finales"
            },
            {
                "step": 4,
                "name": "Generación de Respuesta",
                "description": "Se generó respuesta final combinando fortalezas de búsqueda semántica y léxica",
                "time_ms": answer_time,
                "output": f"Respuesta basada en {len(final_results)} fragmentos híbridos"
            }
        ],

        # Technical metadata
        "technical_details": {
            "bm25_parameters": {"k1": 1.5, "b": 0.75},
            "rrf_k_parameter": 60 if use_rrf else None,
            "semantic_expansion_factor": 3,  # k*3 for broader semantic search
            "keyword_expansion_factor": 3,   # k*3 for broader keyword search
            "embedding_model": "multilingual-e5-base",
            "filters_applied": filters or {},
            "normalization_applied": "max-score normalization" if not use_rrf else "rank-based (RRF)"
        }
    }


# ============================================================================
# 5. MULTI-ROUND (ITERATIVE) RETRIEVAL
# ============================================================================

def iterative_retrieval(
    query: str,
    backend: str = "qdrant",
    k: int = 5,
    max_rounds: int = 3,
    model: str = "phi3:mini",
    filters: Optional[Dict] = None,
    collection_suffix: str = None
) -> Dict[str, Any]:
    """
    Perform multi-round iterative retrieval with query refinement.
    Enhanced with comprehensive timing and debugging information.

    Args:
        query: Original user query
        backend: Vector database backend
        k: Number of results per round
        max_rounds: Maximum number of retrieval rounds
        model: LLM model for query refinement and synthesis
        filters: Optional metadata filters

    Returns:
        Results from all rounds with final synthesized answer and detailed timing
    """
    import time
    start_time = time.time()

    if ollama is None:
        logger.warning(
            "Ollama not available, falling back to single-round search")
        if embed_query_clean:
            emb = embed_query_clean(query)
        else:
            emb = embed_e5([query], is_query=True)[0]
        results = BACKENDS[backend](
            emb, k=k, where=filters, collection_suffix=collection_suffix)
        return {
            "query": query,
            "method": "Single-Round (Ollama unavailable)",
            "rounds": [{"round": 1, "query": query, "results": results}],
            "final_answer": "Servicio Ollama no disponible para búsqueda iterativa"
        }

    all_rounds = []
    accumulated_context = []
    current_query = query
    round_timings = []

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=ollama_host)

    for round_num in range(1, max_rounds + 1):
        round_start = time.time()

        # Perform search for current query
        search_start = time.time()
        if embed_query_clean:
            emb = embed_query_clean(current_query)
        else:
            emb = embed_e5([current_query], is_query=True)[0]

        results = BACKENDS[backend](
            emb, k=k, where=filters, collection_suffix=collection_suffix)
        accumulated_context.extend(results)
        search_time = round((time.time() - search_start) * 1000, 1)

        # Evaluate information sufficiency
        evaluation_time = 0
        needs_refinement = False
        refined_query = None

        if round_num < max_rounds:
            eval_start = time.time()
            context_snippet = build_context(
                accumulated_context[-6:], max_tokens=800)

            evaluation_prompt = f"""Analiza si los siguientes fragmentos contienen información suficiente para responder completamente la pregunta original.

Pregunta original: {query}

Fragmentos disponibles hasta ahora:
{context_snippet}

INSTRUCCIONES:
- Responde SOLO con una de estas opciones:
- "SUFICIENTE" si hay información completa para responder la pregunta
- "INSUFICIENTE: [pregunta específica]" si falta información, seguido de UNA pregunta refinada muy específica para buscar lo que falta

Ejemplo de respuesta válida:
INSUFICIENTE: ¿Cuáles son las ventajas específicas de PostgreSQL sobre otros sistemas?

Tu evaluación:
"""

            try:
                eval_response = client.generate(
                    model=model,
                    prompt=evaluation_prompt,
                    options={"temperature": 0.2, "num_predict": 100}
                )

                evaluation = eval_response.get('response', '').strip()
                evaluation_time = round((time.time() - eval_start) * 1000, 1)

                if evaluation.startswith("SUFICIENTE"):
                    logger.info(
                        f"Sufficient information found in round {round_num}")
                elif evaluation.startswith("INSUFICIENTE"):
                    # Extract refined query
                    if ":" in evaluation:
                        refined_query = evaluation.split(":", 1)[1].strip()
                        current_query = refined_query
                        needs_refinement = True
                        logger.info(
                            f"Refining query for round {round_num + 1}: {refined_query}")
                else:
                    # If unclear, stop refinement
                    pass

            except Exception as e:
                logger.error(f"Evaluation failed in round {round_num}: {e}")
                evaluation = "Error en evaluación"

        round_time = round((time.time() - round_start) * 1000, 1)
        round_timings.append(round_time)

        # Store round results with detailed information
        all_rounds.append({
            "round": round_num,
            "query": current_query,
            "is_original_query": current_query == query,
            "results": results[:3],  # Top 3 per round for display
            "total_found": len(results),
            "search_time_ms": search_time,
            "evaluation_time_ms": evaluation_time,
            "total_round_time_ms": round_time,
            "evaluation_result": evaluation if round_num < max_rounds else "Final round",
            "needs_refinement": needs_refinement,
            "refined_query": refined_query
        })

        # Stop if we have sufficient information or reached max rounds
        if not needs_refinement or round_num == max_rounds:
            break

    # Deduplicate accumulated context
    dedup_start = time.time()
    unique_docs = {
        f"{doc.get('path', '')}:{doc.get('chunk_id', doc.get('page', ''))}": doc
        for doc in accumulated_context
    }

    # Apply MMR reranking on all accumulated results
    if mmr:
        candidates = [
            {
                'content': doc['content'],
                'sim': doc.get('score', 0.5),
                'path': doc['path'],
                'chunk_id': doc.get('chunk_id', ''),
                'page': doc.get('page'),
                'metadata': doc.get('metadata', {})
            }
            for doc in unique_docs.values()
        ]
        final_results = mmr(query, candidates, lambda_=0.7, top_k=k)
    else:
        final_results = list(unique_docs.values())[:k]

    dedup_time = round((time.time() - dedup_start) * 1000, 1)

    # Generate final synthesized answer
    synthesis_start = time.time()
    final_answer = None
    try:
        final_context = build_context(final_results, max_tokens=1800)

        synthesis_prompt = f"""Responde la siguiente pregunta usando TODA la información recopilada en múltiples rondas de búsqueda iterativa.

Pregunta original: {query}

Información recopilada en {len(all_rounds)} ronda(s) de búsqueda:
{final_context}

INSTRUCCIONES ESTRICTAS:
- SOLO usa información que aparezca textualmente en los fragmentos
- Integra información de todas las fuentes disponibles de manera coherente
- Usa fechas, números y nombres exactos tal como aparecen
- Si alguna parte de la pregunta no puede responderse con los fragmentos, indica claramente: "Esta información no está disponible en los documentos consultados"
- Mantén un tono académico y profesional en español
- NO inventes o extrapoles información no presente

Respuesta integral basada en búsqueda iterativa:
"""

        response = client.generate(
            model=model,
            prompt=synthesis_prompt,
            options={
                "temperature": 0.1,  # Very low for factual synthesis
                "num_predict": 600,
                "top_p": 0.8,
                "repeat_penalty": 1.1
            }
        )

        final_answer = response.get('response', '').strip()

    except Exception as e:
        logger.error(f"Final synthesis failed: {e}")
        final_answer = f"Error en la síntesis final: {str(e)}"

    synthesis_time = round((time.time() - synthesis_start) * 1000, 1)

    # Use synthesized answer or fallback
    ai_answer = final_answer or generate_ai_answer_from_results(
        query, final_results, model)

    # Calculate total time
    total_time = round((time.time() - start_time) * 1000, 1)

    return {
        "query": query,
        "method": "Multi-Round Iterative Retrieval",
        "model": model,
        "backend": backend.upper(),

        # Core results
        "ai_response": ai_answer,
        "sources": final_results,
        "results": final_results,
        "total_results": len(final_results),

        # Iterative process details
        "total_rounds": len(all_rounds),
        "rounds": all_rounds,
        "total_unique_documents": len(unique_docs),
        "iterations": len(all_rounds),  # For template compatibility

        # Comprehensive timing breakdown
        "timing": {
            "total_time_ms": total_time,
            "round_times_ms": round_timings,
            "deduplication_time_ms": dedup_time,
            "synthesis_time_ms": synthesis_time,
            "average_round_time_ms": round(sum(round_timings) / len(round_timings), 1) if round_timings else 0
        },

        # Process steps for debugging/learning
        "process_steps": [
            {
                "step": 1,
                "name": "Búsqueda Inicial",
                "description": f"Primera ronda con la consulta original: '{query}'",
                "time_ms": round_timings[0] if round_timings else 0,
                "output": f"Fragmentos encontrados: {all_rounds[0]['total_found'] if all_rounds else 0}"
            },
            {
                "step": 2,
                "name": "Análisis Iterativo de Brechas",
                "description": f"Se ejecutaron {len(all_rounds)} rondas evaluando suficiencia de información",
                "time_ms": sum(round['evaluation_time_ms'] for round in all_rounds),
                "output": f"Refinamientos necesarios: {sum(1 for round in all_rounds if round.get('needs_refinement'))}"
            },
            {
                "step": 3,
                "name": "Deduplicación y Re-ranking",
                "description": f"Se aplicó MMR sobre {len(unique_docs)} documentos únicos",
                "time_ms": dedup_time,
                "output": f"Documentos finales seleccionados: {len(final_results)}"
            },
            {
                "step": 4,
                "name": "Síntesis Final",
                "description": f"Se generó respuesta integral usando {model} con toda la información recopilada",
                "time_ms": synthesis_time,
                "output": f"Respuesta basada en {len(all_rounds)} rondas de búsqueda"
            }
        ],

        # Technical metadata
        "technical_details": {
            "max_rounds_configured": max_rounds,
            "rounds_executed": len(all_rounds),
            "early_termination": len(all_rounds) < max_rounds,
            "query_refinements": [r.get('refined_query') for r in all_rounds if r.get('refined_query')],
            "evaluation_strategy": "LLM-based sufficiency analysis",
            "mmr_applied": bool(mmr),
            "filters_applied": filters or {},
            "embedding_model": "multilingual-e5-base"
        }
    }
