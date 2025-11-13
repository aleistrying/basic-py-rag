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

logger = logging.getLogger(__name__)

BACKENDS = {"qdrant": search_qdrant, "pgvector": search_pgvector}

# Improved prompt template for AI answer generation
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


def generate_ai_answer_from_results(
    query: str,
    results: List[Dict],
    model: str = "phi3:mini"
) -> str:
    """Generate AI answer from search results using LLM."""
    if not ollama or not results:
        return "No se pudo generar una respuesta AI."

    # Build context from results
    context = build_context(results, max_tokens=1500)
    if not context.strip():
        return "No se encontró contexto suficiente para generar una respuesta."

    # Generate answer using LLM
    prompt = PROMPT_TEMPLATE.format(q=query, sources=context)

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.3,
                "num_predict": 300
            }
        )

        return response['response'].strip()

    except Exception as e:
        logger.error(f"Error generating AI answer: {e}")
        return f"Error generando respuesta: {str(e)}"


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
        variations = [
            line.strip()
            for line in variations_text.split('\n')
            if line.strip() and not line.strip().startswith(('-', '*', '•', str(i)))
            for i in range(10)  # Remove numbered lines
        ][:num_variations]

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
            # Create unique key for document (use content hash or chunk_id)
            doc_key = f"{doc.get('path', '')}:{doc.get('chunk_id', doc.get('page', rank))}"

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
    filters: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Perform multi-query search with query rephrasing and RRF fusion.

    Args:
        query: Original user query
        backend: Vector database backend
        k: Number of results per query
        num_variations: Number of query variations to generate
        model: LLM model for query rephrasing
        filters: Optional metadata filters

    Returns:
        Enhanced search results with RRF scores
    """
    # Generate query variations
    queries = generate_query_variations(query, model, num_variations)

    # Perform search for each query variation
    all_results = []
    for q in queries:
        # Embed and search
        if embed_query_clean:
            emb = embed_query_clean(q)
        else:
            emb = embed_e5([q], is_query=True)[0]

        results = BACKENDS[backend](emb, k=k, where=filters)
        all_results.append(results)

    # Apply RRF fusion
    fused_results = reciprocal_rank_fusion(all_results, k=60)

    # Generate AI answer from fused results
    ai_answer = generate_ai_answer_from_results(
        query, fused_results[:k], model)

    return {
        "query": query,
        "query_variations": queries[1:],  # Exclude original
        "backend": backend.upper(),
        "method": "Multi-Query + RRF",
        "model": model,
        "total_queries": len(queries),
        "ai_response": ai_answer,
        "sources": fused_results[:k],
        "results": fused_results[:k],
        "total_results": len(fused_results[:k])
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
    synthesize: bool = True
) -> Dict[str, Any]:
    """
    Perform search using query decomposition strategy.

    Args:
        query: Complex user query
        backend: Vector database backend
        k: Number of results per sub-question
        model: LLM model for decomposition and synthesis
        filters: Optional metadata filters
        synthesize: Whether to synthesize final answer from sub-results

    Returns:
        Comprehensive results with synthesis
    """
    # Decompose query
    sub_questions = decompose_query(query, model)

    # Search for each sub-question
    sub_results = []
    all_documents = []

    for sub_q in sub_questions:
        # Embed and search
        if embed_query_clean:
            emb = embed_query_clean(sub_q)
        else:
            emb = embed_e5([sub_q], is_query=True)[0]

        results = BACKENDS[backend](emb, k=k, where=filters)
        sub_results.append({
            "sub_question": sub_q,
            "results": results[:3]  # Top 3 per sub-question
        })
        all_documents.extend(results)

    # Deduplicate and rerank all documents
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

    # Synthesize final answer if requested
    synthesized_answer = None
    if synthesize and ollama:
        try:
            context = build_context(final_results, max_tokens=1500)
            synthesis_prompt = f"""Responde la siguiente pregunta compleja usando SOLO la información de los fragmentos proporcionados.

Pregunta original: {query}

Sub-preguntas analizadas:
{chr(10).join(f"- {sq}" for sq in sub_questions)}

Fragmentos relevantes:
{context}

Instrucciones:
- Proporciona una respuesta completa y coherente
- Integra información de todos los sub-temas relevantes
- Usa fechas, números y nombres exactos de los fragmentos
- Si alguna parte no se puede responder con los fragmentos, indícalo claramente
- Mantén un tono académico y profesional

Respuesta completa:
"""

            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)

            response = client.generate(
                model=model,
                prompt=synthesis_prompt,
                options={
                    "temperature": 0.5,
                    "num_predict": 400
                }
            )

            synthesized_answer = response.get('response', '').strip()

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")

    # Use synthesized answer or fallback to generated answer
    ai_answer = synthesized_answer or generate_ai_answer_from_results(
        query, final_results, model)

    return {
        "query": query,
        "method": "Query Decomposition",
        "model": model,
        "sub_questions": sub_questions,
        "sub_results": sub_results,
        "ai_response": ai_answer,
        "backend": backend.upper(),
        "total_unique_documents": len(unique_docs),
        "sources": final_results,
        "results": final_results,
        "total_results": len(final_results)
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
    generate_final_answer: bool = True
) -> Dict[str, Any]:
    """
    Perform search using Hypothetical Document Embeddings (HyDE).

    Args:
        query: User query
        backend: Vector database backend
        k: Number of results to return
        model: LLM model for generating hypothetical document
        filters: Optional metadata filters
        generate_final_answer: Whether to generate final answer with real docs

    Returns:
        Search results using HyDE approach
    """
    # Generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(query, model)

    # Embed the hypothetical document (not the query!)
    if embed_query_clean:
        # Note: We embed as document, not as query
        hyp_embedding = embed_e5([hypothetical_doc], is_query=False)[0]
    else:
        hyp_embedding = embed_e5([hypothetical_doc], is_query=False)[0]

    # Search using hypothetical document embedding
    results = BACKENDS[backend](hyp_embedding, k=k, where=filters)

    # Generate final answer with real documents if requested
    final_answer = None
    if generate_final_answer and ollama and results:
        try:
            context = build_context(results, max_tokens=1500)
            answer_prompt = f"""Responde la siguiente pregunta usando SOLO la información de los fragmentos de documentos reales proporcionados.

Pregunta: {query}

Fragmentos de documentos:
{context}

Instrucciones:
- Usa solo información de los fragmentos
- Sé específico con fechas, números y nombres
- Si la información no está disponible, indícalo
- Mantén tono académico

Respuesta:
"""

            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)

            response = client.generate(
                model=model,
                prompt=answer_prompt,
                options={
                    "temperature": 0.5,
                    "num_predict": 300
                }
            )

            final_answer = response.get('response', '').strip()

        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")

    # Use generated answer or fallback
    ai_answer = final_answer or generate_ai_answer_from_results(
        query, results, model)

    return {
        "query": query,
        "method": "HyDE (Hypothetical Document Embeddings)",
        "model": model,
        "hypothetical_document": hypothetical_doc,
        "backend": backend.upper(),
        "ai_response": ai_answer,
        "sources": results,
        "results": results,
        "total_results": len(results)
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
    use_rrf: bool = True
) -> Dict[str, Any]:
    """
    Perform hybrid search combining semantic (vector) and keyword (BM25) search.

    Args:
        query: User query
        backend: Vector database backend
        k: Number of final results
        semantic_weight: Weight for semantic search (0-1), keyword = 1 - semantic_weight
        filters: Optional metadata filters
        use_rrf: Whether to use RRF fusion (vs weighted score)

    Returns:
        Hybrid search results
    """
    # Perform semantic search
    if embed_query_clean:
        emb = embed_query_clean(query)
    else:
        emb = embed_e5([query], is_query=True)[0]

    semantic_results = BACKENDS[backend](emb, k=k*3, where=filters)

    # Perform keyword search on the same set
    keyword_results = bm25_search(query, semantic_results, k=k*3)
    keyword_results_dict = {
        f"{doc.get('path', '')}:{doc.get('chunk_id', doc.get('page', ''))}": score
        for doc, score in keyword_results
    }

    if use_rrf:
        # Use RRF to combine rankings
        semantic_list = [[r] for r in semantic_results[:k*2]]
        keyword_list = [[doc] for doc, _ in keyword_results[:k*2]]

        fused_results = reciprocal_rank_fusion(
            [semantic_results[:k*2], [doc for doc, _ in keyword_results[:k*2]]])
        final_results = fused_results[:k]
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

    # Generate AI answer from hybrid results
    ai_answer = generate_ai_answer_from_results(
        query, final_results, "phi3:mini")

    return {
        "query": query,
        "method": "Hybrid Search (Semantic + Keyword)",
        "model": "phi3:mini",
        "fusion_method": "RRF" if use_rrf else "Weighted Score",
        "semantic_weight": semantic_weight,
        "backend": backend.upper(),
        "ai_response": ai_answer,
        "sources": final_results,
        "results": final_results,
        "total_results": len(final_results)
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
    filters: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Perform multi-round iterative retrieval with query refinement.

    Args:
        query: Original user query
        backend: Vector database backend
        k: Number of results per round
        max_rounds: Maximum number of retrieval rounds
        model: LLM model for query refinement and synthesis
        filters: Optional metadata filters

    Returns:
        Results from all rounds with final synthesized answer
    """
    if ollama is None:
        logger.warning(
            "Ollama not available, falling back to single-round search")
        if embed_query_clean:
            emb = embed_query_clean(query)
        else:
            emb = embed_e5([query], is_query=True)[0]
        results = BACKENDS[backend](emb, k=k, where=filters)
        return {
            "query": query,
            "method": "Single-Round (Ollama unavailable)",
            "rounds": [{"round": 1, "query": query, "results": results}],
            "final_answer": None
        }

    all_rounds = []
    accumulated_context = []
    current_query = query

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.Client(host=ollama_host)

    for round_num in range(1, max_rounds + 1):
        # Perform search for current query
        if embed_query_clean:
            emb = embed_query_clean(current_query)
        else:
            emb = embed_e5([current_query], is_query=True)[0]

        results = BACKENDS[backend](emb, k=k, where=filters)
        accumulated_context.extend(results)

        # Store round results
        all_rounds.append({
            "round": round_num,
            "query": current_query,
            "results": results[:3]  # Top 3 per round
        })

        # Check if we have enough information (stop early if yes)
        if round_num < max_rounds:
            context_snippet = build_context(
                accumulated_context[-6:], max_tokens=800)

            evaluation_prompt = f"""Analiza si los siguientes fragmentos contienen suficiente información para responder completamente la pregunta original.

Pregunta original: {query}

Fragmentos disponibles:
{context_snippet}

Responde SOLO con una de estas opciones:
- "SUFICIENTE" si hay información completa para responder
- "INSUFICIENTE: [pregunta más específica]" si falta información, seguido de una pregunta refinada para buscar lo que falta

Tu respuesta:
"""

            try:
                eval_response = client.generate(
                    model=model,
                    prompt=evaluation_prompt,
                    options={"temperature": 0.3, "num_predict": 100}
                )

                evaluation = eval_response.get('response', '').strip()

                if evaluation.startswith("SUFICIENTE"):
                    logger.info(
                        f"Sufficient information found in round {round_num}")
                    break
                elif evaluation.startswith("INSUFICIENTE"):
                    # Extract refined query
                    refined_query = evaluation.split(
                        ":", 1)[1].strip() if ":" in evaluation else query
                    current_query = refined_query
                    logger.info(
                        f"Refining query for round {round_num + 1}: {refined_query}")
                else:
                    # If unclear, continue with original query
                    break

            except Exception as e:
                logger.error(f"Evaluation failed in round {round_num}: {e}")
                break

    # Deduplicate accumulated context
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

    # Generate final synthesized answer
    final_answer = None
    try:
        final_context = build_context(final_results, max_tokens=1800)

        synthesis_prompt = f"""Responde la siguiente pregunta usando TODA la información recopilada en múltiples rondas de búsqueda.

Pregunta original: {query}

Información recopilada en {len(all_rounds)} ronda(s):
{final_context}

Instrucciones:
- Proporciona una respuesta completa y bien estructurada
- Integra información de todas las fuentes disponibles
- Usa fechas, números y nombres exactos
- Si alguna parte de la pregunta no puede responderse, indícalo claramente
- Mantén un tono académico y profesional

Respuesta final:
"""

        response = client.generate(
            model=model,
            prompt=synthesis_prompt,
            options={
                "temperature": 0.5,
                "num_predict": 500
            }
        )

        final_answer = response.get('response', '').strip()

    except Exception as e:
        logger.error(f"Final synthesis failed: {e}")

    # Use synthesized answer or fallback
    ai_answer = final_answer or generate_ai_answer_from_results(
        query, final_results, model)

    return {
        "query": query,
        "method": "Multi-Round Iterative Retrieval",
        "model": model,
        "backend": backend.upper(),
        "total_rounds": len(all_rounds),
        "rounds": all_rounds,
        "total_unique_documents": len(unique_docs),
        "ai_response": ai_answer,
        "sources": final_results,
        "results": final_results,
        "total_results": len(final_results)
    }
