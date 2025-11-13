"""
Orchestrated RAG Pipeline - Intelligent Multi-Technique System

This module implements an intelligent orchestrator that automatically:
1. Runs baseline hybrid search (always)
2. Checks answerability
3. Conditionally applies advanced techniques based on query characteristics
4. Uses RRF fusion throughout
5. Stops early when sufficient information found

Based on battle-tested ordering and fast heuristics.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

try:
    import ollama
except ImportError:
    ollama = None

try:
    from scripts.query_embed import embed_query as embed_query_clean, embed_e5
except ImportError:
    embed_query_clean = None
    embed_e5 = None

from app.qdrant_backend import search_qdrant
from app.pgvector_backend import search_pgvector
from app.rerank import mmr, build_context
from app.advanced_rag import (
    reciprocal_rank_fusion,
    generate_query_variations,
    decompose_query,
    generate_hypothetical_document,
    bm25_search
)

logger = logging.getLogger(__name__)

BACKENDS = {"qdrant": search_qdrant, "pgvector": search_pgvector}


# ============================================================================
# SIGNAL DETECTION - Fast Heuristics
# ============================================================================

def compute_score_gap(results: List[Dict]) -> float:
    """Calculate gap between top and 10th result (or last if < 10)."""
    if len(results) < 2:
        return 1.0

    scores = [r.get('score', 0) for r in results]
    top_score = max(scores) if scores else 0
    bottom_score = scores[min(9, len(scores) - 1)
                          ] if len(scores) > 1 else top_score

    return top_score - bottom_score


def is_query_short(query: str) -> bool:
    """Check if query is very short (≤ 3 meaningful tokens)."""
    tokens = [t for t in re.findall(r'\w+', query.lower()) if len(t) > 2]
    return len(tokens) <= 3


def is_query_abstract(query: str) -> bool:
    """Check if query uses abstract/vague terms."""
    abstract_terms = {
        'overview', 'resumen', 'introducción', 'introduccion', 'guidelines',
        'guía', 'guia', 'explicación', 'explicacion', 'concepto', 'teoría',
        'teoria', 'general', 'básico', 'basico'
    }
    query_lower = query.lower()
    return any(term in query_lower for term in abstract_terms)


def has_conjunctions(query: str) -> bool:
    """Check if query has multiple parts (and, or, vs, etc.)."""
    conjunctions = r'\b(y|and|o|or|pero|but|vs|versus|además|ademas|también|tambien)\b'
    return bool(re.search(conjunctions, query.lower()))


def has_enumerations(query: str) -> bool:
    """Check if query lists things."""
    enum_patterns = [
        r'\d+[.)]\s',  # 1. or 1)
        r'[a-z][.)]\s',  # a. or a)
        r',.*,',  # Multiple commas
        r'\by\b.*\by\b'  # Multiple "y"
    ]
    return any(re.search(pattern, query) for pattern in enum_patterns)


def check_answerability(
    query: str,
    context: str,
    model: str = "phi3:mini"
) -> Dict[str, Any]:
    """
    Fast LLM check: Can we answer with current context?

    Returns:
        {
            "answerable": bool,
            "confidence": float (0-1),
            "missing_fields": List[str],
            "reasoning": str
        }
    """
    if ollama is None:
        # Fallback: use heuristics
        return {
            "answerable": len(context) > 200,
            "confidence": 0.5,
            "missing_fields": [],
            "reasoning": "LLM unavailable, using heuristics"
        }

    prompt = f"""Analiza si puedes responder completamente la siguiente pregunta con el contexto disponible.

Pregunta: {query}

Contexto disponible:
{context[:1500]}

Responde en este formato EXACTO:
RESPUESTA: [SI/NO/PARCIAL]
CONFIANZA: [0-100]
FALTA: [lista separada por comas de información que falta, o "nada"]
RAZON: [una frase corta explicando por qué]

Tu análisis:
"""

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.2,
                "num_predict": 150
            }
        )

        text = response.get('response', '').strip()

        # Parse response
        answerable = "SI" in text.upper() and "NO" not in text.split(
            "RESPUESTA:")[1].split("\n")[0].upper()

        confidence_match = re.search(r'CONFIANZA:\s*(\d+)', text)
        confidence = float(confidence_match.group(1)) / \
            100 if confidence_match else 0.5

        missing_match = re.search(r'FALTA:\s*(.+?)(?:\n|$)', text)
        missing_raw = missing_match.group(1) if missing_match else "nada"
        missing_fields = [m.strip() for m in missing_raw.split(
            ',') if m.strip().lower() != 'nada']

        reason_match = re.search(r'RAZON:\s*(.+?)(?:\n|$)', text)
        reasoning = reason_match.group(
            1) if reason_match else "Analysis completed"

        return {
            "answerable": answerable and confidence > 0.6,
            "confidence": confidence,
            "missing_fields": missing_fields,
            "reasoning": reasoning
        }

    except Exception as e:
        logger.error(f"Answerability check failed: {e}")
        return {
            "answerable": len(context) > 200,
            "confidence": 0.5,
            "missing_fields": [],
            "reasoning": f"Error: {str(e)}"
        }


def propose_followup_query(
    original_query: str,
    current_context: str,
    missing_fields: List[str],
    model: str = "phi3:mini"
) -> Optional[str]:
    """Generate a targeted followup query to fill gaps."""
    if ollama is None or not missing_fields:
        return None

    missing_str = ", ".join(missing_fields[:3])

    prompt = f"""Genera UNA pregunta específica y concreta para buscar la información que falta.

Pregunta original: {original_query}
Información que falta: {missing_str}

Responde SOLO con la nueva pregunta específica, nada más.

Pregunta específica:
"""

    try:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.4,
                "num_predict": 50
            }
        )

        followup = response.get('response', '').strip()
        return followup if followup and len(followup) > 5 else None

    except Exception as e:
        logger.error(f"Followup query generation failed: {e}")
        return None


# ============================================================================
# ORCHESTRATED PIPELINE - Main Intelligence
# ============================================================================

def retrieve_hybrid_baseline(
    query: str,
    backend: str = "qdrant",
    k: int = 50,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Phase 1: Always-on baseline - Hybrid search (BM25 + Dense) with RRF.
    This is the foundation that always runs.
    """
    # Dense semantic search
    if embed_query_clean:
        emb = embed_query_clean(query)
    else:
        emb = embed_e5([query], is_query=True)[0]

    semantic_results = BACKENDS[backend](emb, k=k, where=filters)

    # BM25 keyword search on same results
    keyword_results = bm25_search(query, semantic_results, k=k)

    # RRF fusion
    keyword_as_dicts = [doc for doc, _ in keyword_results]
    fused = reciprocal_rank_fusion([semantic_results, keyword_as_dicts], k=60)

    return fused


def orchestrated_rag_pipeline(
    query: str,
    backend: str = "qdrant",
    k: int = 10,
    model: str = "phi3:mini",
    filters: Optional[Dict] = None,
    max_retrieval_calls: int = 8,
    max_rounds: int = 2,
    early_exit: bool = True
) -> Dict[str, Any]:
    """
    Intelligent orchestrated RAG pipeline with automatic technique selection.

    Algorithm:
    1. Phase 0: Preflight (normalize, detect language)
    2. Phase 1: Baseline (Hybrid + RRF, always)
    3. Check answerability
    4. Phase 2: Conditional enrichment (MQ, HyDE, QD based on signals)
    5. Phase 3: Iterative refinement (if needed)

    Args:
        query: User query
        backend: Vector database
        k: Final number of results
        model: LLM model
        filters: Metadata filters
        max_retrieval_calls: Budget limit
        max_rounds: Max iterative rounds
        early_exit: Stop as soon as answerable

    Returns:
        Comprehensive results with execution trace
    """
    execution_trace = []
    retrieval_call_count = 0
    all_results_pool = []  # Accumulated results from all phases

    # ========== PHASE 0: PREFLIGHT ==========
    execution_trace.append({
        "phase": "0_preflight",
        "action": "Query normalization and analysis",
        "query": query
    })

    # Detect query characteristics (fast heuristics)
    is_short = is_query_short(query)
    is_abstract = is_query_abstract(query)
    has_conj = has_conjunctions(query)
    has_enum = has_enumerations(query)

    signals = {
        "is_short": is_short,
        "is_abstract": is_abstract,
        "has_conjunctions": has_conj,
        "has_enumerations": has_enum,
        "query_length": len(query.split())
    }

    execution_trace.append({
        "phase": "0_preflight",
        "action": "Signal detection",
        "signals": signals
    })

    # ========== PHASE 1: BASELINE (ALWAYS) ==========
    logger.info("Phase 1: Running baseline hybrid search...")

    baseline_results = retrieve_hybrid_baseline(
        query, backend, k=50, filters=filters)
    retrieval_call_count += 2  # Semantic + BM25
    all_results_pool.extend(baseline_results[:k*3])

    execution_trace.append({
        "phase": "1_baseline",
        "action": "Hybrid search (BM25 + Dense) with RRF",
        "results_count": len(baseline_results),
        "top_score": baseline_results[0].get('score', 0) if baseline_results else 0
    })

    # Build initial context and check answerability
    context = build_context(baseline_results[:k], max_tokens=1500)
    answerability = check_answerability(query, context, model)

    execution_trace.append({
        "phase": "1_baseline",
        "action": "Answerability check",
        "result": answerability
    })

    # Early exit if sufficient
    if early_exit and answerability["answerable"] and answerability["confidence"] > 0.7:
        logger.info("✅ Early exit: Query answerable with baseline")

        # Apply MMR reranking to final results
        final_results = rerank_final_results(all_results_pool, query, k)

        return {
            "query": query,
            "method": "Orchestrated RAG (Early Exit)",
            "techniques_used": ["hybrid_baseline"],
            "retrieval_calls": retrieval_call_count,
            "execution_trace": execution_trace,
            "signals": signals,
            "answerability": answerability,
            "results": final_results[:k],
            "context_preview": context[:500]
        }

    # ========== PHASE 2: CONDITIONAL ENRICHMENT ==========

    # Compute score gap for fuzzy query detection
    score_gap = compute_score_gap(baseline_results[:10])

    execution_trace.append({
        "phase": "2_enrichment",
        "action": "Score gap analysis",
        "score_gap": round(score_gap, 4),
        "threshold": 0.05
    })

    techniques_used = ["hybrid_baseline"]

    # A. Multi-Query Rephrasing (if query is fuzzy/ambiguous)
    if (score_gap < 0.05 or is_short or len(query.split()) < 5) and retrieval_call_count < max_retrieval_calls:
        logger.info("🔄 Applying Multi-Query rephrasing...")

        variants = generate_query_variations(query, model, num_variations=2)

        for variant in variants[1:]:  # Skip original
            if retrieval_call_count >= max_retrieval_calls:
                break

            variant_results = retrieve_hybrid_baseline(
                variant, backend, k=30, filters=filters)
            retrieval_call_count += 2
            all_results_pool.extend(variant_results[:k*2])

        techniques_used.append("multi_query")

        execution_trace.append({
            "phase": "2_enrichment",
            "action": "Multi-Query rephrasing applied",
            "variants_generated": len(variants) - 1,
            "reason": "Low score gap or short query"
        })

        # Re-check answerability
        fused_results = deduplicate_and_fuse(all_results_pool)
        context = build_context(fused_results[:k], max_tokens=1500)
        answerability = check_answerability(query, context, model)

        if early_exit and answerability["answerable"] and answerability["confidence"] > 0.75:
            logger.info("✅ Early exit after Multi-Query")
            final_results = rerank_final_results(fused_results, query, k)

            return {
                "query": query,
                "method": "Orchestrated RAG (Multi-Query Exit)",
                "techniques_used": techniques_used,
                "retrieval_calls": retrieval_call_count,
                "execution_trace": execution_trace,
                "signals": signals,
                "answerability": answerability,
                "results": final_results[:k],
                "context_preview": context[:500]
            }

    # B. HyDE (if abstract or domain gap)
    if (is_abstract or is_short) and retrieval_call_count < max_retrieval_calls:
        logger.info("📄 Applying HyDE...")

        hypothetical_doc = generate_hypothetical_document(query, model)

        if hypothetical_doc and len(hypothetical_doc) > 50:
            # Embed as document (not query)
            hyde_emb = embed_e5([hypothetical_doc], is_query=False)[0]
            hyde_results = BACKENDS[backend](hyde_emb, k=30, where=filters)
            retrieval_call_count += 1
            all_results_pool.extend(hyde_results[:k*2])

            techniques_used.append("hyde")

            execution_trace.append({
                "phase": "2_enrichment",
                "action": "HyDE applied",
                "hypothetical_doc_length": len(hypothetical_doc),
                "reason": "Abstract/short query"
            })

            # Re-check
            fused_results = deduplicate_and_fuse(all_results_pool)
            context = build_context(fused_results[:k], max_tokens=1500)
            answerability = check_answerability(query, context, model)

            if early_exit and answerability["answerable"] and answerability["confidence"] > 0.75:
                logger.info("✅ Early exit after HyDE")
                final_results = rerank_final_results(fused_results, query, k)

                return {
                    "query": query,
                    "method": "Orchestrated RAG (HyDE Exit)",
                    "techniques_used": techniques_used,
                    "retrieval_calls": retrieval_call_count,
                    "execution_trace": execution_trace,
                    "signals": signals,
                    "answerability": answerability,
                    "results": final_results[:k],
                    "context_preview": context[:500],
                    "hypothetical_document": hypothetical_doc[:300]
                }

    # C. Query Decomposition (if compound query)
    if (has_conj or has_enum or len(answerability.get("missing_fields", [])) > 1) and retrieval_call_count < max_retrieval_calls:
        logger.info("🧩 Applying Query Decomposition...")

        sub_questions = decompose_query(query, model)

        if len(sub_questions) > 1:
            for sub_q in sub_questions[:4]:  # Max 4 sub-questions
                if retrieval_call_count >= max_retrieval_calls:
                    break

                sub_results = retrieve_hybrid_baseline(
                    sub_q, backend, k=20, filters=filters)
                retrieval_call_count += 2
                all_results_pool.extend(sub_results[:k])

            techniques_used.append("query_decomposition")

            execution_trace.append({
                "phase": "2_enrichment",
                "action": "Query Decomposition applied",
                "sub_questions": sub_questions,
                "reason": "Compound query or multiple missing fields"
            })

            # Re-check
            fused_results = deduplicate_and_fuse(all_results_pool)
            context = build_context(fused_results[:k], max_tokens=1500)
            answerability = check_answerability(query, context, model)

            if early_exit and answerability["answerable"] and answerability["confidence"] > 0.75:
                logger.info("✅ Early exit after Query Decomposition")
                final_results = rerank_final_results(fused_results, query, k)

                return {
                    "query": query,
                    "method": "Orchestrated RAG (Decomposition Exit)",
                    "techniques_used": techniques_used,
                    "retrieval_calls": retrieval_call_count,
                    "execution_trace": execution_trace,
                    "signals": signals,
                    "answerability": answerability,
                    "results": final_results[:k],
                    "context_preview": context[:500],
                    "sub_questions": sub_questions
                }

    # ========== PHASE 3: ITERATIVE REFINEMENT ==========

    rounds = 0
    while (not answerability["answerable"] or answerability["confidence"] < 0.7) and rounds < max_rounds and retrieval_call_count < max_retrieval_calls:
        logger.info(f"🔁 Iterative round {rounds + 1}...")

        # Generate targeted followup
        followup = propose_followup_query(
            query,
            context,
            answerability.get("missing_fields", []),
            model
        )

        if not followup:
            logger.info("No followup query generated, stopping iteration")
            break

        # Retrieve with followup
        followup_results = retrieve_hybrid_baseline(
            followup, backend, k=30, filters=filters)
        retrieval_call_count += 2
        all_results_pool.extend(followup_results[:k*2])

        techniques_used.append(f"iterative_round_{rounds + 1}")

        execution_trace.append({
            "phase": "3_iterative",
            "action": f"Iterative round {rounds + 1}",
            "followup_query": followup,
            "results_added": len(followup_results)
        })

        # Re-check
        fused_results = deduplicate_and_fuse(all_results_pool)
        context = build_context(fused_results[:k], max_tokens=1800)
        answerability = check_answerability(query, context, model)

        rounds += 1

        if answerability["answerable"] and answerability["confidence"] > 0.75:
            logger.info(f"✅ Sufficient information after round {rounds}")
            break

    # ========== FINAL: RERANK AND RETURN ==========

    final_results = deduplicate_and_fuse(all_results_pool)
    final_results = rerank_final_results(final_results, query, k)

    # Generate final answer if answerable
    final_answer = None
    if answerability["answerable"] and ollama:
        try:
            answer_prompt = f"""Responde la siguiente pregunta usando SOLO la información proporcionada.

Pregunta: {query}

Contexto:
{context}

Instrucciones:
- Respuesta clara y completa
- Usa información específica de los fragmentos
- Si falta algo, indícalo brevemente

Respuesta:
"""

            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            client = ollama.Client(host=ollama_host)

            response = client.generate(
                model=model,
                prompt=answer_prompt,
                options={
                    "temperature": 0.5,
                    "num_predict": 400
                }
            )

            final_answer = response.get('response', '').strip()

        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")

    return {
        "query": query,
        "method": "Orchestrated RAG (Full Pipeline)",
        "techniques_used": techniques_used,
        "retrieval_calls": retrieval_call_count,
        "max_calls_budget": max_retrieval_calls,
        "execution_trace": execution_trace,
        "signals": signals,
        "answerability": answerability,
        "results": final_results[:k],
        "total_documents_retrieved": len(all_results_pool),
        "unique_documents": len(set(f"{r.get('path', '')}:{r.get('chunk_id', '')}" for r in final_results)),
        "final_answer": final_answer,
        "context_preview": context[:500],
        "performance": {
            "retrieval_calls": retrieval_call_count,
            "techniques_count": len(techniques_used),
            "iterative_rounds": rounds
        }
    }


def deduplicate_and_fuse(results: List[Dict]) -> List[Dict]:
    """Deduplicate by doc key and use RRF to re-score."""
    # Group by document key
    doc_map = {}
    for r in results:
        key = f"{r.get('path', '')}:{r.get('chunk_id', r.get('page', ''))}"
        if key not in doc_map or r.get('score', 0) > doc_map[key].get('score', 0):
            doc_map[key] = r

    unique_results = list(doc_map.values())

    # Sort by score (already has RRF or base scores)
    unique_results.sort(key=lambda x: x.get(
        'rrf_score', x.get('score', 0)), reverse=True)

    return unique_results


def rerank_final_results(results: List[Dict], query: str, k: int) -> List[Dict]:
    """Apply final MMR reranking for diversity."""
    if not mmr or len(results) <= k:
        return results[:k]

    candidates = [
        {
            'content': r.get('content', ''),
            'sim': r.get('rrf_score', r.get('score', 0.5)),
            'path': r.get('path', ''),
            'chunk_id': r.get('chunk_id', ''),
            'page': r.get('page'),
            'metadata': r.get('metadata', {})
        }
        for r in results
    ]

    reranked = mmr(query, candidates, lambda_=0.7, top_k=k)
    return reranked
