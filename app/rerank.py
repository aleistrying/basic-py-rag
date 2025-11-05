"""
Reranking utilities for improving retrieval quality
"""
from rapidfuzz import fuzz
from typing import List, Dict
import re


def keyword_score(query: str, text: str) -> float:
    """
    Calculate keyword overlap score between query and text.
    Used for lightweight lexical boost after vector search.

    Args:
        query: Search query
        text: Document text

    Returns:
        Score between 0 and 1 based on term overlap
    """
    if not query or not text:
        return 0.0

    # Extract terms (words)
    q_terms = set(re.findall(r'\w+', str(query).lower()))
    t_terms = set(re.findall(r'\w+', str(text).lower()))

    if not q_terms:
        return 0.0

    # Calculate Jaccard similarity
    overlap = len(q_terms & t_terms)
    return overlap / len(q_terms)


def mmr(query: str, candidates: List[Dict], lambda_: float = 0.7, top_k: int = 5) -> List[Dict]:
    """
    Maximal Marginal Relevance (MMR) with lexical boost

    Args:
        query: The search query
        candidates: List of dicts with 'content' and 'sim' (0..1)
        lambda_: Balance between relevance and diversity (0.7 = more relevance)
        top_k: Number of results to return

    Returns:
        Reranked list of candidates
    """
    if not candidates or not query:
        return []

    selected, rest = [], candidates[:]
    q = str(query).lower()

    while rest and len(selected) < top_k:
        best, best_score = None, -1
        for c in rest:
            # Calculate diversity penalty
            diversity = 0 if not selected else max(
                fuzz.partial_ratio(c['content'], s['content'])/100 for s in selected
            )
            # MMR score: balance relevance vs diversity
            score = lambda_ * c['sim'] - (1 - lambda_) * diversity
            if score > best_score:
                best, best_score = c, score

        if best:
            selected.append(best)
            rest.remove(best)

    # Lexical boost for exact token matches
    for s in selected:
        content = s.get('content', '')
        if content:
            # Use our keyword_score function
            lexical_boost = keyword_score(q, str(content))
            s['sim'] += 0.1 * lexical_boost  # 10% boost for keyword matches
            s['sim'] = min(s['sim'], 1.0)  # Cap at 1.0

    # Re-sort by updated similarity
    return sorted(selected, key=lambda x: x['sim'], reverse=True)


def build_context(hits: List[Dict], max_tokens: int = 1500) -> str:
    """
    Build context from search hits, grouping by document and limiting tokens

    Args:
        hits: List of search results with 'path', 'content', 'chunk_id' fields
        max_tokens: Maximum total tokens to include

    Returns:
        Formatted context string
    """
    lines = []
    current_tokens = 0
    doc_chunks = {}  # Group by document

    # Group chunks by document (max 2 per doc to avoid flooding)
    for h in hits:
        doc_path = h.get('path', 'unknown')
        if doc_path not in doc_chunks:
            doc_chunks[doc_path] = []
        if len(doc_chunks[doc_path]) < 2:  # Max 2 chunks per document
            doc_chunks[doc_path].append(h)

    # Build context
    for doc_path, chunks in doc_chunks.items():
        for h in chunks:
            # Trim long content
            content = h.get('content', '') or ''  # Handle None content
            if not content:
                continue
            snippet = (content[:700] + "…") if len(content) > 700 else content

            # Estimate tokens (rough approximation)
            tokens = len(snippet.split())
            if current_tokens + tokens > max_tokens:
                break

            chunk_id = h.get('chunk_id', h.get('page', 'unknown'))
            doc_name = doc_path.split('/')[-1] if '/' in doc_path else doc_path
            lines.append(f"- ({doc_name}:{chunk_id}) {snippet}")
            current_tokens += tokens

    return "\n".join(lines)
