from app.pgvector_backend import search_pgvector
from app.qdrant_backend import search_qdrant
from typing import List
from pathlib import Path
import logging
import os
import re
import time

# Initialize logger
logger = logging.getLogger(__name__)

# Import clean pipeline embedder and query utilities (consolidated)
try:
    from scripts.query_embed import embed_query as embed_query_clean, expand_query, build_queries, embed_e5
except ImportError:
    print("Warning: query_embed module not available")
    print("Install: pip install sentence-transformers")
    embed_e5 = None
    def expand_query(x): return x
    def build_queries(x, **_): return [x]
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


LAW_SYSTEM_PROMPT = (
    "You are an expert in Canadian constitutional law, specializing in the division of powers, "
    "federalism, and the constitutional validity of legislation. "
    "The user will ask a question and you will be given relevant excerpts from a legal reference document.\n\n"
    "Rules:\n"
    "- Start directly with your answer — no preamble.\n"
    "- FOR YES/NO QUESTIONS: state Oui/Non (or Yes/No) as your very first word, then justify using "
    "the single most explicit excerpt. If 'Key propositions' are provided, anchor there first.\n"
    "- CRITICAL DISTINCTION — 'limits on cooperative federalism' does NOT mean 'no overlap exists'. "
    "It means overlap can exist but the constitutional division of powers is not erased by it. "
    "Answer 'yes, overlap can exist' if the excerpts say so, even if they also mention limits.\n"
    "- HIGHEST-EVIDENCE RULE: if an excerpt contains an explicit tabular answer or arrow-format proposition "
    "(e.g. 'double aspect → Oui', 'POBG super-exclusif → Non'), treat that as your primary anchor "
    "and do not contradict it with more general discussion.\n"
    "- SELF-CHECK before writing: does your answer address the EXACT case, section, or doctrine named in the question? "
    "If not, correct your focus.\n"
    "- Do NOT conflate different doctrines: validity (pith and substance) ≠ double aspect ≠ paramountcy (operability) "
    "≠ interjurisdictional immunity. Name the correct doctrine for each point.\n"
    "- Do NOT borrow legal facts or outcomes from a different case unless the excerpts explicitly connect them.\n"
    "- Use specific case names, section numbers (§), article references, and legal tests from the excerpts.\n"
    "- Structure legal analysis: (1) identify the heads of power at issue, (2) apply the relevant test, (3) state the outcome.\n"
    "- If the excerpts are insufficient to fully answer: state what IS supported, explicitly name what is missing, "
    "and suggest which section/doctrine would cover it.\n"
    "- DO NOT list sources or references — the UI shows them separately.\n"
    "- Respond in the same language as the question (French, English, or Spanish)."
)

# French / English legal terms that trigger law-mode prompt
_LAW_KEYWORDS = {
    "tarification", "carbone", "ltpges", "carbon", "pogg", "pobg",
    "fédéralisme", "federalism", "renvoi", "reference",
    "double aspect", "chevauchement", "opérabilité", "operability",
    "paramountcy", "prépondérance", "applicabilité", "immunité",
    "interjurisdictional", "validité", "validity",
    "pith and substance", "caractère véritable",
    "compétence", "compétences", "provincial", "fédéral", "federal",
    "constitution", "constitutionnel", "constitutional",
    "article 91", "article 92", "art. 91", "art. 92",
    "loi sur", "loi de", "charte", "charter",
}

_BIBLE_MARKERS = {"bible", "consti"}


def detect_law_mode(query: str, hits: list) -> bool:
    """Return True when the query or retrieved docs indicate a legal constitutional question."""
    ql = query.lower()
    # Check query keywords
    if any(kw in ql for kw in _LAW_KEYWORDS):
        return True
    # Check if any retrieved doc comes from the Bible/constitution corpus
    for hit in hits:
        path = (hit.get("path", "") or hit.get("document", "") or "").lower()
        if any(m in path for m in _BIBLE_MARKERS):
            return True
    return False


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

# Law-mode template: surfaces decisive propositional anchors before full excerpts
LAW_RAG_TEMPLATE = """{system_prompt}

Question: {q}

KEY PROPOSITIONS extracted from the excerpts (highest-evidence anchors — consult these first):
{key_props}

Full relevant excerpts:
{sources}

Answer (direct, no source list, same language as the question):
"""


# ---------------------------------------------------------------------------
# Contradiction detection + symbolic boost helpers
# ---------------------------------------------------------------------------

# Signals that the model concluded "no overlap"
_NO_OVERLAP_SIGNALS = [
    "pas de chevauchement", "aucun chevauchement", "il n'y a pas de chevauchement",
    "no overlap", "no chevauchement", "ne se chevauchent pas",
    "there is no overlap", "pas d'overlap",
]

# Signals in the retrieved text that actually support overlap
_OVERLAP_EVIDENCE = [
    "double aspect", "situations de fait identiques",
    "peut s'appliquer", "pas super-exclusif", "non super-exclusif",
    "ne supprime pas", "coexister", "fédéralisme coopératif favorise",
    "chevauchement", "souplesse et chevauchement",
]


def likely_contradiction(answer: str, candidates: list) -> bool:
    """Return True when the answer denies overlap but the retrieved chunks support it."""
    a = answer.lower()
    if not any(sig in a for sig in _NO_OVERLAP_SIGNALS):
        return False
    r = " ".join((c.get('content', '') or '') for c in candidates).lower()
    has_double_aspect = "double aspect" in r
    has_explicit_signal = any(sig in r for sig in _OVERLAP_EVIDENCE)
    return has_double_aspect and has_explicit_signal


# Case → section/content markers for symbolic boost
_CASE_BOOST_TRIGGERS: list = [
    (["tarification", "carbone", "ltpges", "carbon"],
     ["tarification", "S06", "S33", "S34", "ltpges"]),
    (["sécession", "secession"], ["S03", "sécession"]),
    (["trans mountain", "burnaby"], ["trans mountain", "burnaby"]),
    (["sénat", "senate"], ["sénat", "senate"]),
]

# Doctrine → content markers for symbolic boost
_DOCTRINE_BOOST_TRIGGERS: list = [
    (["double aspect", "chevauchement"], ["double aspect"]),
    (["prépondérance", "paramountcy", "opérabilité",
     "operability"], ["prépondérance", "inopérant"]),
    (["immunité interjuridictionnelle", "interjurisdictional"],
     ["immunité interjuridictionnelle"]),
    (["pogg", "pobg", "intérêt national"], ["pobg", "pogg", "intérêt national"]),
    (["pith and substance", "caractère véritable", "validité"],
     ["caractère véritable", "pith and substance"]),
]


def _symbolic_boost(query: str, candidate: dict) -> float:
    """Return a score boost [0.0, 0.35] for same-case / same-doctrine alignment."""
    ql = query.lower()
    text = (candidate.get('content', '') or '').lower()
    sid = (candidate.get('section_id', '') or
           candidate.get('metadata', {}).get('section_id', '') or '').lower()
    boost = 0.0
    for triggers, markers in _CASE_BOOST_TRIGGERS:
        if any(t in ql for t in triggers):
            if any(m.lower() in text or m.lower() in sid for m in markers):
                boost += 0.20
                break
    for triggers, markers in _DOCTRINE_BOOST_TRIGGERS:
        if any(t in ql for t in triggers):
            if any(m.lower() in text for m in markers):
                boost += 0.15
                break
    return min(boost, 0.35)


# Correction prompt fired when contradiction is detected
CORRECTION_PROMPT = """{system_prompt}

IMPORTANT — Your previous draft may conflict with the retrieved excerpts.
Check: the excerpts explicitly support overlap via double-aspect doctrine,
even if they also note that no *formal* concurrent jurisdiction exists.
"No formal concurrent jurisdiction" ≠ "no overlap".

Re-read the KEY PROPOSITIONS and excerpts below, then provide a corrected answer.

Question: {q}

KEY PROPOSITIONS (decisive anchors):
{key_props}

Full relevant excerpts:
{sources}

Corrected answer (Oui/Non first, then one sentence of justification using the most explicit excerpt):
"""

# Regex compiled once at module load
_ARROW_PROP_RE = re.compile(
    r'(→\s*(Oui|Non|Yes|No|Vrai|Faux|True|False)\b'           # arrow: → Oui
    # table cell: | **Oui**
    r'|\|\s*\*\*(Oui|Non|Yes|No)\*\*'
    # table cell: | Oui |
    r'|\|\s*(Oui|Non|Yes|No)\s*\|)',
    re.IGNORECASE,
)
_NOTE_PROP_RE = re.compile(r'\b(NOTA|NOTE|NB)\s*[:：]', re.IGNORECASE)


def _extract_decisive_props(hits: list) -> str:
    """
    Scan retrieved chunks for explicit propositional anchors:
    - Arrow-format propositions: "double aspect → Oui", "POBG super-exclusif → Non"
    - NOTA/NOTE/NB lines with key doctrinal statements
    Returns a compact bullet list capped at 8 items (empty string if none found).
    """
    props: list = []
    seen: set = set()
    for hit in hits:
        content = hit.get('content', '') or ''
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _ARROW_PROP_RE.search(stripped) or _NOTE_PROP_RE.search(stripped):
                key = stripped[:80]
                if key not in seen:
                    seen.add(key)
                    props.append('• ' + stripped)
    return '\n'.join(props[:8])


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

    # Build focused retrieval variants (1-4 queries, doctrine-isolated)
    query_variants = build_queries(query, max_variants=4)

    def _embed(q: str):
        if embed_query_clean is not None:
            try:
                return embed_query_clean(q)
            except Exception as e:
                logger.warning(f"Clean embedder failed: {e}")
        return embed_e5([q], is_query=True)[0]

    # Run each variant and collect all hits, scored by RRF
    from collections import defaultdict
    rrf_scores: dict = defaultdict(float)
    rrf_docs: dict = {}
    RRF_K = 60

    for rank_offset, variant in enumerate(query_variants):
        emb = _embed(variant)
        variant_hits = BACKENDS[backend](
            emb, k=k, where=filters, collection_suffix=collection_suffix)
        for rank, hit in enumerate(variant_hits, start=1):
            doc_key = (
                hit.get('path', hit.get('document', '')),
                hit.get('chunk_id', hit.get('page', '')),
            )
            rrf_scores[doc_key] += 1.0 / (RRF_K + rank)
            if doc_key not in rrf_docs or hit['score'] > rrf_docs[doc_key]['score']:
                rrf_docs[doc_key] = hit

    # Sort fused results and take top-k
    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    hits = [rrf_docs[dk] for dk in ranked_keys[:k]]

    # The primary expanded query (first variant that differs from original)
    expanded_query = query_variants[1] if len(query_variants) > 1 else query

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
            "section_id": hit.get('section_id', metadata.get('section_id', '')),
            "section_title": hit.get('section_title', metadata.get('section_title', '')),
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

    # Build focused retrieval variants (1-4 queries, doctrine-isolated)
    query_variants = build_queries(query, max_variants=4)
    expanded_query = query_variants[1] if len(query_variants) > 1 else query

    def _embed_llm(q: str):
        if embed_query_clean is not None:
            try:
                return embed_query_clean(q)
            except Exception as e:
                logger.warning(f"Clean embedder failed: {e}")
        return embed_e5([q], is_query=True)[0]

    # Run each variant and fuse with RRF, using wider k for reranking
    from collections import defaultdict
    rrf_scores_llm: dict = defaultdict(float)
    rrf_docs_llm: dict = {}
    RRF_K = 60
    fetch_k = max(k * 3, 12)

    for rank_offset, variant in enumerate(query_variants):
        emb = _embed_llm(variant)
        variant_hits = BACKENDS[backend](
            emb, k=fetch_k, where=filters, collection_suffix=collection_suffix)
        for rank, hit in enumerate(variant_hits, start=1):
            doc_key = (
                hit.get('path', hit.get('document', '')),
                hit.get('chunk_id', hit.get('page', '')),
            )
            rrf_scores_llm[doc_key] += 1.0 / (RRF_K + rank)
            if doc_key not in rrf_docs_llm or hit['score'] > rrf_docs_llm[doc_key]['score']:
                rrf_docs_llm[doc_key] = hit

    ranked_keys = sorted(
        rrf_scores_llm, key=lambda k: rrf_scores_llm[k], reverse=True)
    hits = [rrf_docs_llm[dk] for dk in ranked_keys[:fetch_k]]

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
            'document': hit.get('document', ''),
            'chunk_id': _cid,
            'page': hit.get('page'),
            'section_id': hit.get('section_id', ''),
            'section_title': hit.get('section_title', ''),
            'metadata': hit.get('metadata', {})
        })

    # Apply symbolic score boost (same-case / same-doctrine alignment) before MMR
    if detect_law_mode(query, candidates):
        for c in candidates:
            boost = _symbolic_boost(query, c)
            if boost > 0:
                c['sim'] = min(c['sim'] + boost, 1.0)
                logger.debug(
                    f"Symbolic boost +{boost:.2f} → {c.get('section_id', c.get('chunk_id', ''))}")

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
            "section_id": item.get('section_id', metadata.get('section_id', '')),
            "section_title": item.get('section_title', metadata.get('section_title', '')),
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
            "preview": content[:120] + "..." if len(content) > 120 else content,
            "path": item.get('path', ''),
            "content": content,
            "chunk_id": item.get('chunk_id'),
        })

    # Build context for LLM using enhanced method if available
    if build_context_enhanced:
        sources = build_context_enhanced(reranked, max_tokens=3000)
    else:
        sources = build_context(reranked)

    # Choose system prompt and template: law-mode adds decisive-prop anchoring
    if detect_law_mode(query, reranked):
        active_system_prompt = LAW_SYSTEM_PROMPT
        key_props = _extract_decisive_props(reranked)
        if key_props:
            prompt = LAW_RAG_TEMPLATE.format(
                system_prompt=active_system_prompt, q=query,
                key_props=key_props, sources=sources)
        else:
            prompt = RAG_TEMPLATE.format(
                system_prompt=active_system_prompt, q=query, sources=sources)
    else:
        active_system_prompt = load_system_prompt()
        prompt = RAG_TEMPLATE.format(
            system_prompt=active_system_prompt, q=query, sources=sources)

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
        # Contradiction check: if law mode and answer denies overlap but chunks support it,
        # retry once with a focused correction prompt at lower temperature.
        if enhanced_response and detect_law_mode(query, reranked):
            if likely_contradiction(enhanced_response, reranked):
                logger.info(
                    "⚠️ Contradiction detected — retrying with correction prompt")
                _key_props_corr = _extract_decisive_props(reranked)
                correction_prompt = CORRECTION_PROMPT.format(
                    system_prompt=LAW_SYSTEM_PROMPT,
                    q=query,
                    key_props=_key_props_corr or "(see excerpts below)",
                    sources=sources,
                )
                try:
                    corr_opts = {"repeat_penalty": 1.1, "temperature": 0.2,
                                 "num_predict": 1024, "top_p": 0.85}  # think=False passed as kwarg below
                    if ollama_generate_with_retry:
                        corr_resp = ollama_generate_with_retry(
                            model=model, prompt=correction_prompt,
                            max_retries=2, auto_fallback=False, auto_restart=False,
                            think=False, options=corr_opts,
                        )
                    else:
                        _client = ollama.Client(
                            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
                        corr_resp = _client.generate(
                            model=model, prompt=correction_prompt, think=False, options=corr_opts)
                    corr_raw = corr_resp.get('response', '') if hasattr(
                        corr_resp, 'get') else getattr(corr_resp, 'response', '')
                    corr_raw = _re.sub(
                        r'<think>[\s\S]*?</think>\s*', '', corr_raw).strip()
                    if corr_raw:
                        enhanced_response = corr_raw
                        logger.info("✅ Contradiction corrected")
                except Exception as _ce:
                    logger.warning(f"Correction retry failed: {_ce}")
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
