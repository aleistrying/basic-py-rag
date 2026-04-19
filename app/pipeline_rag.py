"""
Staged RAG pipeline — combines multi-query, doctrine-aware retrieval,
gap analysis, iterative second pass, structured evidence extraction,
and contradiction-safe final generation.

Stages:
  A  Analyze question     → QueryAnalysis
  B  Build query buckets  → {case, doctrine, hyde, decompose}
  C  Retrieve + RRF fuse  → ranked candidate chunks
  D  Symbolic rerank      → score boosts for same-case / same-doctrine
  E  Gap analysis         → {have_case, have_doctrine, have_signal}, confidence
  F  Iterative 2nd pass   → only when confidence < 0.70
  G  Evidence extraction  → EvidenceSheet (structured JSON from small model)
  H  Final answer         → from evidence sheet + excerpts
  I  Contradiction check  → retry at temp=0.1 if needed
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── embedder ──────────────────────────────────────────────────────────────────
try:
    from scripts.query_embed import embed_query as embed_query_clean, embed_e5
except ImportError:
    embed_query_clean = None
    embed_e5 = None

# ── LLM ───────────────────────────────────────────────────────────────────────
try:
    import ollama as _ollama_module
except ImportError:
    _ollama_module = None

try:
    from app.ollama_utils import ollama_generate_with_retry
except ImportError:
    ollama_generate_with_retry = None

# ── retrieval backends ────────────────────────────────────────────────────────
from app.qdrant_backend import search_qdrant
from app.pgvector_backend import search_pgvector

BACKENDS: dict = {"qdrant": search_qdrant, "pgvector": search_pgvector}

# ── reranker + context builder ────────────────────────────────────────────────
from app.rerank import mmr as _mmr_fn, build_context

# ── existing helpers from rag.py ──────────────────────────────────────────────
from app.rag import (
    LAW_SYSTEM_PROMPT,
    LAW_RAG_TEMPLATE,
    RAG_TEMPLATE,
    CORRECTION_PROMPT,
    _extract_decisive_props,
    _symbolic_boost,
    detect_law_mode,
    likely_contradiction,
)

# ─────────────────────────────────────────────────────────────────────────────
# A.  Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryAnalysis:
    language: str                               # "fr" | "en" | "es"
    answer_type: str                            # "yes_no" | "analytical" | "definitional"
    case_name: Optional[str] = None
    statute: Optional[str] = None
    primary_doctrine: Optional[str] = None
    secondary_doctrines: List[str] = field(default_factory=list)


@dataclass
class EvidenceSheet:
    answer_polarity: Optional[str]              # "yes" | "no" | "partly" | "insufficient"
    controlling_doctrine: Optional[str]
    case_support: List[str]
    doctrine_support: List[str]
    warnings: List[str]
    missing: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# A.  Question analyzer
# ─────────────────────────────────────────────────────────────────────────────

# (trigger_words, canonical_case_name, primary_statute)
_CASE_PATTERNS: list = [
    (["tarification", "carbone", "ltpges"], "Renvoi tarification carbone", "LTPGES"),
    (["sécession", "secession", "québec secession"], "Renvoi relatif à la sécession", None),
    (["trans mountain", "burnaby"], "Trans Mountain / Burnaby", "EMA"),
    (["réforme du sénat", "senate reform", "sénat"],
     "Renvoi relatif à la réforme du Sénat", None),
    (["valeurs mobilières", "securities regulation"],
     "Renvoi relatif aux valeurs mobilières", None),
]

# (trigger_words, canonical_doctrine_name)
_DOCTRINE_PATTERNS: list = [
    (["chevauchement", "overlap", "double aspect", "double-aspect",
       "situations de fait identiques"], "double aspect"),
    (["prépondérance", "paramountcy", "opérabilité", "operability",
       "inopérant", "inoperative"], "paramountcy"),
    (["immunité interjuridictionnelle", "interjurisdictional immunity",
       "applicabilité", "iji"], "interjurisdictional immunity"),
    (["pogg", "pobg", "intérêt national", "national concern",
       "résiduel fédéral"], "POGG"),
    (["pith and substance", "caractère véritable", "validité",
       "validity", "qualification"], "pith and substance"),
    (["droit criminel", "criminal law", "92(27)", "91(27)"],
     "criminal law power"),
]

# Yes-or-No question starters (French + English)
_YES_NO_STARTS = (
    "est-ce", "y a-t-il", "est-il", "est-elle", "peut-on",
    "s'applique", "existe-t-il", "avez-vous", "avait",
    "is ", "does ", "did ", "can ", "was ", "were ", "has ", "have ",
    "is there", "do the", "does the",
)


def analyze_question(question: str) -> QueryAnalysis:
    """Stage A — extract language, answer type, case, statute, doctrine."""
    q = question.lower()

    # Language
    fr_markers = ["renvoi", "chevauchement", "fédér", "provinc",
                  "compétence", "tarification", "validité"]
    language = "fr" if any(w in q for w in fr_markers) else "en"

    # Answer type
    answer_type = "yes_no" if any(
        q.strip().startswith(s) or s in q[:60] for s in _YES_NO_STARTS
    ) else "analytical"

    # Case + statute
    case_name = statute = None
    for triggers, name, stat in _CASE_PATTERNS:
        if any(t in q for t in triggers):
            case_name = name
            statute = stat
            break

    # Doctrines (first match → primary, rest → secondary)
    primary_doctrine = None
    secondary: List[str] = []
    for triggers, doctrine in _DOCTRINE_PATTERNS:
        if any(t in q for t in triggers):
            if primary_doctrine is None:
                primary_doctrine = doctrine
            elif doctrine not in secondary:
                secondary.append(doctrine)

    # Carbon pricing → always carries a POGG angle secondarily
    if case_name == "Renvoi tarification carbone" and "POGG" not in secondary:
        if primary_doctrine != "POGG":
            secondary.append("POGG")

    return QueryAnalysis(
        language=language,
        answer_type=answer_type,
        case_name=case_name,
        statute=statute,
        primary_doctrine=primary_doctrine,
        secondary_doctrines=secondary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# B.  Query bucket builder
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_queries(question: str, qa: QueryAnalysis) -> Dict[str, List[str]]:
    """
    Stage B — Return four named buckets of retrieval queries.
    Buckets are retrieved separately so case / doctrine / semantic signals
    don't drown each other out.
    """
    buckets: Dict[str, List[str]] = {
        "case": [],
        "doctrine": [],
        "hyde": [],
        "decompose": [],
    }

    # ── Case queries ──
    if qa.case_name == "Renvoi tarification carbone":
        buckets["case"] += [
            "renvoi tarification carbone fédéralisme",
            "LTPGES renvoi 2021 constitutionnel",
            "renvoi tarification carbone POBG intérêt national",
        ]
    elif qa.case_name:
        buckets["case"].append(qa.case_name.lower())

    # Always include original question in case bucket as high-weight fallback
    if question not in buckets["case"]:
        buckets["case"].insert(0, question)

    # ── Doctrine queries ──
    if qa.primary_doctrine == "double aspect":
        buckets["doctrine"] += [
            "double aspect renvoi tarification carbone",
            "chevauchement pouvoirs fédéraux provinciaux tarification",
            "situations de fait identiques perspectives différentes",
        ]
    elif qa.primary_doctrine == "paramountcy":
        buckets["doctrine"] += [
            "prépondérance fédérale inopérant loi provinciale",
            "double aspect prépondérance tarification carbone",
        ]
    elif qa.primary_doctrine == "POGG":
        buckets["doctrine"] += [
            "POBG intérêt national tarification carbone",
            "POBG super-exclusif tarification carbone double aspect",
        ]
    elif qa.primary_doctrine == "interjurisdictional immunity":
        buckets["doctrine"] += [
            "immunité interjuridictionnelle noyau fédéral compétence",
            "applicabilité éléments vitaux entreprise fédérale",
        ]
    elif qa.primary_doctrine == "pith and substance":
        buckets["doctrine"] += [
            "caractère véritable pith and substance tarification carbone",
            "qualification constitutionnelle double aspect",
        ]

    for sec in qa.secondary_doctrines:
        if sec == "POGG":
            buckets["doctrine"].append("POBG super-exclusif tarification carbone")
        elif sec == "double aspect":
            buckets["doctrine"].append("double aspect situations de fait identiques")

    # ── HyDE query ──
    # A synthetic "ideal passage" description that will embed closer to the answer
    if qa.answer_type == "yes_no" and qa.case_name:
        if qa.language == "fr":
            buckets["hyde"].append(
                f"La Cour suprême reconnaît que les pouvoirs fédéraux et provinciaux "
                f"peuvent se chevaucher dans le cadre du {qa.case_name}, "
                f"notamment par la théorie du double aspect."
            )
        else:
            buckets["hyde"].append(
                f"The Supreme Court recognized that federal and provincial powers can "
                f"overlap in {qa.case_name} through the double aspect doctrine."
            )
    elif qa.answer_type == "yes_no":
        # Fallback HyDE without specific case
        buckets["hyde"].append(
            "Oui, il peut y avoir chevauchement entre pouvoirs fédéraux et provinciaux "
            "selon la théorie du double aspect."
        )

    # ── Decompose queries ──
    if qa.case_name:
        buckets["decompose"] += [
            f"Quel est le résultat du {qa.case_name} sur la compétence fédérale?",
            f"Quelle doctrine explique le chevauchement dans le {qa.case_name}?",
        ]
    if qa.primary_doctrine:
        buckets["decompose"].append(
            f"{qa.primary_doctrine} {qa.case_name or ''}".strip()
        )

    return buckets


# ─────────────────────────────────────────────────────────────────────────────
# C + D.  Multi-bucket retrieval + RRF fusion with symbolic boost
# ─────────────────────────────────────────────────────────────────────────────

# Relative importance of each query bucket
_BUCKET_WEIGHTS: Dict[str, float] = {
    "case":      1.4,
    "doctrine":  1.2,
    "decompose": 1.0,
    "hyde":      0.9,
}

# Original question weight (inserted first in all retrievals)
_ORIGIN_WEIGHT = 1.5


def _embed_query(q: str):
    if embed_query_clean is not None:
        try:
            return embed_query_clean(q)
        except Exception:
            pass
    if embed_e5 is not None:
        return embed_e5([q], is_query=True)[0]
    raise RuntimeError("No embedder available — install sentence-transformers")


def retrieve_and_fuse(
    query_buckets: Dict[str, List[str]],
    original_query: str,
    backend: str,
    fetch_k: int,
    filters: dict,
    collection_suffix: Optional[str],
    qa: QueryAnalysis,
) -> List[dict]:
    """
    Stage C+D — retrieve from each bucket separately, fuse via weighted RRF,
    apply symbolic boost inside the scoring.
    """
    RRF_K = 60
    rrf_scores: dict = defaultdict(float)
    rrf_docs: dict = {}

    _backend_fn = BACKENDS.get(backend, search_qdrant)
    is_law = detect_law_mode(original_query, [])  # lightweight check on query alone

    # Build flat list: (query_text, rrf_weight)
    weighted_queries: List[tuple] = [(original_query, _ORIGIN_WEIGHT)]
    for bucket, queries in query_buckets.items():
        w = _BUCKET_WEIGHTS.get(bucket, 1.0)
        for qt in queries:
            if qt != original_query:           # avoid duplicate embedding
                weighted_queries.append((qt, w))

    for query_text, weight in weighted_queries:
        try:
            emb = _embed_query(query_text)
        except Exception as exc:
            logger.warning(f"Embed failed for '{query_text[:50]}': {exc}")
            continue

        try:
            hits = _backend_fn(
                emb, k=fetch_k,
                where=filters if filters else None,
                collection_suffix=collection_suffix,
            )
        except Exception as exc:
            logger.warning(f"Retrieval failed for '{query_text[:50]}': {exc}")
            continue

        for rank, hit in enumerate(hits, start=1):
            doc_key = (
                hit.get('path', hit.get('document', '')),
                hit.get('chunk_id', hit.get('page', '')),
            )
            # Symbolic boost folded directly into the RRF score
            sym = _symbolic_boost(original_query, hit) if is_law else 0.0
            rrf_scores[doc_key] += weight * (1.0 + sym) / (RRF_K + rank)

            # Keep the highest-score hit for each key
            if doc_key not in rrf_docs or hit.get('score', 0) > rrf_docs[doc_key].get('score', 0):
                rrf_docs[doc_key] = hit

    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [rrf_docs[dk] for dk in ranked_keys[:fetch_k]]


# ─────────────────────────────────────────────────────────────────────────────
# E.  Gap analysis + confidence
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that constitute an explicit yes/no overlap signal
_EXPLICIT_OVERLAP_SIGNALS = [
    "chevauchement",
    "peut s'appliquer",
    "pas super-exclusif",
    "non super-exclusif",
    "situations de fait identiques",
    "double aspect",
    "coexister",
    "souplesse et chevauchement",
    "fédéralisme coopératif",
]


def gap_analysis(chunks: List[dict], qa: QueryAnalysis) -> Dict[str, bool]:
    """Stage E — check which evidence types are present in the top chunks."""
    full_text = " ".join(
        (c.get('content', '') or '') for c in chunks[:8]
    ).lower()

    # Case support: at least half the case-name tokens present
    have_case = True
    if qa.case_name:
        tokens = [t for t in qa.case_name.lower().split() if len(t) > 3]
        have_case = (
            sum(1 for t in tokens if t in full_text) >= max(1, len(tokens) // 2)
        )

    # Doctrine support
    have_doctrine = True
    if qa.primary_doctrine:
        have_doctrine = qa.primary_doctrine.lower() in full_text

    # Explicit overlap / yes-no signal
    have_signal = any(s in full_text for s in _EXPLICIT_OVERLAP_SIGNALS)

    return {
        "have_case":     have_case,
        "have_doctrine": have_doctrine,
        "have_signal":   have_signal,
    }


def compute_confidence(gaps: Dict[str, bool], top_chunks: List[dict]) -> float:
    score = 0.0
    if gaps["have_case"]:     score += 0.40
    if gaps["have_doctrine"]: score += 0.30
    if gaps["have_signal"]:   score += 0.20
    if top_chunks and top_chunks[0].get('score', 0) > 0.88:
        score += 0.10
    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# F.  Second-pass query builder
# ─────────────────────────────────────────────────────────────────────────────

def build_second_pass_queries(qa: QueryAnalysis, gaps: Dict[str, bool]) -> List[str]:
    """Stage F — targeted queries for the specific gaps that were identified."""
    q: List[str] = []

    if not gaps["have_doctrine"]:
        if qa.primary_doctrine == "double aspect":
            q += [
                "double aspect renvoi tarification carbone",
                "double aspect POBG tarification carbone",
            ]
        elif qa.primary_doctrine == "POGG":
            q += ["POBG super-exclusif tarification carbone intérêt national"]
        elif qa.primary_doctrine == "paramountcy":
            q += ["prépondérance fédérale opérabilité inopérant"]
        elif qa.primary_doctrine == "interjurisdictional immunity":
            q += ["immunité interjuridictionnelle noyau fédéral"]
        elif qa.primary_doctrine:
            q.append(f"{qa.primary_doctrine} {qa.case_name or ''}".strip())

    if not gaps["have_signal"]:
        q += [
            "chevauchement POBG tarification carbone fédéralisme coopératif",
            "pas super-exclusif tarification carbone double aspect souplesse",
        ]

    if not gaps["have_case"] and qa.case_name:
        q.append(f"{qa.case_name} compétence fédérale constitutionnel")

    return q[:4]   # cap at 4 second-pass queries


# ─────────────────────────────────────────────────────────────────────────────
# G.  Structured evidence extraction
# ─────────────────────────────────────────────────────────────────────────────

_EVIDENCE_PROMPT = """\
You are a legal evidence extractor. Read the following excerpts carefully.
Return ONLY a valid JSON object — no other text, no explanation.

Question: {question}

Excerpts:
{sources}

Fill exactly this JSON (use null if not supported by excerpts):
{{
  "answer_polarity": "yes | no | partly | insufficient",
  "controlling_doctrine": "doctrine name or null",
  "case_support": ["brief proposition 1", "brief proposition 2"],
  "doctrine_support": ["brief doctrine proposition"],
  "warnings": ["doctrine confusion to avoid"],
  "missing": ["missing piece of evidence"]
}}
"""


def _extract_evidence_sheet(
    question: str,
    sources: str,
    model: str,
) -> Optional[EvidenceSheet]:
    """
    Stage G — ask the model to fill a structured evidence sheet.
    Returns None if the model call fails or produces unparseable output.
    """
    prompt = _EVIDENCE_PROMPT.format(
        question=question,
        sources=sources[:2500],   # keep prompt short for the extraction call
    )
    opts = {"temperature": 0.05, "num_predict": 512, "top_p": 0.9}

    try:
        if ollama_generate_with_retry:
            resp = ollama_generate_with_retry(
                model=model,
                prompt=prompt,
                max_retries=2,
                auto_fallback=False,
                auto_restart=False,
                options=opts,
            )
        elif _ollama_module:
            client = _ollama_module.Client(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
            resp = client.generate(model=model, prompt=prompt, options=opts)
        else:
            return None

        raw = resp.get('response', '') if hasattr(resp, 'get') else getattr(resp, 'response', '')
        # Strip <think> blocks emitted by reasoning models
        raw = re.sub(r'<think>[\s\S]*?</think>\s*', '', raw).strip()

        # Extract the first JSON object
        m = re.search(r'\{[\s\S]*\}', raw)
        if not m:
            logger.debug("Evidence extraction: no JSON found in response")
            return None

        import json as _json
        data = _json.loads(m.group(0))
        return EvidenceSheet(
            answer_polarity=data.get('answer_polarity'),
            controlling_doctrine=data.get('controlling_doctrine'),
            case_support=data.get('case_support') or [],
            doctrine_support=data.get('doctrine_support') or [],
            warnings=data.get('warnings') or [],
            missing=data.get('missing') or [],
        )
    except Exception as exc:
        logger.warning(f"Evidence extraction failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# H.  Final-answer prompt
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_FINAL_PROMPT = """\
{system_prompt}

STRUCTURED EVIDENCE (extracted from the retrieved excerpts):
  Answer polarity : {answer_polarity}
  Controlling doctrine : {controlling_doctrine}
  Key propositions :
{props}
  ⚠ Doctrine confusions to avoid :
{warnings}

KEY PROPOSITIONAL ANCHORS from excerpts:
{key_props}

Full excerpts:
{sources}

Question: {q}

Answer (for yes/no questions: start with Oui/Non; then 2–5 sentences of justification):
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_search(
    query: str,
    backend: str = "qdrant",
    k: int = 5,
    model: str = "qwen3:4b",
    filters: Optional[dict] = None,
    distance_metric: str = "cosine",
    index_algorithm: str = "hnsw",
    collection_suffix: Optional[str] = None,
    sources_only: bool = False,
) -> dict:
    """
    Full staged evidence-grounded RAG pipeline.
    Returns a dict whose schema is compatible with the standard generate_llm_answer response.

    When ``sources_only=True`` the function stops after stage F (retrieval +
    reranking) and returns just the sources list without running any LLM calls.
    This is used by the fast /api/research/search endpoint so the sources panel
    is populated from the same multi-bucket retrieval as the AI answer, without
    paying the cost of evidence extraction or generation.
    """
    t0 = time.time()
    fetch_k = max(k * 3, 15)

    # ── A: Analyze ────────────────────────────────────────────────────────────
    qa = analyze_question(query)
    logger.info(
        f"[Pipeline] case={qa.case_name!r} doctrine={qa.primary_doctrine!r} "
        f"type={qa.answer_type} lang={qa.language}"
    )

    # ── B: Build query buckets ────────────────────────────────────────────────
    buckets = build_combined_queries(query, qa)
    total_bucket_queries = sum(len(v) for v in buckets.values())
    logger.info(f"[Pipeline] {total_bucket_queries} bucket queries across {len(buckets)} buckets")

    # ── C+D: Retrieve + fuse ──────────────────────────────────────────────────
    fused = retrieve_and_fuse(
        buckets, query, backend, fetch_k, filters if filters else None,
        collection_suffix, qa,
    )

    # Deduplicate by (path, chunk_id)
    seen_keys: set = set()
    deduped: List[dict] = []
    for h in fused:
        dk = f"{h.get('path', '')}::{h.get('chunk_id', h.get('page', ''))}"
        if dk not in seen_keys:
            seen_keys.add(dk)
            deduped.append(h)

    # ── E: Gap analysis + confidence ──────────────────────────────────────────
    gaps = gap_analysis(deduped, qa)
    confidence = compute_confidence(gaps, deduped)
    logger.info(f"[Pipeline] gaps={gaps} confidence={confidence:.2f}")

    # ── F: Iterative second pass (only when confidence < 0.70) ───────────────
    second_pass_queries: List[str] = []
    if confidence < 0.70:
        second_pass_queries = build_second_pass_queries(qa, gaps)
        if second_pass_queries:
            logger.info(f"[Pipeline] second pass: {second_pass_queries}")
            sp_buckets = {
                "case": [],
                "doctrine": second_pass_queries,
                "hyde": [],
                "decompose": [],
            }
            extra = retrieve_and_fuse(
                sp_buckets, query, backend, max(fetch_k // 2, 6),
                filters if filters else None, collection_suffix, qa,
            )
            # Merge new chunks not already present
            for h in extra:
                dk = f"{h.get('path', '')}::{h.get('chunk_id', h.get('page', ''))}"
                if dk not in seen_keys:
                    seen_keys.add(dk)
                    deduped.append(h)
            # Re-check gaps after merge
            gaps = gap_analysis(deduped, qa)
            confidence = compute_confidence(gaps, deduped)
            logger.info(f"[Pipeline] post-2nd-pass confidence={confidence:.2f}")

    # ── Prepare candidates for MMR ────────────────────────────────────────────
    candidates: List[dict] = []
    for hit in deduped[:fetch_k]:
        raw_sim = float(hit.get('score', 0.0))
        raw_sim = max(0.0, min(raw_sim, 1.0))
        meta = hit.get('metadata', {}) or {}
        candidates.append({
            'content':       hit.get('content', ''),
            'sim':           raw_sim,
            'path':          hit.get('path', ''),
            'document':      hit.get('document', ''),
            'chunk_id':      hit.get('chunk_id', hit.get('page', '')),
            'page':          hit.get('page'),
            'section_id':    hit.get('section_id', '') or meta.get('section_id', ''),
            'section_title': hit.get('section_title', '') or meta.get('section_title', ''),
            'metadata':      meta,
        })

    # Apply symbolic boost before MMR
    if detect_law_mode(query, candidates):
        for c in candidates:
            boost = _symbolic_boost(query, c)
            if boost > 0:
                c['sim'] = min(c['sim'] + boost, 1.0)

    # ── MMR reranking ─────────────────────────────────────────────────────────
    try:
        reranked = (
            _mmr_fn(query, candidates, lambda_=0.7, top_k=k)
            if len(candidates) > k else candidates[:k]
        )
    except Exception as exc:
        logger.warning(f"MMR failed: {exc}")
        reranked = candidates[:k]

    # ── Build context strings ─────────────────────────────────────────────────
    sources_text = build_context(reranked, max_tokens=3000)
    key_props = _extract_decisive_props(reranked)

    # ── Early exit for sources-only mode (used by /api/research/search) ──────
    if sources_only:
        results: List[dict] = []
        for item in reranked:
            content = item.get('content', '') or ''
            path_val = item.get('path', '')
            doc_name = path_val.split('/')[-1] if '/' in path_val else (
                path_val or item.get('document', 'Document')
            )
            section_id = item.get('section_id', '')
            section_title = item.get('section_title', '')
            results.append({
                "document":      doc_name,
                "reference":     f"{doc_name} [{section_id}]" if section_id else doc_name,
                "page":          item.get('page'),
                "chunk_id":      item.get('chunk_id'),
                "section_id":    section_id,
                "section_title": section_title,
                "similarity":    f"{item.get('sim', 0):.3f}",
                "preview":       (content[:120] + "…") if len(content) > 120 else content,
                "path":          path_val,
                "content":       content,
                "score":         item.get('sim', 0),
            })
        total_ms = round((time.time() - t0) * 1000, 1)
        return {
            "query":        query,
            "method":       "pipeline",
            "backend":      backend.upper(),
            "search_time_ms": total_ms,
            "total_results": len(results),
            "results":      results,
            "sources":      results,
            "pipeline_info": {
                "gaps":       gaps,
                "confidence": confidence,
            },
        }

    # ── G: Structured evidence extraction ────────────────────────────────────
    # Only do the extra LLM call when law mode is active (adds latency but
    # gives the final prompt a structured anchor to reason from).
    evidence: Optional[EvidenceSheet] = None
    if detect_law_mode(query, reranked):
        evidence = _extract_evidence_sheet(query, sources_text, model)
        if evidence:
            logger.info(
                f"[Pipeline] evidence sheet: polarity={evidence.answer_polarity} "
                f"doctrine={evidence.controlling_doctrine}"
            )

    # ── H: Build final prompt ─────────────────────────────────────────────────
    if evidence and evidence.answer_polarity not in (None, "insufficient"):
        props_txt = "\n".join(
            f"    • {p}"
            for p in (evidence.case_support + evidence.doctrine_support)
        ) or "    (see excerpts)"
        warnings_txt = "\n".join(
            f"    ⚠ {w}" for w in evidence.warnings
        ) or "    (none)"
        final_prompt = _PIPELINE_FINAL_PROMPT.format(
            system_prompt=LAW_SYSTEM_PROMPT,
            answer_polarity=evidence.answer_polarity or "see excerpts",
            controlling_doctrine=evidence.controlling_doctrine or "see excerpts",
            props=props_txt,
            warnings=warnings_txt,
            key_props=key_props or "(see excerpts)",
            sources=sources_text,
            q=query,
        )
    elif key_props:
        final_prompt = LAW_RAG_TEMPLATE.format(
            system_prompt=LAW_SYSTEM_PROMPT,
            q=query,
            key_props=key_props,
            sources=sources_text,
        )
    else:
        final_prompt = RAG_TEMPLATE.format(
            system_prompt=LAW_SYSTEM_PROMPT,
            q=query,
            sources=sources_text,
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    ai_response = ""
    final_model = model
    gen_opts = {
        "temperature": 0.3,
        "num_predict": 1024,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
    }

    try:
        if ollama_generate_with_retry:
            resp = ollama_generate_with_retry(
                model=model,
                prompt=final_prompt,
                max_retries=2,
                auto_fallback=True,
                auto_restart=False,
                options=gen_opts,
            )
            meta_resp = resp.get('_metadata', {}) or {}
            final_model = meta_resp.get('model_used', model)
        elif _ollama_module:
            client = _ollama_module.Client(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
            resp = client.generate(model=model, prompt=final_prompt, options=gen_opts)
        else:
            raise RuntimeError("Ollama not available")

        raw = (resp.get('response', '') if hasattr(resp, 'get')
               else getattr(resp, 'response', ''))
        ai_response = re.sub(r'<think>[\s\S]*?</think>\s*', '', raw).strip()

        # ── I: Contradiction check ─────────────────────────────────────────
        if (ai_response
                and detect_law_mode(query, reranked)
                and likely_contradiction(ai_response, reranked)):
            logger.info("[Pipeline] ⚠️ Contradiction detected — retrying")
            corr_prompt = CORRECTION_PROMPT.format(
                system_prompt=LAW_SYSTEM_PROMPT,
                q=query,
                key_props=key_props or "(see excerpts)",
                sources=sources_text,
            )
            corr_opts = {**gen_opts, "temperature": 0.1, "num_predict": 512}
            try:
                if ollama_generate_with_retry:
                    cr = ollama_generate_with_retry(
                        model=model, prompt=corr_prompt,
                        max_retries=2, auto_fallback=False,
                        auto_restart=False, options=corr_opts,
                    )
                elif _ollama_module:
                    cr = client.generate(model=model, prompt=corr_prompt, options=corr_opts)
                else:
                    raise RuntimeError("No LLM available")
                corr_raw = (cr.get('response', '') if hasattr(cr, 'get')
                            else getattr(cr, 'response', ''))
                corr_raw = re.sub(r'<think>[\s\S]*?</think>\s*', '', corr_raw).strip()
                if corr_raw:
                    ai_response = corr_raw
                    logger.info("[Pipeline] ✅ Contradiction corrected")
            except Exception as ce:
                logger.warning(f"[Pipeline] Correction retry failed: {ce}")

    except Exception as exc:
        logger.error(f"[Pipeline] Generation error: {exc}")
        ai_response = f"[Pipeline generation error: {exc}]"

    # ── Build results list for UI ─────────────────────────────────────────────
    results: List[dict] = []
    for item in reranked:
        content = item.get('content', '') or ''
        path_val = item.get('path', '')
        doc_name = path_val.split('/')[-1] if '/' in path_val else (
            path_val or item.get('document', 'Document')
        )
        section_id = item.get('section_id', '')
        section_title = item.get('section_title', '')
        results.append({
            "document":      doc_name,
            "reference":     f"{doc_name} [{section_id}]" if section_id else doc_name,
            "page":          item.get('page'),
            "chunk_id":      item.get('chunk_id'),
            "section_id":    section_id,
            "section_title": section_title,
            "similarity":    f"{item.get('sim', 0):.3f}",
            "preview":       (content[:120] + "…") if len(content) > 120 else content,
            "path":          path_val,
            "content":       content,
            "score":         item.get('sim', 0),
        })

    total_ms = round((time.time() - t0) * 1000, 1)

    return {
        # ── Core answer fields (compatible with existing UI rendering) ──
        "query":       query,
        "ai_response": ai_response,
        "answer":      ai_response,
        "model":       final_model,
        "method":      "pipeline",
        "backend":     backend.upper(),
        "sources":     results,
        "results":     results,
        "total_results": len(results),
        "prompt_used": final_prompt,

        # ── Pipeline-specific diagnostics ──
        "pipeline_info": {
            "query_analysis": {
                "language":           qa.language,
                "answer_type":        qa.answer_type,
                "case_name":          qa.case_name,
                "statute":            qa.statute,
                "primary_doctrine":   qa.primary_doctrine,
                "secondary_doctrines": qa.secondary_doctrines,
            },
            "query_buckets":      {bk: len(bv) for bk, bv in buckets.items()},
            "second_pass_used":   bool(second_pass_queries),
            "second_pass_queries": second_pass_queries,
            "gaps":               gaps,
            "confidence":         confidence,
            "key_props_found":    bool(key_props),
            "evidence_sheet": {
                "answer_polarity":    evidence.answer_polarity,
                "controlling_doctrine": evidence.controlling_doctrine,
                "warnings":           evidence.warnings,
                "missing":            evidence.missing,
            } if evidence else None,
        },

        # ── Timing ──
        "backend_info": {
            "search_time_ms":  total_ms,
            "distance_metric": distance_metric,
            "index_algorithm": index_algorithm,
            "collection_suffix": collection_suffix,
        },
    }
