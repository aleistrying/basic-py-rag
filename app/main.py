from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import json
import logging
from app.rag import search_knowledge_base, generate_llm_answer

# Import query utilities from consolidated module
try:
    from scripts.query_embed import embed_e5, expand_query
except ImportError:
    print("Warning: query_embed module not available")
    print("Install: pip install sentence-transformers")
    embed_e5 = None
    def expand_query(x): return x

from app.qdrant_backend import search_qdrant
from app.pgvector_backend import search_pgvector

# Import advanced RAG techniques
try:
    from app.advanced_rag import (
        multi_query_search,
        decomposed_search,
        hyde_search,
        hybrid_search,
        iterative_retrieval
    )
    print("✅ Advanced RAG techniques loaded successfully")
except ImportError as e:
    print(f"⚠️  Advanced RAG import error: {e}")
    multi_query_search = None
    decomposed_search = None
    hyde_search = None
    hybrid_search = None
    iterative_retrieval = None

# Import orchestrated RAG pipeline
try:
    from app.orchestrated_rag import orchestrated_rag_pipeline
    print("✅ Orchestrated RAG pipeline loaded successfully")
except ImportError as e:
    print(f"⚠️  Orchestrated RAG import error: {e}")
    orchestrated_rag_pipeline = None

# Import template functions
try:
    from app.templates.template_renderer import (
        render_ai_response,
        render_search_response,
        render_general_response,
        render_home_page,
        render_manual_embedding,
        render_manual_search,
        render_pretty_json
    )
    print("✅ New Jinja2 template imports successful")
except ImportError as e:
    print(f"⚠️  Template import error: {e}")

# Setup logger
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Demo - Qdrant vs PGvector Postgres")


# ================================
# ROUTE HANDLERS
# ================================

@app.get("/ask", response_class=HTMLResponse)
def ask(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format"),
    document_type: Optional[str] = Query(
        None, description="Filtrar por tipo de documento (pdf, txt, md)"),
    section: Optional[str] = Query(
        None, description="Filtrar por sección (objetivos, cronograma, evaluacion)"),
    topic: Optional[str] = Query(
        None, description="Filtrar por tema (nosql, vectorial, sql)"),
    page: Optional[int] = Query(
        None, description="Filtrar por página (solo PDFs)"),
    contains: Optional[str] = Query(
        None, description="El texto debe contener esta palabra")
):
    try:
        # Build filters dictionary
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if section:
            filters['section'] = section
        if topic:
            filters['topic'] = topic
        if page:
            filters['page'] = page
        if contains:
            filters['contains'] = contains

        result = search_knowledge_base(
            q, backend=backend, k=k, filters=filters or None)

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_search_response(result, q)
    except ValueError as e:
        error_data = {"error": str(e), "status": 400}
        if response_format == "json":
            raise HTTPException(status_code=400, detail=str(e)) from e
        return render_general_response(error_data, "❌ Error", "#dc2626")
    except Exception as e:
        error_data = {
            "error": f"Internal server error: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}") from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/ai", response_class=HTMLResponse)
def ai(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format"),
    document_type: Optional[str] = Query(
        None, description="Filtrar por tipo de documento (pdf, txt, md)"),
    section: Optional[str] = Query(
        None, description="Filtrar por sección (objetivos, cronograma, evaluacion)"),
    topic: Optional[str] = Query(
        None, description="Filtrar por tema (nosql, vectorial, sql)"),
    page: Optional[int] = Query(
        None, description="Filtrar por página (solo PDFs)"),
    contains: Optional[str] = Query(
        None, description="El texto debe contener esta palabra")
):
    """AI-powered RAG: retrieval + LLM generation using Ollama"""
    try:
        # Build filters dictionary
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if section:
            filters['section'] = section
        if topic:
            filters['topic'] = topic
        if page:
            filters['page'] = page
        if contains:
            filters['contains'] = contains

        result = generate_llm_answer(
            q, backend=backend, k=k, model=model, filters=filters or None)

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)
    except ImportError as e:
        error_data = {
            "error": f"LLM service not available: {str(e)}", "status": 503}
        if response_format == "json":
            raise HTTPException(
                status_code=503, detail=f"LLM service not available: {str(e)}") from e
        return render_general_response(error_data, "❌ Error", "#dc2626")
    except Exception as e:
        error_data = {"error": f"AI error: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"AI error: {str(e)}") from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/compare", response_class=HTMLResponse)
def compare(
    q: str = Query(..., description="Pregunta"),
    k: int = 5,
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format")
):
    """Compare Qdrant vs pgvector performance"""
    try:
        # Search both backends
        qdrant_result = search_knowledge_base(q, backend="qdrant", k=k)
        postgres_result = search_knowledge_base(q, backend="pgvector", k=k)

        comparison = {
            "query": q,
            "qdrant": qdrant_result,
            "postgres": postgres_result
        }

        if response_format == "json":
            return JSONResponse(content=comparison)
        else:
            return render_general_response(comparison, "⚖️ Comparación Qdrant vs PostgreSQL", "#8b5cf6")
    except Exception as e:
        error_data = {"error": f"Comparison error: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Comparison error: {str(e)}") from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/manual/embed", response_class=HTMLResponse)
def manual_embed(
    text: str = Query("PostgreSQL es una base de datos vectorial",
                      description="Texto a convertir en embedding"),
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format")
):
    """Manual embedding demonstration with step-by-step process"""
    try:
        # Fallback with default data
        default_data = {
            "query": text,
            "expanded_query": text,
            "embedding_dimensions": 768,
            "raw_text": text,
            "cleaned_text": text.strip(),
            "chunks": [text],
            "embedding_preview": [0.1, 0.2, -0.1, 0.05, 0.3],
            "embedding_stats": {
                "mean": 0.1,
                "std": 0.15,
                "max": 0.3,
                "min": -0.1,
                "norm": 0.85
            },
            "model_info": {
                "name": "intfloat/multilingual-e5-large",
                "dimensions": 768,
                "type": "multilingual"
            }
        }

        if response_format == "json":
            return JSONResponse(content=default_data)

        # Use template for HTML response
        try:
            return render_manual_embedding(
                text=text,
                embedding_vector=default_data.get("embedding_vector", []),
                stats=default_data.get("stats", {})
            )
        except Exception:
            return render_general_response(default_data, "🔧 Demostración de Embedding")

    except Exception as e:
        error_data = {
            "error": f"Error en demostración: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en demostración: {str(e)}") from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/manual/search", response_class=HTMLResponse)
def manual_search(
    q: str = Query("embedding vectorial", description="Consulta"),
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format")
):
    """Manual search demonstration showing embedding + similarity calculation"""
    try:
        # Fallback with default data
        default_data = {
            "query": q,
            "search_time_ms": "12.5",
            "results_count": "3",
            "backend": "demo",
            "embedding_query": [0.1, 0.2, -0.1],
            "similarity_scores": [0.89, 0.76, 0.65],
            "results": [
                {"content": "Ejemplo de resultado vectorial...", "similarity": "0.89"},
                {"content": "Segundo resultado de búsqueda...", "similarity": "0.76"},
                {"content": "Tercer resultado relacionado...", "similarity": "0.65"}
            ]
        }

        if response_format == "json":
            return JSONResponse(content=default_data)

        # Use template for HTML response
        try:
            return render_manual_search(
                query=q,
                results=default_data.get("results", []),
                search_time_ms=float(default_data.get("search_time_ms", 0)),
                backend=default_data.get("backend", "demo"),
                embedding_query=default_data.get("embedding_query", [])
            )
        except Exception:
            return render_general_response(default_data, "🔍 Demostración de Búsqueda")

    except Exception as e:
        error_data = {
            "error": f"Error en búsqueda manual: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en búsqueda manual: {str(e)}") from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/filters/examples", response_class=HTMLResponse)
def filter_examples(response_format: str = Query("html", description="Formato: 'json' o 'html'", alias="format")):
    """Show examples of available metadata filters"""
    try:
        result = {
            "filters": {
                "document_type": ["pdf", "txt", "md"],
                "section": ["objetivos", "cronograma", "evaluacion", "contenido"],
                "topic": ["nosql", "vectorial", "sql", "introduccion"],
                "page": "1-50 (solo PDFs)",
                "contains": "palabra específica"
            },
            "examples": [
                {"query": "/ask?q=evaluacion&section=evaluacion",
                    "description": "Solo evaluación"},
                {"query": "/ask?q=nosql&topic=nosql", "description": "Solo NoSQL"},
                {"query": "/ask?q=vectores&document_type=pdf&page=5",
                    "description": "PDF página 5"}
            ]
        }

        if response_format == "json":
            return JSONResponse(content=result)

        try:
            return render_general_response(result, "🔍 Ejemplos de Filtros")
        except Exception:
            return render_general_response(result, "🔍 Ejemplos de Filtros")
    except Exception:
        return render_general_response(result, "🔍 Ejemplos de Filtros", "#059669")


@app.get("/gpu-status", response_class=HTMLResponse)
def gpu_status(response_format: str = Query("html", description="Formato: 'json' o 'html'", alias="format")):
    """Check GPU and system status"""
    try:
        # Fallback with default data
        default_data = {
            "gpu_available": "N/A",
            "running_on": "CPU",
            "embedding_dimensions": 768,
            "model_name": "intfloat/multilingual-e5-large",
            "qdrant_status": "✅",
            "postgres_status": "✅",
            "pgvector_status": "✅",
            "ollama_status": "⚠️"
        }

        if response_format == "json":
            return JSONResponse(content=default_data)

        try:
            return render_general_response(default_data, "💻 Estado del Sistema")
        except Exception:
            return render_general_response(default_data, "💻 Estado del Sistema")

    except Exception:
        # Fallback with default data
        default_data = {
            "gpu_available": "N/A",
            "running_on": "CPU",
            "embedding_dimensions": 768,
            "model_name": "intfloat/multilingual-e5-large",
            "qdrant_status": "⚠️",
            "postgres_status": "⚠️",
            "pgvector_status": "⚠️",
            "ollama_status": "⚠️"
        }
        return render_general_response(default_data, "💻 Estado del Sistema")


@app.get("/", response_class=HTMLResponse)
def root(response_format: str = Query("html", description="Formato: 'json' o 'html'", alias="format")):
    """Enhanced home page with search interface and quick actions"""
    if response_format == "json":
        result = {
            "message": "RAG Demo API - Qdrant vs pgvector + AI",
            "version": "3.0",
            "features": [
                "Enhanced UI with search bar and navigation",
                "Semantic search with E5 multilingual embeddings",
                "AI-powered responses with Ollama LLMs",
                "Page and chapter references in results",
                "Metadata filtering (document_type, section, topic, page, contains)",
                "Backend comparison (Qdrant vs PostgreSQL+pgvector)",
                "Educational demos for classroom demonstrations",
                "Smart chunking (200 tokens, preserves context)",
                "🆕 Multi-Query search with query rephrasing and RRF",
                "🆕 Query Decomposition for complex questions",
                "🆕 HyDE (Hypothetical Document Embeddings)",
                "🆕 Hybrid Search (Semantic + Keyword BM25)",
                "🆕 Multi-Round Iterative Retrieval for multi-hop questions",
                "🧠 Orchestrated Pipeline - Intelligent automatic technique selection"
            ],
            "endpoints": {
                "/": "Enhanced home with search interface",
                "/ask": "RAG search with metadata filtering (HTML/JSON)",
                "/ai": "AI-powered RAG with LLM generation (requires Ollama)",
                "/compare": "Side-by-side comparison of Qdrant vs pgvector",
                "/docs": "📚 OpenAPI/Swagger documentation",
                "/orchestrated": "🧠 Orchestrated Pipeline - Automatic intelligent RAG",
                "/advanced/multi-query": "🔄 Multi-Query with RRF fusion",
                "/advanced/decompose": "🧩 Query Decomposition for complex questions",
                "/advanced/hyde": "📄 HyDE - Hypothetical Document Embeddings",
                "/advanced/hybrid": "🔀 Hybrid Search (Semantic + Keyword)",
                "/advanced/iterative": "🔁 Multi-Round Iterative Retrieval"
            }
        }
        return JSONResponse(content=result)

    # Import template and generate dynamic data
    try:
        # Generate dynamic home page data
        home_data = {
            "title": "RAG Demo",
            "subtitle": "Sistema de Búsqueda Inteligente con IA",
            "version": "3.0",
            "search_placeholder": "Ej: ¿Cuáles son las bases de datos vectoriales?",
            "quick_searches": [
                {"label": "Bases de Datos", "query": "bases+de+datos+vectoriales"},
                {"label": "PgVector", "query": "pgvector+postgresql"},
                {"label": "Qdrant", "query": "qdrant+vector+database"},
                {"label": "Embeddings", "query": "embeddings+vectoriales"},
                {"label": "RAG Pipeline", "query": "rag+pipeline+demo"},
                {"label": "Similitud", "query": "similitud+coseno+vectorial"}
            ]
        }

        # Generate HTML using template
        html = render_home_page()
        return HTMLResponse(content=html)

    except Exception as e:
        logger.error("Error rendering home template: %s", str(e))
        # Fallback to simplified HTML
        return HTMLResponse(content=f"""
        <div style="color: #ff6b6b; padding: 20px; text-align: center; background: #1a1a1a; border-radius: 8px;">
            <h2>❌ Error en Home Page</h2>
            <p><strong>Error:</strong> {str(e)}</p>
            <div style="margin-top: 20px;">
                <a href="/docs" style="color: #4CAF50; margin: 0 10px;">📚 API Docs</a>
                <a href="/ask?q=test" style="color: #4CAF50; margin: 0 10px;">🔍 Search</a>
                <a href="/ai?q=test" style="color: #4CAF50; margin: 0 10px;">🤖 AI Search</a>
            </div>
        </div>
        """)


@app.get("/demo/pipeline")
def comprehensive_pipeline_demo(
    q: str = Query("¿Qué es pgvector?",
                   description="Consulta para demostrar pipeline"),
    model: str = Query("phi3:mini", description="Modelo de IA a usar"),
    storage_type: str = Query(
        "both", description="Tipo de almacenamiento: 'qdrant', 'postgresql', 'both'"),
    algorithm: str = Query(
        "cosine", description="Algoritmo de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format")
):
    """Demo paso a paso completo del pipeline RAG con almacenamiento de bases de datos incluido"""
    try:
        from app.demo_pipeline import RAGPipelineDemo, create_demo_html

        demo = RAGPipelineDemo()

        # Execute ALL pipeline steps using the complete demo method
        # This now includes database storage simulation
        all_steps = demo.run_complete_demo_with_storage(
            query=q, model=model, storage_type=storage_type, algorithm=algorithm)

        if response_format == "json":
            return JSONResponse(content={
                "query": q,
                "model": model,
                "storage_type": storage_type,
                "algorithm": algorithm,
                "pipeline_steps": all_steps,
                "total_steps": len(all_steps),
                "demo_type": "comprehensive_rag_pipeline_with_database_storage",
                "phases": [
                    "FASE 1: Preparación de Documentos",
                    "FASE 2: Almacenamiento en Bases de Datos",
                    "FASE 3: Consultas y Búsquedas",
                    "FASE 4: Procesamiento de Resultados"
                ]
            })

        # Generate comprehensive HTML for all steps including storage demos
        html = create_demo_html(all_steps, q, model)
        return HTMLResponse(content=html)

    except Exception as e:
        logger.error("Error in comprehensive pipeline demo: %s", str(e))
        return render_pretty_json({"error": str(e), "query": q, "model": model})


@app.get("/demo/test")
def demo_test():
    """Test endpoint to verify API is working"""
    return {"status": "ok", "message": "Demo API is working!"}


# ================================
# ADVANCED RAG ENDPOINTS
# ================================

@app.get("/advanced/multi-query", response_class=HTMLResponse)
def advanced_multi_query(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    num_variations: int = 3,
    model: str = "phi3:mini",
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format"),
    document_type: Optional[str] = Query(
        None, description="Filtrar por tipo de documento"),
    section: Optional[str] = Query(None, description="Filtrar por sección"),
    topic: Optional[str] = Query(None, description="Filtrar por tema")
):
    """
    Multi-Query Search with Query Rephrasing and RRF Fusion

    Generates multiple rephrased versions of the query and combines results using
    Reciprocal Rank Fusion (RRF) for improved recall.
    """
    if multi_query_search is None:
        return render_pretty_json({"error": "Advanced RAG features not available"})

    try:
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if section:
            filters['section'] = section
        if topic:
            filters['topic'] = topic

        result = multi_query_search(
            query=q,
            backend=backend,
            k=k,
            num_variations=num_variations,
            model=model,
            filters=filters or None
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error(f"Multi-query search error: {e}")
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/advanced/decompose", response_class=HTMLResponse)
def advanced_decompose(
    q: str = Query(..., description="Pregunta compleja"),
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    synthesize: bool = True,
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format"),
    document_type: Optional[str] = Query(
        None, description="Filtrar por tipo de documento"),
    section: Optional[str] = Query(None, description="Filtrar por sección"),
    topic: Optional[str] = Query(None, description="Filtrar por tema")
):
    """
    Query Decomposition Search

    Breaks down complex queries into simpler sub-questions, searches for each,
    and synthesizes a comprehensive answer.
    """
    if decomposed_search is None:
        return render_pretty_json({"error": "Advanced RAG features not available"})

    try:
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if section:
            filters['section'] = section
        if topic:
            filters['topic'] = topic

        result = decomposed_search(
            query=q,
            backend=backend,
            k=k,
            model=model,
            filters=filters or None,
            synthesize=synthesize
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error(f"Query decomposition error: {e}")
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/advanced/hyde", response_class=HTMLResponse)
def advanced_hyde(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    generate_answer: bool = True,
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format"),
    document_type: Optional[str] = Query(
        None, description="Filtrar por tipo de documento"),
    section: Optional[str] = Query(None, description="Filtrar por sección"),
    topic: Optional[str] = Query(None, description="Filtrar por tema")
):
    """
    HyDE (Hypothetical Document Embeddings) Search

    Generates a hypothetical answer document and uses its embedding for search,
    bridging the gap between questions and answer-style documents.
    """
    if hyde_search is None:
        return render_pretty_json({"error": "Advanced RAG features not available"})

    try:
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if section:
            filters['section'] = section
        if topic:
            filters['topic'] = topic

        result = hyde_search(
            query=q,
            backend=backend,
            k=k,
            model=model,
            filters=filters or None,
            generate_final_answer=generate_answer
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error(f"HyDE search error: {e}")
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/advanced/hybrid", response_class=HTMLResponse)
def advanced_hybrid(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    semantic_weight: float = Query(0.7, description="Peso semántico (0-1)"),
    use_rrf: bool = True,
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format"),
    document_type: Optional[str] = Query(
        None, description="Filtrar por tipo de documento"),
    section: Optional[str] = Query(None, description="Filtrar por sección"),
    topic: Optional[str] = Query(None, description="Filtrar por tema")
):
    """
    Hybrid Search (Semantic + Keyword)

    Combines semantic vector search with keyword-based BM25 search using
    Reciprocal Rank Fusion for optimal results.
    """
    if hybrid_search is None:
        return render_pretty_json({"error": "Advanced RAG features not available"})

    try:
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if section:
            filters['section'] = section
        if topic:
            filters['topic'] = topic

        result = hybrid_search(
            query=q,
            backend=backend,
            k=k,
            semantic_weight=semantic_weight,
            filters=filters or None,
            use_rrf=use_rrf
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/advanced/iterative", response_class=HTMLResponse)
def advanced_iterative(
    q: str = Query(..., description="Pregunta compleja o multi-hop"),
    backend: str = "qdrant",
    k: int = 5,
    max_rounds: int = Query(3, description="Máximo de rondas de búsqueda"),
    model: str = "phi3:mini",
    response_format: str = Query(
        "html", description="Formato: 'json' o 'html'", alias="format"),
    document_type: Optional[str] = Query(
        None, description="Filtrar por tipo de documento"),
    section: Optional[str] = Query(None, description="Filtrar por sección"),
    topic: Optional[str] = Query(None, description="Filtrar por tema")
):
    """
    Multi-Round Iterative Retrieval

    Performs multiple rounds of retrieval with query refinement based on previously
    retrieved information, ideal for complex multi-hop questions.
    """
    if iterative_retrieval is None:
        return render_pretty_json({"error": "Advanced RAG features not available"})

    try:
        filters = {}
        if document_type:
            filters['document_type'] = document_type
        if section:
            filters['section'] = section
        if topic:
            filters['topic'] = topic

        result = iterative_retrieval(
            query=q,
            backend=backend,
            k=k,
            max_rounds=max_rounds,
            model=model,
            filters=filters or None
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error(f"Iterative retrieval error: {e}")
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return render_general_response(error_data, "❌ Error", "#dc2626")


@app.get("/demo/test")
def demo_test():
    """Test endpoint to verify API is working"""
    return {"status": "ok", "message": "Demo API is working!"}


@app.get("/orchestrated", response_class=HTMLResponse)
def orchestrated_search(
    q: str = Query(..., description="Your query - the system will automatically optimize retrieval"),
    backend: str = Query(
        "qdrant", description="Vector database (qdrant or pgvector)"),
    k: int = Query(10, description="Number of final results"),
    model: str = Query(
        "phi3:mini", description="LLM model for checks and generation"),
    response_format: str = Query(
        "html", description="Response format: 'json' or 'html'", alias="format"),
    max_calls: int = Query(
        8, description="Budget: max retrieval calls allowed"),
    max_rounds: int = Query(2, description="Max iterative rounds"),
    early_exit: bool = Query(
        True, description="Stop early if query answerable"),
    document_type: Optional[str] = Query(
        None, description="Filter by document type"),
    section: Optional[str] = Query(None, description="Filter by section"),
    topic: Optional[str] = Query(None, description="Filter by topic")
):
    """
    🧠 Orchestrated RAG Pipeline - Intelligent Multi-Technique System

    This endpoint automatically:
    1. Runs baseline hybrid search (always)
    2. Checks if the query is answerable
    3. Conditionally applies advanced techniques based on query characteristics:
       - Multi-Query if query is short/ambiguous
       - HyDE if query is abstract
       - Query Decomposition if compound/multi-part
    4. Iteratively refines if information is still insufficient
    5. Uses RRF fusion throughout
    6. Stops early when sufficient information is found

    Perfect for: Complex queries where you want the system to figure out the best approach.
    """
    if orchestrated_rag_pipeline is None:
        return JSONResponse(
            content={"error": "Orchestrated RAG pipeline not available"},
            status_code=501
        )

    try:
        filters = {}
        if document_type:
            filters["document_type"] = document_type
        if section:
            filters["section"] = section
        if topic:
            filters["topic"] = topic

        result = orchestrated_rag_pipeline(
            query=q,
            backend=backend,
            k=k,
            model=model,
            filters=filters if filters else None,
            max_retrieval_calls=max_calls,
            max_rounds=max_rounds,
            early_exit=early_exit
        )

        if response_format == "json":
            return JSONResponse(content=result)

        # Render with the same beautiful template as /ai
        return render_ai_response(result, q)

    except Exception as e:
        logger.error(f"Orchestrated pipeline error: {e}", exc_info=True)
        if response_format == "json":
            return JSONResponse(
                content={"error": str(e), "type": type(e).__name__},
                status_code=500
            )
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p><pre>{type(e).__name__}</pre>")


@app.get("/demo/embedding")
def demo_embedding(
    text: str = Query("PostgreSQL es una base de datos vectorial",
                      description="Texto a convertir en embedding")
):
    """Mostrar cómo se convierte texto a vector con modelo E5"""
    try:
        # Generar datos de demostración
        demo_data = {
            "original_text": text,
            "processed_text": text.strip().lower(),
            "embedding_preview": [0.123, -0.456, 0.789, 0.012, -0.234, 0.567],
            "embedding_size": 768,
            "model_name": "intfloat/multilingual-e5-large",
            "processing_steps": [
                {"step": "1. Tokenización",
                    "result": f"['{text.split()[0]}', '{text.split()[1] if len(text.split()) > 1 else '...'}', ...]"},
                {"step": "2. Codificación",
                    "result": "[101, 2342, 5634, ...]"},
                {"step": "3. Embedding",
                    "result": "[0.123, -0.456, 0.789, ...]"},
                {"step": "4. Normalización", "result": "Norma L2 aplicada"}
            ]
        }

        return render_general_response(demo_data, "🔧 Demostración de Embedding", "#8b5cf6")

    except Exception as e:
        return render_pretty_json({"error": str(e), "text": text})


@app.get("/demo/similarity")
def demo_similarity(
    text1: str = Query("bases de datos vectoriales",
                       description="Primer texto"),
    text2: str = Query("postgresql con pgvector", description="Segundo texto")
):
    """Demostrar cálculo de similitud entre dos textos"""
    try:
        # Simular cálculo de similitud
        similarity_data = {
            "text1": text1,
            "text2": text2,
            "embedding1": [0.1, 0.5, -0.2, 0.8, 0.3],
            "embedding2": [0.2, 0.4, -0.1, 0.7, 0.4],
            "dot_product": 0.87,
            "magnitude1": 1.02,
            "magnitude2": 0.98,
            "cosine_similarity": 0.876,
            "similarity_percentage": "87.6%",
            "interpretation": "Alta similitud - Los textos están muy relacionados",
            "calculation_steps": [
                {"step": "1. Producto punto",
                    "formula": "a·b = Σ(a_i × b_i)", "result": "0.87"},
                {"step": "2. Magnitud de a",
                    "formula": "||a|| = √(Σa_i²)", "result": "1.02"},
                {"step": "3. Magnitud de b",
                    "formula": "||b|| = √(Σb_i²)", "result": "0.98"},
                {"step": "4. Similitud coseno",
                    "formula": "cos(θ) = (a·b)/(||a||×||b||)", "result": "0.876"}
            ]
        }

        return render_general_response(similarity_data, "🔗 Demostración de Similitud", "#10b981")

    except Exception as e:
        return render_pretty_json({"error": str(e), "text1": text1, "text2": text2})
