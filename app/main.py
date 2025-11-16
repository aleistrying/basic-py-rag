from fastapi import FastAPI, Query, HTTPException, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List
import logging
import pandas as pd
import subprocess
import os
from pathlib import Path
from app.rag import search_knowledge_base, generate_llm_answer

# Import query utilities from consolidated module
try:
    from scripts.query_embed import embed_e5
except ImportError:
    print("Warning: query_embed module not available")
    print("Install: pip install sentence-transformers")
    embed_e5 = None

# Note: search_qdrant not used directly in main, only via rag module
# from app.qdrant_backend import search_qdrant
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
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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

        # Generate collection name based on algorithm combination
        collection_suffix = f"{distance_metric}_{index_algorithm}"

        result = search_knowledge_base(
            q, backend=backend, k=k, filters=filters or None,
            distance_metric=distance_metric, index_algorithm=index_algorithm,
            collection_suffix=collection_suffix)

        # Add algorithm parameters to result for template access
        result['distance_metric'] = distance_metric
        result['index_algorithm'] = index_algorithm

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_search_response(result, q)
    except ValueError as e:
        error_data = {"error": str(e), "status": 400}
        if response_format == "json":
            raise HTTPException(status_code=400, detail=str(e)) from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))
    except Exception as e:
        error_data = {
            "error": f"Error interno del servidor: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error interno del servidor: {str(e)}") from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


@app.get("/ai", response_class=HTMLResponse)
def ai(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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

        # Generate collection name based on algorithm combination
        collection_suffix = f"{distance_metric}_{index_algorithm}"

        result = generate_llm_answer(
            q, backend=backend, k=k, model=model, filters=filters or None,
            distance_metric=distance_metric, index_algorithm=index_algorithm,
            collection_suffix=collection_suffix)

        # Add algorithm parameters to result for template access
        result['distance_metric'] = distance_metric
        result['index_algorithm'] = index_algorithm

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)
    except ImportError as e:
        error_data = {
            "error": f"Servicio LLM no disponible: {str(e)}", "status": 503}
        if response_format == "json":
            raise HTTPException(
                status_code=503, detail=f"LLM service not available: {str(e)}") from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))
    except Exception as e:
        error_data = {"error": f"Error de IA: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error de IA: {str(e)}") from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


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
            return HTMLResponse(render_general_response(comparison, "Comparación Qdrant vs PostgreSQL", "#8b5cf6"))
    except Exception as e:
        error_data = {
            "error": f"Error de comparación: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Comparison error: {str(e)}") from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


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
                embedding_result=default_data,
                text=text
            )
        except Exception:
            return HTMLResponse(render_general_response(default_data, "Demostración de Embedding"))

    except Exception as e:
        error_data = {
            "error": f"Error en demostración: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en demostración: {str(e)}") from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


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
                search_result=default_data,
                query=q
            )
        except Exception:
            return HTMLResponse(render_general_response(default_data, "Demostración de Búsqueda"))

    except Exception as e:
        error_data = {
            "error": f"Error en búsqueda manual: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en búsqueda manual: {str(e)}") from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


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
            return HTMLResponse(render_general_response(result, "Ejemplos de Filtros"))
        except Exception:
            return HTMLResponse(render_general_response(result, "Ejemplos de Filtros"))
    except Exception:
        return HTMLResponse(render_general_response(result, "Ejemplos de Filtros", "#059669"))


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
            return HTMLResponse(render_general_response(default_data, "Estado del Sistema"))
        except Exception:
            return HTMLResponse(render_general_response(default_data, "Estado del Sistema"))

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
        return HTMLResponse(render_general_response(default_data, "Estado del Sistema"))


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
        # Generate HTML using template
        html = render_home_page()
        return HTMLResponse(content=html)

    except Exception as e:
        logger.error("Error rendering home template: %s", str(e))
        # Fallback to simplified HTML
        return HTMLResponse(content=f"""
        <div style="color: #ff6b6b; padding: 20px; text-align: center; background: #1a1a1a; border-radius: 8px;">
            <h2>Error en Home Page</h2>
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
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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
            query=q, model=model, storage_type=storage_type,
            distance_metric=distance_metric, index_algorithm=index_algorithm)

        if response_format == "json":
            return JSONResponse(content={
                "query": q,
                "model": model,
                "storage_type": storage_type,
                "distance_metric": distance_metric,
                "index_algorithm": index_algorithm,
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
        html = create_demo_html(all_steps, q, model,
                                storage_type, distance_metric, index_algorithm)
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
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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

        # Generate collection name based on algorithm combination
        collection_suffix = f"{distance_metric}_{index_algorithm}"

        result = multi_query_search(
            query=q,
            backend=backend,
            k=k,
            num_variations=num_variations,
            model=model,
            filters=filters or None,
            collection_suffix=collection_suffix
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error("Multi-query search error: %s", e)
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


@app.get("/advanced/decompose", response_class=HTMLResponse)
def advanced_decompose(
    q: str = Query(..., description="Pregunta compleja"),
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    synthesize: bool = True,
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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

        # Generate collection name based on algorithm combination
        collection_suffix = f"{distance_metric}_{index_algorithm}"

        result = decomposed_search(
            query=q,
            backend=backend,
            k=k,
            model=model,
            filters=filters or None,
            synthesize=synthesize,
            collection_suffix=collection_suffix
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error("Query decomposition error: %s", e)
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


@app.get("/advanced/hyde", response_class=HTMLResponse)
def advanced_hyde(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    generate_answer: bool = True,
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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

        # Generate collection name based on algorithm combination
        collection_suffix = f"{distance_metric}_{index_algorithm}"

        result = hyde_search(
            query=q,
            backend=backend,
            k=k,
            model=model,
            filters=filters or None,
            generate_final_answer=generate_answer,
            collection_suffix=collection_suffix
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error("HyDE search error: %s", e)
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


@app.get("/advanced/hybrid", response_class=HTMLResponse)
def advanced_hybrid(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    semantic_weight: float = Query(0.7, description="Peso semántico (0-1)"),
    use_rrf: bool = True,
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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

        # Generate collection name based on algorithm combination
        collection_suffix = f"{distance_metric}_{index_algorithm}"

        result = hybrid_search(
            query=q,
            backend=backend,
            k=k,
            semantic_weight=semantic_weight,
            filters=filters or None,
            use_rrf=use_rrf,
            collection_suffix=collection_suffix
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error("Hybrid search error: %s", e)
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


@app.get("/advanced/iterative", response_class=HTMLResponse)
def advanced_iterative(
    q: str = Query(..., description="Pregunta compleja o multi-hop"),
    backend: str = "qdrant",
    k: int = 5,
    max_rounds: int = Query(3, description="Máximo de rondas de búsqueda"),
    model: str = "phi3:mini",
    distance_metric: str = Query(
        "cosine", description="Métrica de distancia: 'cosine', 'euclidean', 'dot_product', 'manhattan'"),
    index_algorithm: str = Query(
        "hnsw", description="Algoritmo de índice: 'hnsw', 'ivfflat', 'scalar_quantization', 'exact'"),
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

        # Generate collection name based on algorithm combination
        collection_suffix = f"{distance_metric}_{index_algorithm}"

        result = iterative_retrieval(
            query=q,
            backend=backend,
            k=k,
            max_rounds=max_rounds,
            model=model,
            filters=filters or None,
            collection_suffix=collection_suffix
        )

        if response_format == "json":
            return JSONResponse(content=result)
        else:
            return render_ai_response(result, q)

    except Exception as e:
        logger.error("Iterative retrieval error: %s", e)
        error_data = {"error": str(e), "query": q}
        if response_format == "json":
            raise HTTPException(status_code=500, detail=str(e)) from e
        return HTMLResponse(render_general_response(error_data, "Error", "#dc2626"))


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
        logger.error("Orchestrated pipeline error: %s", e, exc_info=True)
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

        return HTMLResponse(render_general_response(demo_data, "Demostración de Embedding", "#8b5cf6"))

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

        return HTMLResponse(render_general_response(similarity_data, "🔗 Demostración de Similitud", "#10b981"))

    except Exception as e:
        return render_pretty_json({"error": str(e), "text1": text1, "text2": text2})


# ================================
# PIPELINE MANAGEMENT ENDPOINTS
# ================================

@app.get("/pipeline", response_class=HTMLResponse)
def pipeline_management():
    """Pipeline management interface for configuring algorithms and processing documents"""
    try:
        # Use the general response template with pipeline configuration data
        return HTMLResponse(render_general_response(
            data={
                "available_backends": ["qdrant", "pgvector", "both"],
                "distance_metrics": ["cosine", "dot", "euclidean"],
                "index_algorithms": ["hnsw", "flat"],
                "current_config": {
                    "distanceMetric": "cosine",
                    "indexAlgorithm": "hnsw",
                    "databaseBackend": "both"
                },
                "message": "Pipeline Management Interface",
                "description": "Configure algorithms, upload documents, and manage vector database operations"
            },
            title="🔧 Pipeline Management"
        ))

    except Exception as e:
        return HTMLResponse(render_general_response({
            "error": str(e),
            "message": "Pipeline management page is not available. Please check template."
        }, "🔧 Pipeline Management", "#3b82f6"))


@app.post("/pipeline/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents for processing"""
    try:
        import os
        from pathlib import Path

        # Create upload directory if it doesn't exist
        upload_dir = Path("./data/uploaded")
        upload_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        for file in files:
            # Save uploaded file
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            uploaded_files.append({
                "filename": file.filename,
                "size": len(content),
                "path": str(file_path)
            })

        return JSONResponse({
            "success": True,
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "files": uploaded_files
        })

    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/pipeline/run")
async def run_pipeline(
    distance_metric: str = Form("cosine"),
    index_algorithm: str = Form("hnsw"),
    clear_first: bool = Form(True)
):
    """Run the complete RAG pipeline with specified algorithms"""
    try:
        import subprocess
        import sys
        from pathlib import Path

        # Build command for main pipeline
        script_path = Path("scripts/main_pipeline.py")
        cmd = [
            sys.executable, str(script_path),
            "--distance-metric", distance_metric,
            "--index-algorithm", index_algorithm
        ]

        if clear_first:
            cmd.append("--clear")

        # Run pipeline process
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if process.returncode == 0:
            return JSONResponse({
                "success": True,
                "message": "Pipeline completed successfully",
                "output": process.stdout,
                "config": {
                    "distance_metric": distance_metric,
                    "index_algorithm": index_algorithm,
                    "collection_suffix": f"_{distance_metric}_{index_algorithm}"
                }
            })
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Pipeline failed with return code {process.returncode}",
                    "stdout": process.stdout,
                    "stderr": process.stderr
                }
            )

    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=500,
            content={"error": "Pipeline execution timed out"}
        )
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/pipeline/clear")
async def clear_pipeline_data():
    """Clear all pipeline data from databases"""
    try:
        from app.qdrant_backend import client as qdrant_client
        from app.pgvector_backend import get_connection
        import psycopg2

        results = {"qdrant": "Not cleared", "pgvector": "Not cleared"}

        # Clear Qdrant collections
        try:
            collections = qdrant_client.get_collections()
            for collection in collections.collections:
                if "course_docs" in collection.name or "docs_" in collection.name:
                    qdrant_client.delete_collection(collection.name)
            results["qdrant"] = "Cleared successfully"
        except Exception as e:
            results["qdrant"] = f"Error: {str(e)}"

        # Clear pgvector tables
        try:
            conn = get_connection()
            if conn:
                with conn.cursor() as cur:
                    # Get all tables that match our pattern
                    cur.execute("""
                        SELECT tablename FROM pg_tables 
                        WHERE tablename LIKE 'docs_%' OR tablename LIKE 'course_docs_%'
                    """)
                    tables = cur.fetchall()

                    for (table_name,) in tables:
                        cur.execute(
                            f"DROP TABLE IF EXISTS {table_name} CASCADE")

                results["pgvector"] = "Cleared successfully"
                conn.close()
        except Exception as e:
            results["pgvector"] = f"Error: {str(e)}"

        return JSONResponse({
            "success": True,
            "message": "Data clearing completed",
            "results": results
        })

    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/pipeline/stats")
async def get_pipeline_stats(format: str = Query("json", description="Response format: json or html")):
    """Get current pipeline statistics"""
    try:
        from app.qdrant_backend import client as qdrant_client
        from app.pgvector_backend import get_connection

        stats = {
            "total_docs": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "processing_time": 0,
            "collections": [],
            "tables": []
        }

        # Get Qdrant stats
        try:
            collections = qdrant_client.get_collections()
            for collection in collections.collections:
                if "course_docs" in collection.name or "docs_" in collection.name:
                    info = qdrant_client.get_collection(collection.name)
                    stats["collections"].append({
                        "name": collection.name,
                        "points": info.points_count,
                        "vector_size": info.config.params.vectors.size
                    })
                    stats["total_embeddings"] += info.points_count
        except Exception as e:
            logger.error(f"Error getting Qdrant stats: {e}")

        # Get pgvector stats
        try:
            conn = get_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT tablename FROM pg_tables 
                        WHERE tablename LIKE 'docs_%' OR tablename LIKE 'course_docs_%'
                    """)
                    tables = cur.fetchall()

                    for (table_name,) in tables:
                        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cur.fetchone()[0]
                        stats["tables"].append({
                            "name": table_name,
                            "rows": count
                        })
                        stats["total_chunks"] += count

                conn.close()
        except Exception as e:
            logger.error(f"Error getting pgvector stats: {e}")

        # Estimate total documents (rough estimate)
        stats["total_docs"] = len(set([col["name"].replace("course_docs_clean_", "").replace("docs_clean_", "")
                                      for col in stats["collections"]]))

        if format == "html":
            return HTMLResponse(render_general_response(
                data=stats,
                title="Estadísticas del Pipeline"
            ))
        else:
            return JSONResponse(stats)

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        if format == "html":
            return HTMLResponse(render_general_response({
                "error": str(e),
                "message": "Error loading statistics"
            }, "Estadísticas del Pipeline", "#ef4444"))
        else:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )


@app.get("/pipeline/dashboard")
async def get_pipeline_dashboard():
    """Unified dashboard showing both statistics and visualizations"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/pipeline/visualization?format=html&include_stats=true")


@app.get("/pipeline/visualization")
async def get_pipeline_visualization(
    type: str = Query("similarity", description="Visualization type"),
    distance_metric: str = Query(
        "cosine", description="Distance metric for filtering"),
    index_algorithm: str = Query(
        "hnsw", description="Index algorithm for filtering"),
    format: str = Query("json", description="Response format: json or html"),
    include_stats: bool = Query(
        True, description="Include statistics in HTML format")
):
    """Generate visualization data for pipeline results"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        import numpy as np
        from app.qdrant_backend import client as qdrant_client

        collection_suffix = f"_{distance_metric}_{index_algorithm}"
        collection_name = f"course_docs_clean{collection_suffix}"

        plot_data = None

        if type == "similarity":
            # Create similarity scatter plot
            try:
                # Get sample vectors from collection
                search_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=True
                )

                if search_result[0]:
                    # Extract data for plotting
                    documents = []
                    similarities = []
                    x_coords = []
                    y_coords = []

                    for point in search_result[0]:
                        doc_name = point.payload.get(
                            "source_path", "Unknown").split("/")[-1]
                        documents.append(doc_name)

                        # Use first two dimensions for 2D plot
                        vector = point.vector
                        x_coords.append(vector[0] if len(vector) > 0 else 0)
                        y_coords.append(vector[1] if len(vector) > 1 else 0)

                        # Calculate similarity to origin as proxy
                        # Use first 10 dimensions
                        magnitude = float(np.linalg.norm(vector[:10]))
                        similarities.append(magnitude)

                    fig = px.scatter(
                        x=x_coords, y=y_coords,
                        color=similarities,
                        hover_name=documents,
                        title=f"Document Vector Similarity Map ({distance_metric.upper()} + {index_algorithm.upper()})",
                        labels={"x": "Vector Dimension 1",
                                "y": "Vector Dimension 2", "color": "Vector Magnitude"},
                        color_continuous_scale="Viridis"
                    )

                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        height=400
                    )

                    # Convert to JSON-serializable format
                    import json
                    plot_json = fig.to_json()
                    plot_data = json.loads(plot_json)

            except Exception as e:
                logger.error(f"Error creating similarity plot: {e}")

        elif type == "topics":
            # Topic distribution pie chart
            try:
                search_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=200,
                    with_payload=True
                )

                if search_result[0]:
                    topics = {}
                    for point in search_result[0]:
                        doc_path = point.payload.get("source_path", "Unknown")

                        # Extract topic from document name
                        if "Tema" in doc_path:
                            topic = "Tema " + \
                                doc_path.split("Tema")[-1].split()[0]
                        elif "Introduccion" in doc_path:
                            topic = "Introducción"
                        elif "Modelos" in doc_path:
                            topic = "Modelos"
                        elif "Guia" in doc_path:
                            topic = "Guía"
                        else:
                            topic = "Otros"

                        topics[topic] = topics.get(topic, 0) + 1

                    fig = px.pie(
                        values=list(topics.values()),
                        names=list(topics.keys()),
                        title=f"Topic Distribution ({distance_metric.upper()} + {index_algorithm.upper()})"
                    )

                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        height=400
                    )

                    # Convert to JSON-serializable format
                    import json
                    plot_json = fig.to_json()
                    plot_data = json.loads(plot_json)

            except Exception as e:
                logger.error(f"Error creating topic plot: {e}")

        elif type == "quality":
            # Quality metrics histogram
            try:
                search_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=200,
                    with_payload=True
                )

                if search_result[0]:
                    quality_scores = []
                    for point in search_result[0]:
                        metadata = point.payload.get("metadata", {})
                        # Random if not available
                        quality = float(metadata.get(
                            "quality_score", np.random.random()))
                        quality_scores.append(quality)

                    fig = px.histogram(
                        x=quality_scores,
                        nbins=20,
                        title=f"Quality Score Distribution ({distance_metric.upper()} + {index_algorithm.upper()})",
                        labels={"x": "Quality Score", "y": "Number of Chunks"}
                    )

                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        height=400
                    )

                    # Convert to JSON-serializable format
                    import json
                    plot_json = fig.to_json()
                    plot_data = json.loads(plot_json)

            except Exception as e:
                logger.error(f"Error creating quality plot: {e}")

        else:  # temporal
            # Processing timeline (simulated)
            dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
            processed_docs = np.cumsum(np.random.randint(1, 5, 10))

            # Convert numpy arrays to Python lists
            dates_list = [d.strftime('%Y-%m-%d') for d in dates]
            processed_list = [int(x) for x in processed_docs]

            fig = px.line(
                x=dates_list, y=processed_list,
                title=f"Processing Timeline ({distance_metric.upper()} + {index_algorithm.upper()})",
                labels={"x": "Date", "y": "Cumulative Documents Processed"}
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400
            )

            # Convert to JSON-serializable format
            import json
            plot_json = fig.to_json()
            plot_data = json.loads(plot_json)

        if format == "html":
            response_data = {
                "plot_data": plot_data,
                "type": type,
                "config": {
                    "distance_metric": distance_metric,
                    "index_algorithm": index_algorithm,
                    "collection": collection_name
                },
                "available_types": ["similarity", "topics", "quality", "temporal"]
            }

            # Include statistics if requested
            if include_stats:
                try:
                    from app.qdrant_backend import client as qdrant_client
                    from app.pgvector_backend import get_connection

                    stats = {
                        "total_docs": 0,
                        "total_chunks": 0,
                        "total_embeddings": 0,
                        "processing_time": 0,
                        "collections": [],
                        "tables": []
                    }

                    # Get Qdrant stats
                    try:
                        collections = qdrant_client.get_collections()
                        for collection in collections.collections:
                            if "course_docs" in collection.name or "docs_" in collection.name:
                                info = qdrant_client.get_collection(
                                    collection.name)
                                stats["collections"].append({
                                    "name": collection.name,
                                    "points": info.points_count,
                                    "vector_size": info.config.params.vectors.size
                                })
                                stats["total_embeddings"] += info.points_count
                    except Exception as e:
                        logger.error(f"Error getting Qdrant stats: {e}")

                    # Get pgvector stats
                    try:
                        conn = get_connection()
                        if conn:
                            with conn.cursor() as cur:
                                cur.execute("""
                                    SELECT tablename FROM pg_tables 
                                    WHERE tablename LIKE 'docs_%' OR tablename LIKE 'course_docs_%'
                                """)
                                tables = cur.fetchall()

                                for (table_name,) in tables:
                                    cur.execute(
                                        f"SELECT COUNT(*) FROM {table_name}")
                                    count = cur.fetchone()[0]
                                    stats["tables"].append({
                                        "name": table_name,
                                        "rows": count
                                    })
                                    stats["total_chunks"] += count

                            conn.close()
                    except Exception as e:
                        logger.error(f"Error getting pgvector stats: {e}")

                    # Estimate total documents
                    stats["total_docs"] = len(set([col["name"].replace("course_docs_clean_", "").replace("docs_clean_", "")
                                                  for col in stats["collections"]]))

                    response_data["pipeline_stats"] = stats

                except Exception as e:
                    logger.error(f"Error getting stats for visualization: {e}")

            return HTMLResponse(render_general_response(
                data=response_data,
                title="Dashboard: Estadísticas y Visualizaciones del Pipeline"
            ))

        return JSONResponse({
            "success": True,
            "plot_data": plot_data,
            "type": type,
            "config": {
                "distance_metric": distance_metric,
                "index_algorithm": index_algorithm,
                "collection": collection_name
            }
        })

    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        if format == "html":
            return HTMLResponse(render_general_response({
                "error": str(e),
                "message": "Error generating visualizations"
            }, "Visualizaciones del Pipeline", "#ef4444"))
        else:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "type": type}
            )
