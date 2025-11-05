from fastapi import FastAPI, Query, HTTPException
from app.rag import rag_answer, generate_llm_answer

app = FastAPI(title="RAG Demo — Qdrant vs pgvector")


@app.get("/ask")
def ask(q: str = Query(..., description="Pregunta"), backend: str = "qdrant", k: int = 5):
    try:
        return rag_answer(q, backend=backend, k=k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/ai")
def ai(q: str = Query(..., description="Pregunta"), backend: str = "qdrant", k: int = 5, model: str = "phi3:mini"):
    """AI-powered RAG: retrieval + LLM generation using Ollama"""
    try:
        return generate_llm_answer(q, backend=backend, k=k, model=model)
    except ImportError as e:
        raise HTTPException(
            status_code=503, detail=f"LLM service not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


@app.get("/compare")
def compare(q: str = Query(..., description="Pregunta"), k: int = 3):
    """Compare results from both Qdrant and pgvector backends"""
    try:
        qdrant_results = rag_answer(q, backend="qdrant", k=k)
        pgvector_results = rag_answer(q, backend="pgvector", k=k)

        return {
            "query": q,
            "comparison": {
                "qdrant": {
                    "total_results": qdrant_results["total_results"],
                    "top_similarity": qdrant_results["results"][0]["similarity"] if qdrant_results["results"] else "0.000",
                    "results": qdrant_results["results"]
                },
                "pgvector": {
                    "total_results": pgvector_results["total_results"],
                    "top_similarity": pgvector_results["results"][0]["similarity"] if pgvector_results["results"] else "0.000",
                    "results": pgvector_results["results"]
                }
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Comparison error: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "🚀 RAG Demo API - Qdrant vs pgvector + AI",
        "endpoints": {
            "/ask": "Single backend RAG search",
            "/ai": "AI-powered RAG with LLM generation (requires Ollama)",
            "/compare": "Compare both backends side-by-side",
            "/docs": "API documentation"
        },
        "available_backends": ["qdrant", "pgvector"],
        "available_models": ["phi3:mini", "llama2", "mistral"],
        "examples": {
            "single_search": "/ask?q=vectores&backend=qdrant&k=3",
            "ai_search": "/ai?q=¿Qué son las bases de datos vectoriales?&backend=qdrant&k=3&model=phi3:mini",
            "comparison": "/compare?q=vectores&k=3"
        }
    }
