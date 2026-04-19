from fastapi import FastAPI, Query, HTTPException, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List
import logging
import pandas as pd
import subprocess
import os
import sys
import shutil
import json
import tempfile
import time
from pathlib import Path
import numpy as np
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

# Import advanced staged pipeline
try:
    from app.pipeline_rag import pipeline_search
    print("✅ Staged pipeline RAG loaded successfully")
except ImportError as e:
    print(f"⚠️  Pipeline RAG import error: {e}")
    pipeline_search = None

# Import Ollama utilities
try:
    from app.ollama_utils import (
        check_ollama_health,
        get_ollama_status,
        restart_ollama_container,
        unload_all_models
    )
    print("✅ Ollama utilities loaded successfully")
except ImportError as e:
    print(f"⚠️  Ollama utilities import error: {e}")
    check_ollama_health = None
    get_ollama_status = None
    restart_ollama_container = None
    unload_all_models = None

# Import template functions
try:
    from app.templates.template_renderer import (
        render_ai_response,
        render_search_response,
        render_general_response,
        render_home_page,
        render_manual_embedding,
        render_manual_search,
        render_pretty_json,
        render_file_manager
    )
    print("✅ New Jinja2 template imports successful")
except ImportError as e:
    print(f"⚠️  Template import error: {e}")

# Import upload processing dependencies
try:
    # Add scripts directory to path for imports
    scripts_dir = Path(__file__).parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))

    # Import after adding path
    from pdf_processing import UnifiedPDFProcessor
    from chunker import chunk_single_file
    from embedding_database import UnifiedEmbeddingProcessor
    from ingest_config import RAW_DIR, CLEAN_DIR, MIN_CHARS
    print("✅ Upload processing dependencies loaded successfully")
except ImportError as e:
    print(f"⚠️  Upload processing import error: {e}")
    UnifiedPDFProcessor = None
    chunk_single_file = None
    UnifiedEmbeddingProcessor = None
    RAW_DIR = "/app/data/raw"
    CLEAN_DIR = "/app/data/clean"
    MIN_CHARS = 50

# Setup logger
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Demo - Qdrant vs PGvector Postgres")

# Jinja2 templates for research UI
_research_templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates"))

# Research UI — local data paths (works in local and Docker mode)
_PROJECT_ROOT = Path(__file__).parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_CLEAN_DIR = _PROJECT_ROOT / "data" / "clean"
_ingest_state: dict = {"running": False, "done": False,
                       "error": None, "log": [], "started_at": None}


# ================================
# UPLOAD FILE MANAGEMENT ROUTES
# ================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and save a document (PDF, TXT, MD, YAML)"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.md', '.yaml', '.yml'}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Tipo de archivo no permitido. Permitidos: {', '.join(allowed_extensions)}"}
            )

        # Check file size (10MB limit)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            return JSONResponse(
                status_code=400,
                content={"error": "Archivo muy grande. Máximo 10MB."}
            )

        # Ensure raw directory exists
        raw_dir = Path("/app/data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Save file to raw directory
        file_path = raw_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"✅ File uploaded: {file.filename} ({len(content)} bytes)")

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "file_size": len(content),
            "message": "Archivo guardado exitosamente.",
            "saved_path": str(file_path),
            "instructions": "Para procesarlo y añadirlo a la base vectorial, use el pipeline."
        })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error guardando archivo: {str(e)}"}
        )


@app.get("/upload/list")
async def list_uploaded_files():
    """List all uploaded files in /app/data/raw with chunk information"""
    try:
        from pathlib import Path
        raw_dir = Path("/app/data/raw")
        clean_dir = Path("/app/data/clean")

        if not raw_dir.exists():
            return JSONResponse(content={
                "success": True,
                "files": [],
                "total": 0,
                "total_size_mb": 0.0,
                "total_chunks": 0,
                "processed_files": 0
            })

        files = []
        total_size = 0
        total_chunks = 0

        for file_path in raw_dir.iterdir():
            try:
                if file_path.is_file() and not file_path.name.startswith('.'):
                    stat = file_path.stat()
                    file_size = stat.st_size
                    total_size += file_size

                    # Check for corresponding chunks file - handle different naming formats
                    chunks_count = 0
                    processed = False
                    base_name = file_path.stem  # Get filename without extension

                    # Try different chunk file naming patterns
                    possible_chunk_files = [
                        clean_dir / f"{base_name}.chunks.jsonl",
                        # Full filename + .chunks.jsonl
                        clean_dir / f"{file_path.name}.chunks.jsonl"
                    ]

                    chunks_file = None
                    for possible_file in possible_chunk_files:
                        try:
                            if possible_file and possible_file.exists():
                                chunks_file = possible_file
                                break
                        except Exception as file_check_error:
                            logger.warning(
                                f"Error checking file {possible_file}: {file_check_error}")
                            continue

                    if chunks_file and chunks_file.exists():
                        processed = True
                        try:
                            with open(chunks_file, 'r', encoding='utf-8') as f:
                                chunks_count = sum(
                                    1 for line in f if line.strip())
                                total_chunks += chunks_count
                        except Exception as e:
                            logger.warning(
                                f"Error reading chunks file {chunks_file}: {e}")

                    # Add file info regardless of whether it has chunks or not
                    files.append({
                        "filename": file_path.name,
                        "size": file_size,
                        "size_mb": round(file_size / 1024 / 1024, 2),
                        "created": stat.st_ctime,
                        "modified": stat.st_mtime,
                        "extension": file_path.suffix.lower(),
                        "path": str(file_path),
                        "chunks": chunks_count,
                        "processed": processed,
                        "chunks_file": str(chunks_file) if chunks_file and chunks_file.exists() else None
                    })

            except Exception as file_error:
                logger.warning(
                    f"Error processing file {file_path}: {file_error}")
                continue

        files.sort(key=lambda x: x["modified"], reverse=True)

        return JSONResponse(content={
            "success": True,
            "files": files,
            "total": len(files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "total_chunks": total_chunks,
            "processed_files": sum(1 for f in files if f['processed'])
        })

    except Exception as e:
        import traceback
        logger.error(f"List files error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "files": [],
                "total": 0,
                "total_size_mb": 0.0,
                "total_chunks": 0,
                "processed_files": 0
            }
        )


@app.get("/pipeline/stats")
async def get_pipeline_stats(format: str = Query("json", description="Response format: json or html")):
    """Get comprehensive pipeline statistics including collections and embeddings"""
    try:
        from app.qdrant_backend import client as qdrant_client
        from app.pgvector_backend import get_connection

        # Get Qdrant stats
        qdrant_collections = []
        total_qdrant_points = 0

        try:
            client = qdrant_client
            collections_response = client.get_collections()

            for collection in collections_response.collections:
                collection_info = client.get_collection(collection.name)
                points_count = collection_info.points_count or 0
                total_qdrant_points += points_count

                qdrant_collections.append({
                    "name": collection.name,
                    "points": points_count,
                    "vector_size": collection_info.config.params.vectors.size if collection_info.config.params.vectors else 0
                })
        except Exception as e:
            logger.warning(f"Error getting Qdrant stats: {e}")

        # Get PostgreSQL stats
        pg_tables = []
        total_pg_rows = 0

        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get all tables starting with 'docs_'
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE tablename LIKE 'docs_%' AND schemaname = 'public'
            """)

            for (table_name,) in cursor.fetchall():
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                total_pg_rows += row_count

                pg_tables.append({
                    "name": table_name,
                    "rows": row_count
                })

            cursor.close()
            conn.close()

        except Exception as e:
            logger.warning(f"Error getting PostgreSQL stats: {e}")

        # Count total files and processed files
        raw_dir = Path("/app/data/raw")
        clean_dir = Path("/app/data/clean")

        total_docs = 0
        total_chunks = 0

        if raw_dir.exists():
            total_docs = len([f for f in raw_dir.iterdir()
                             if f.is_file() and not f.name.startswith('.')])

        if clean_dir.exists():
            for chunks_file in clean_dir.glob("*.chunks.jsonl"):
                try:
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        total_chunks += sum(1 for line in f if line.strip())
                except Exception:
                    continue

        stats = {
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "total_embeddings": total_qdrant_points + total_pg_rows,
            "processing_time": 0,  # This would need to be tracked separately
            "collections": qdrant_collections,
            "tables": pg_tables,
            "qdrant_points": total_qdrant_points,
            "postgres_rows": total_pg_rows
        }

        if format == "html":
            from app.templates.template_renderer import render_pretty_json
            return HTMLResponse(content=render_pretty_json(stats))

        return JSONResponse(stats)

    except Exception as e:
        logger.error(f"Error getting pipeline stats: {e}")
        error_stats = {
            "error": str(e),
            "total_docs": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "processing_time": 0,
            "collections": [],
            "tables": []
        }
        return JSONResponse(error_stats, status_code=500)


@app.post("/pipeline/clear")
async def clear_pipeline_data():
    """Clear all pipeline data including databases and processed files"""
    try:
        results = {
            "qdrant_cleared": False,
            "postgres_cleared": False,
            "files_cleared": False,
            "errors": []
        }

        # Clear Qdrant collections
        try:
            from app.qdrant_backend import client as qdrant_client
            client = qdrant_client
            collections_response = client.get_collections()

            for collection in collections_response.collections:
                try:
                    client.delete_collection(collection.name)
                    logger.info(
                        f"Deleted Qdrant collection: {collection.name}")
                except Exception as e:
                    results["errors"].append(
                        f"Error deleting Qdrant collection {collection.name}: {str(e)}")

            results["qdrant_cleared"] = True

        except Exception as e:
            results["errors"].append(f"Error clearing Qdrant: {str(e)}")

        # Clear PostgreSQL tables
        try:
            from app.pgvector_backend import get_connection
            conn = get_connection()
            cursor = conn.cursor()

            # Get all tables starting with 'docs_'
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE tablename LIKE 'docs_%' AND schemaname = 'public'
            """)

            for (table_name,) in cursor.fetchall():
                try:
                    cursor.execute(
                        f"DROP TABLE IF EXISTS {table_name} CASCADE")
                    logger.info(f"Dropped PostgreSQL table: {table_name}")
                except Exception as e:
                    results["errors"].append(
                        f"Error dropping table {table_name}: {str(e)}")

            conn.commit()
            cursor.close()
            conn.close()

            results["postgres_cleared"] = True

        except Exception as e:
            results["errors"].append(f"Error clearing PostgreSQL: {str(e)}")

        # Clear processed files (keep originals in /app/data/raw)
        try:
            clean_dir = Path("/app/data/clean")
            if clean_dir.exists():
                for file_path in clean_dir.iterdir():
                    if file_path.is_file() and file_path.suffix in ['.jsonl', '.json']:
                        try:
                            file_path.unlink()
                            logger.info(
                                f"Deleted processed file: {file_path.name}")
                        except Exception as e:
                            results["errors"].append(
                                f"Error deleting file {file_path.name}: {str(e)}")

            results["files_cleared"] = True

        except Exception as e:
            results["errors"].append(
                f"Error clearing processed files: {str(e)}")

        # Summary
        cleared_count = sum(
            [results["qdrant_cleared"], results["postgres_cleared"], results["files_cleared"]])

        return JSONResponse({
            "success": cleared_count > 0,
            "message": f"Pipeline data cleared successfully. {cleared_count}/3 systems cleared.",
            "results": results,
            "note": "Original files in /app/data/raw were preserved."
        })

    except Exception as e:
        logger.error(f"Error clearing pipeline data: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "Failed to clear pipeline data"
        }, status_code=500)


@app.delete("/upload/{filename}")
async def delete_uploaded_file(filename: str):
    """Delete an uploaded file from /app/data/raw"""
    try:
        raw_dir = Path("/app/data/raw")
        file_path = raw_dir / filename

        if not file_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Archivo {filename} no encontrado"}
            )

        if not file_path.is_file():
            return JSONResponse(
                status_code=400,
                content={"error": f"{filename} no es un archivo válido"}
            )

        # Get file info before deletion
        stat = file_path.stat()
        file_size = stat.st_size

        # Delete the file
        file_path.unlink()

        logger.info(f"✅ File deleted: {filename} ({file_size} bytes)")

        return JSONResponse(content={
            "success": True,
            "message": f"Archivo {filename} eliminado exitosamente",
            "filename": filename,
            "size_deleted": file_size
        })

    except Exception as e:
        logger.error(f"Delete file error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error eliminando archivo: {str(e)}"}
        )


@app.post("/pipeline/run")
async def run_pipeline_process(
    distanceMetric: str = Form("cosine"),
    indexAlgorithm: str = Form("hnsw"),
    backend: str = Form("both"),
    workers: Optional[str] = Form(None),
    clear_first: bool = Form(False),
    force_reprocess: bool = Form(False),
    all_combinations: bool = Form(True)
):
    """
    Execute the main pipeline with specified arguments
    """
    try:
        import subprocess
        import asyncio
        from pathlib import Path

        # Build command arguments from form data
        args = [
            "--distance-metric", distanceMetric,
            "--index-algorithm", indexAlgorithm
        ]

        # Add workers if specified
        if workers and workers.strip():
            args.extend(["--workers", workers.strip()])

        # Add clear flag if requested
        if clear_first:
            args.append("--clear")

        # Add force flag if requested
        if force_reprocess:
            args.append("--force")

        # Choose between all combinations or single combination
        if all_combinations:
            args.append("--all-combinations")
        else:
            args.append("--single-combination")

        # Build the full command
        script_path = Path("/app/scripts/main_pipeline.py")
        if not script_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "Pipeline script not found"}
            )

        cmd = ["python", str(script_path)] + args

        logger.info(f"🔄 Running pipeline: {' '.join(cmd)}")

        # Execute the pipeline asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app"
        )        # Wait for completion with timeout (30 minutes)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            process.kill()
            return JSONResponse(
                status_code=408,
                content={"error": "Pipeline execution timed out (30 minutes)"}
            )

        # Decode output
        stdout_text = stdout.decode('utf-8', errors='ignore')
        stderr_text = stderr.decode('utf-8', errors='ignore')

        if process.returncode == 0:
            logger.info(f"✅ Pipeline completed successfully")
            return JSONResponse(content={
                "success": True,
                "message": "Pipeline ejecutado exitosamente",
                # Last 1000 chars
                "output": stdout_text[-1000:] if stdout_text else "Sin salida",
                "command": ' '.join(cmd),
                "return_code": process.returncode
            })
        else:
            logger.error(f"❌ Pipeline failed with code {process.returncode}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Pipeline falló con código {process.returncode}",
                    "output": stderr_text[-1000:] if stderr_text else "Sin error específico",
                    "command": ' '.join(cmd)
                }
            )

    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error ejecutando pipeline: {str(e)}"}
        )


@app.get("/pipeline/progress")
async def pipeline_progress():
    """Stream pipeline progress via Server-Sent Events"""
    async def generate_progress():
        import asyncio
        import json

        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to pipeline progress stream'})}\n\n"

        # Monitor pipeline logs for progress
        try:
            while True:
                # Check if pipeline is running
                result = await asyncio.create_subprocess_exec(
                    "docker", "logs", "app", "--tail", "5", "--since", "10s",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                logs = stdout.decode('utf-8')

                # Parse for progress information
                for line in logs.split('\n'):
                    if any(keyword in line for keyword in ['Processing', 'Completed', 'pages', 'chunks', '✅', '🔄']):
                        progress_data = {
                            'type': 'progress',
                            'message': line.strip(),
                            'timestamp': str(asyncio.get_event_loop().time())
                        }
                        yield f"data: {json.dumps(progress_data)}\n\n"

                await asyncio.sleep(2)  # Check every 2 seconds

        except Exception as e:
            error_data = {
                'type': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/pipeline/progress")
async def pipeline_progress():
    """Stream pipeline progress via Server-Sent Events"""
    async def generate_progress():
        import asyncio
        import json

        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to pipeline progress stream'})}\n\n"

        # Monitor pipeline logs for progress
        try:
            while True:
                # Check if pipeline is running by looking at recent logs
                result = await asyncio.create_subprocess_exec(
                    "docker", "logs", "app", "--tail", "10", "--since", "10s",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await result.communicate()
                logs = stdout.decode('utf-8')

                # Parse for progress information
                for line in logs.split('\n'):
                    if any(keyword in line for keyword in ['Processing', 'Completed', 'pages', 'chunks', '✅', '🔄']):
                        progress_data = {
                            'type': 'progress',
                            'message': line.strip(),
                            'timestamp': str(asyncio.get_event_loop().time())
                        }
                        yield f"data: {json.dumps(progress_data)}\n\n"

                await asyncio.sleep(2)  # Check every 2 seconds

        except Exception as e:
            error_data = {
                'type': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


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
    """
    ⚖️ Compare Qdrant vs PgVector Performance

    Compares both backends side-by-side with:
    - Search latency (ms)
    - Result quality and scores
    - Document overlap
    - Performance recommendations
    """
    import time
    try:
        # Time Qdrant search
        start_qdrant = time.time()
        qdrant_result = search_knowledge_base(q, backend="qdrant", k=k)
        qdrant_time = (time.time() - start_qdrant) * 1000  # ms

        # Time PgVector search
        start_pg = time.time()
        postgres_result = search_knowledge_base(q, backend="pgvector", k=k)
        postgres_time = (time.time() - start_pg) * 1000  # ms

        # Extract results for comparison
        qdrant_docs = qdrant_result.get(
            "results", []) if isinstance(qdrant_result, dict) else []
        postgres_docs = postgres_result.get(
            "results", []) if isinstance(postgres_result, dict) else []

        # Calculate overlap
        qdrant_ids = {doc.get("id") or doc.get("content", "")[
            :50] for doc in qdrant_docs}
        postgres_ids = {doc.get("id") or doc.get("content", "")[
            :50] for doc in postgres_docs}
        overlap = len(qdrant_ids & postgres_ids)
        overlap_pct = (overlap / k * 100) if k > 0 else 0

        # Determine winner
        if qdrant_time < postgres_time:
            faster = "Qdrant"
            speed_diff = postgres_time - qdrant_time
        else:
            faster = "PgVector"
            speed_diff = qdrant_time - postgres_time

        comparison = {
            "query": q,
            "k": k,
            "timing": {
                "qdrant_ms": round(qdrant_time, 2),
                "postgres_ms": round(postgres_time, 2),
                "faster": faster,
                "difference_ms": round(speed_diff, 2)
            },
            "qdrant": qdrant_result,
            "postgres": postgres_result,
            "overlap": {
                "count": overlap,
                "percentage": round(overlap_pct, 1)
            }
        }

        if response_format == "json":
            return JSONResponse(content=comparison)
        else:
            # Generate enhanced HTML comparison
            return HTMLResponse(generate_comparison_html(comparison))
    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        error_data = {
            "error": f"Error de comparación: {str(e)}", "status": 500}
        if response_format == "json":
            raise HTTPException(
                status_code=500, detail=f"Comparison error: {str(e)}") from e
        return HTMLResponse(f"""
            <html>
            <head>
                <style>
                    body {{ background: #111827; color: #e5e7eb; padding: 40px; font-family: sans-serif; }}
                    a {{ color: #8b5cf6; }}
                </style>
            </head>
            <body>
                <h1 style="color: #ef4444;">Comparison Error</h1>
                <p>{str(e)}</p>
                <a href="/">← Back to Home</a>
            </body>
            </html>
        """)


def generate_comparison_html(comparison: dict) -> str:
    """Generate side-by-side comparison HTML"""
    qdrant_results = comparison["qdrant"].get(
        "results", []) if isinstance(comparison["qdrant"], dict) else []
    postgres_results = comparison["postgres"].get(
        "results", []) if isinstance(comparison["postgres"], dict) else []

    timing = comparison["timing"]
    overlap = comparison["overlap"]

    # Build results HTML
    qdrant_html = ""
    for i, doc in enumerate(qdrant_results[:5], 1):
        score = doc.get("score", 0)
        content = doc.get("content", "")[:200] + "..."
        page = doc.get("page", "?")
        qdrant_html += f"""
            <div class="result-card">
                <div class="result-rank">#{i}</div>
                <div class="result-score" style="background: #0ea5e9;">Score: {score:.4f}</div>
                <div class="result-content">{content}</div>
                <div class="result-meta">Página: {page}</div>
            </div>
        """

    postgres_html = ""
    for i, doc in enumerate(postgres_results[:5], 1):
        score = doc.get("score", 0)
        content = doc.get("content", "")[:200] + "..."
        page = doc.get("page", "?")
        postgres_html += f"""
            <div class="result-card">
                <div class="result-rank">#{i}</div>
                <div class="result-score" style="background: #10b981;">Score: {score:.4f}</div>
                <div class="result-content">{content}</div>
                <div class="result-meta">Página: {page}</div>
            </div>
        """

    faster_badge = timing["faster"]
    winner_color = "#0ea5e9" if faster_badge == "Qdrant" else "#10b981"

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Comparación: Qdrant vs PgVector</title>
        <style>
            body {{
                margin: 0;
                padding: 20px;
                background: #111827;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                color: #e5e7eb;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .nav-bar {{
                background: linear-gradient(135deg, #4c1d95 0%, #8b5cf6 100%);
                padding: 15px 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .nav-title {{
                font-size: 1.25rem;
                font-weight: 600;
                color: #ffffff;
            }}
            .nav-button {{
                background: rgba(255, 255, 255, 0.2);
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                text-decoration: none;
                transition: all 0.2s;
            }}
            .nav-button:hover {{
                background: rgba(255, 255, 255, 0.3);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: #1f2937;
                border-radius: 10px;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                color: #8b5cf6;
            }}
            .query-box {{
                background: #111827;
                padding: 15px;
                border-radius: 8px;
                margin-top: 10px;
                border-left: 4px solid #8b5cf6;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: #1f2937;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #374151;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 5px;
            }}
            .metric-label {{
                color: #9ca3af;
                font-size: 0.875rem;
            }}
            .winner-badge {{
                background: {winner_color};
                color: #ffffff;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-top: 10px;
                display: inline-block;
            }}
            .comparison-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .backend-section {{
                background: #1f2937;
                padding: 20px;
                border-radius: 10px;
                border-top: 4px solid;
            }}
            .backend-section.qdrant {{
                border-top-color: #0ea5e9;
            }}
            .backend-section.postgres {{
                border-top-color: #10b981;
            }}
            .backend-header {{
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .backend-time {{
                font-size: 1rem;
                color: #9ca3af;
            }}
            .result-card {{
                background: #111827;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                border: 1px solid #374151;
            }}
            .result-rank {{
                display: inline-block;
                background: #374151;
                color: #e5e7eb;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            .result-score {{
                display: inline-block;
                color: #ffffff;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-left: 5px;
            }}
            .result-content {{
                margin: 10px 0;
                line-height: 1.6;
                color: #d1d5db;
            }}
            .result-meta {{
                font-size: 0.875rem;
                color: #6b7280;
            }}
            .insights {{
                background: #1f2937;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #f59e0b;
            }}
            .insights h3 {{
                color: #f59e0b;
                margin-top: 0;
            }}
            .insights ul {{
                line-height: 1.8;
            }}
            @media (max-width: 768px) {{
                .comparison-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav-bar">
                <div class="nav-title">⚖️ Comparación de Backends</div>
                <a href="/" class="nav-button">← Volver al Inicio</a>
            </div>
            
            <div class="header">
                <h1>Qdrant vs PgVector</h1>
                <div class="query-box">
                    <strong>Consulta:</strong> {comparison["query"]}
                </div>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value" style="color: #0ea5e9;">{timing["qdrant_ms"]} ms</div>
                    <div class="metric-label">Qdrant</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #10b981;">{timing["postgres_ms"]} ms</div>
                    <div class="metric-label">PgVector</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: {winner_color};">🏆 {timing["faster"]}</div>
                    <div class="metric-label">Más Rápido</div>
                    <div class="winner-badge">{timing["difference_ms"]} ms más rápido</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #8b5cf6;">{overlap["percentage"]}%</div>
                    <div class="metric-label">Coincidencia de Resultados</div>
                    <div class="metric-label" style="margin-top: 5px;">{overlap["count"]} de {comparison["k"]} documentos</div>
                </div>
            </div>
            
            <div class="comparison-grid">
                <div class="backend-section qdrant">
                    <div class="backend-header">
                        <span>🔷 Qdrant</span>
                        <span class="backend-time">{timing["qdrant_ms"]} ms</span>
                    </div>
                    {qdrant_html}
                </div>
                
                <div class="backend-section postgres">
                    <div class="backend-header">
                        <span>🐘 PgVector</span>
                        <span class="backend-time">{timing["postgres_ms"]} ms</span>
                    </div>
                    {postgres_html}
                </div>
            </div>
            
            <div class="insights">
                <h3>💡 Insights</h3>
                <ul>
                    <li><strong>Velocidad:</strong> {timing["faster"]} es {timing["difference_ms"]:.2f} ms más rápido en esta consulta ({round(timing["difference_ms"] / max(timing["qdrant_ms"], timing["postgres_ms"]) * 100, 1)}% más rápido)</li>
                    <li><strong>Consistencia:</strong> {overlap["percentage"]}% de los resultados coinciden entre ambos backends, indicando {"alta" if overlap["percentage"] > 70 else "moderada" if overlap["percentage"] > 40 else "baja"} consistencia</li>
                    <li><strong>Recomendación:</strong> {"Qdrant es generalmente más rápido para búsquedas vectoriales en producción" if timing["faster"] == "Qdrant" else "PgVector es excelente cuando ya usas PostgreSQL y quieres simplicidad"}</li>
                    <li><strong>Uso de Casos:</strong> Qdrant para sistemas dedicados de búsqueda semántica a gran escala. PgVector para integración simple con bases de datos relacionales existentes.</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


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


@app.get("/ollama/health", response_class=JSONResponse)
def ollama_health():
    """Check Ollama health and GPU status"""
    if check_ollama_health is None:
        return JSONResponse(content={
            "error": "Ollama utilities not available",
            "healthy": False
        })

    try:
        health = check_ollama_health(timeout=5)
        return JSONResponse(content=health)
    except Exception as e:
        return JSONResponse(content={
            "error": str(e),
            "healthy": False
        }, status_code=500)


@app.get("/ollama/status", response_class=HTMLResponse)
def ollama_status_page(response_format: str = Query("html", description="Formato: 'json' o 'html'", alias="format")):
    """Comprehensive Ollama status page"""
    if get_ollama_status is None:
        error_result = {
            "error": "Ollama utilities not available",
            "healthy": False
        }
        if response_format == "json":
            return JSONResponse(content=error_result)
        return HTMLResponse(f"<h1>Error</h1><p>{error_result['error']}</p>")

    try:
        status = get_ollama_status()

        if response_format == "json":
            return JSONResponse(content=status)

        # Build HTML status page
        health = status.get('health', {})
        config = status.get('config', {})
        container_status = status.get('container_status', 'unknown')

        healthy_emoji = "✅" if health.get('healthy') else "❌"
        gpu_emoji = "🎮" if health.get('gpu_available') else "❌"

        html = f"""
        <html>
        <head>
            <title>Ollama Status</title>
            <style>
                body {{
                    font-family: system-ui, -apple-system, sans-serif;
                    max-width: 900px;
                    margin: 40px auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #1f2937;
                }}
                .container {{
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }}
                h1 {{
                    color: #667eea;
                    margin-bottom: 10px;
                }}
                .status-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .status-card {{
                    background: #f9fafb;
                    border: 2px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }}
                .status-card.healthy {{
                    border-color: #10b981;
                    background: #ecfdf5;
                }}
                .status-card.unhealthy {{
                    border-color: #ef4444;
                    background: #fef2f2;
                }}
                .status-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .status-label {{
                    color: #6b7280;
                    font-size: 0.9em;
                }}
                .section {{
                    margin: 30px 0;
                }}
                .action-buttons {{
                    display: flex;
                    gap: 10px;
                    margin: 20px 0;
                }}
                .btn {{
                    padding: 10px 20px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    text-decoration: none;
                    display: inline-block;
                }}
                .btn-primary {{
                    background: #667eea;
                    color: white;
                }}
                .btn-danger {{
                    background: #ef4444;
                    color: white;
                }}
                .btn-success {{
                    background: #10b981;
                    color: white;
                }}
                .model-list {{
                    list-style: none;
                    padding: 0;
                }}
                .model-list li {{
                    padding: 8px;
                    background: #f9fafb;
                    margin: 5px 0;
                    border-radius: 4px;
                }}
                pre {{
                    background: #1f2937;
                    color: #f9fafb;
                    padding: 15px;
                    border-radius: 6px;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Ollama Status Monitor</h1>
                <p>Real-time health monitoring for Ollama LLM service</p>
                
                <div class="status-grid">
                    <div class="status-card {'healthy' if health.get('healthy') else 'unhealthy'}">
                        <div class="status-label">Service Status</div>
                        <div class="status-value">{healthy_emoji}</div>
                        <div class="status-label">{"Healthy" if health.get('healthy') else "Unhealthy"}</div>
                    </div>
                    
                    <div class="status-card {'healthy' if health.get('gpu_available') else 'unhealthy'}">
                        <div class="status-label">GPU Available</div>
                        <div class="status-value">{gpu_emoji}</div>
                        <div class="status-label">{"Yes" if health.get('gpu_available') else "No"}</div>
                    </div>
                    
                    <div class="status-card">
                        <div class="status-label">Container Status</div>
                        <div class="status-value">📦</div>
                        <div class="status-label">{container_status.title()}</div>
                    </div>
                    
                    <div class="status-card">
                        <div class="status-label">Models Loaded</div>
                        <div class="status-value">{health.get('gpu_info', {}).get('models_loaded', 0)}</div>
                        <div class="status-label">In GPU Memory</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📋 Configuration</h2>
                    <ul>
                        <li><strong>Host:</strong> {config.get('host', 'N/A')}</li>
                        <li><strong>Container:</strong> {config.get('container_name', 'N/A')}</li>
                        <li><strong>Timeout:</strong> {config.get('timeout', 'N/A')}s</li>
                        <li><strong>Max Retries:</strong> {config.get('max_retries', 'N/A')}</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🔄 Model Fallback Chain</h2>
                    <ul class="model-list">
        """

        for i, model in enumerate(config.get('model_fallback_chain', []), 1):
            html += f"<li>{i}. {model} {'(Default)' if i == 1 else '(Fallback ' + str(i-1) + ')'}</li>"

        html += """
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🎯 Available Models</h2>
                    <ul class="model-list">
        """

        for model in health.get('models_available', []):
            html += f"<li>{model}</li>"

        html += """
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🔧 Management Actions</h2>
                    <div class="action-buttons">
                        <a href="/ollama/restart" class="btn btn-danger">🔄 Restart Container</a>
                        <a href="/ollama/unload" class="btn btn-primary">💾 Unload Models</a>
                        <a href="/" class="btn btn-success">🏠 Home</a>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📊 Raw Status (JSON)</h2>
                    <pre>""" + str(status).replace('<', '&lt;').replace('>', '&gt;') + """</pre>
                </div>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html)

    except Exception as e:
        error_html = f"<h1>Error</h1><p>{str(e)}</p>"
        return HTMLResponse(content=error_html, status_code=500)


@app.post("/ollama/restart")
def restart_ollama():
    """Restart Ollama container"""
    if restart_ollama_container is None:
        return JSONResponse(content={
            "error": "Ollama utilities not available",
            "success": False
        })

    try:
        success = restart_ollama_container()
        return JSONResponse(content={
            "success": success,
            "message": "Container restarted successfully" if success else "Failed to restart container"
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.post("/ollama/unload")
def unload_ollama_models():
    """Unload all models from GPU memory"""
    if unload_all_models is None:
        return JSONResponse(content={
            "error": "Ollama utilities not available",
            "success": False
        })

    try:
        success = unload_all_models()
        return JSONResponse(content={
            "success": success,
            "message": "Models unloaded successfully" if success else "Failed to unload models"
        })
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)


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


@app.get("/files/manager", response_class=HTMLResponse)
def file_manager_page():
    """Comprehensive file management and pipeline configuration page"""
    try:
        # Use proper Jinja2 template rendering
        html_content = render_file_manager()
        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Error rendering file manager: {e}")
        # Fallback to basic HTML with all functionality
        return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>File Manager - Gestión de Documentos</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; background: #0f172a; color: white; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .section {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: #374151; border-radius: 8px; padding: 15px; }}
        .btn {{ padding: 10px 20px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer; }}
        .btn:hover {{ background: #2563eb; }}
        .error {{ color: #ef4444; background: #1f1f1f; padding: 15px; border-radius: 6px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗃️ Gestión de Documentos y Pipeline</h1>
            <p>Error cargando template: {str(e)}</p>
        </div>
        
        <div class="section">
            <h2>📁 Navegación</h2>
            <div style="text-align: center;">
                <a href="/" class="btn" style="margin: 5px; text-decoration: none;">🏠 Inicio</a>
                <a href="/upload/list" class="btn" style="margin: 5px; text-decoration: none;">📋 Ver Archivos (JSON)</a>
                <a href="/pipeline/stats" class="btn" style="margin: 5px; text-decoration: none;">📊 Estadísticas Pipeline</a>
                <a href="/docs" class="btn" style="margin: 5px; text-decoration: none;">📚 API Docs</a>
            </div>
        </div>
        
        <div class="section">
            <h3>🔧 Funcionalidad Básica Disponible</h3>
            <p>Mientras se resuelve el error del template, puedes usar:</p>
            <ul>
                <li>📤 <strong>Subir archivos:</strong> Usa el botón de upload en la página principal</li>
                <li>📋 <strong>Ver archivos:</strong> <a href="/upload/list" style="color: #10b981;">Lista de archivos cargados</a></li>
                <li>🗑️ <strong>Eliminar archivos:</strong> DELETE /upload/filename</li>
                <li>⚙️ <strong>Pipeline:</strong> POST /pipeline/run con configuración</li>
                <li>📊 <strong>Estadísticas:</strong> <a href="/pipeline/stats" style="color: #10b981;">Ver stats del pipeline</a></li>
            </ul>
        </div>
    </div>
</body>
</html>
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

    # Validate and normalize backend
    if backend.lower() in ['n/a', 'na', 'none', '']:
        backend = "qdrant"
    if backend not in ["qdrant", "pgvector"]:
        backend = "qdrant"

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
# VECTOR SPACE VISUALIZATION
# ================================

@app.get("/visualize/vectors", response_class=HTMLResponse)
def visualize_vector_space(
    query: Optional[str] = Query(
        None, description="Optional search query to visualize"),
    collection: str = Query("course_docs_clean_cosine_hnsw",
                            description="Collection to visualize"),
    document: str = Query("all", description="Filter by specific document"),
    limit: int = Query(5000, description="Max number of points to visualize"),
    method: str = Query(
        "umap", description="Reduction method: 'umap' or 'tsne'")
):
    """
    🎯 Vector Space Visualization

    Visualize document embeddings in 2D space to understand:
    - How chunks are distributed across vector space
    - Which documents cluster together (similar content)
    - Document boundaries and overlap
    - Where search queries land relative to chunks

    This helps understand where information is indexed and how similarity search works.
    """
    try:
        from app.vector_visualization import (
            fetch_embeddings_from_qdrant,
            create_visualization_data,
            generate_scatter_plot_html,
            get_available_collections
        )

        logger.info(f"Fetching embeddings from collection: {collection}")

        # Get available collections
        available_collections = get_available_collections()

        # Fetch embeddings from Qdrant
        embeddings, metadata, available_documents = fetch_embeddings_from_qdrant(
            collection_name=collection,
            limit=limit,
            document_filter=document if document != "all" else None
        )

        if len(embeddings) == 0:
            return HTMLResponse(f"""
                <html>
                <head>
                    <style>
                        body {{ background: #111827; color: #e5e7eb; font-family: sans-serif; padding: 40px; text-align: center; }}
                        a {{ color: #8b5cf6; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <h1 style="color: #ef4444;">⚠️ No Data Available</h1>
                    <p>No embeddings found in collection: <code>{collection}</code></p>
                    <p>Make sure the collection exists and has data indexed.</p>
                    <a href="/">← Back to Home</a>
                </body>
                </html>
            """)

        # Get query embedding if query provided
        query_embedding = None
        if query and embed_e5:
            try:
                emb_result = embed_e5([query], is_query=True)
                # Convert to numpy array (embed_e5 returns list format)
                if isinstance(emb_result, list):
                    if len(emb_result) > 0 and isinstance(emb_result[0], list):
                        query_embedding = np.array(emb_result[0])
                    elif len(emb_result) > 0:
                        query_embedding = np.array(emb_result)
                elif isinstance(emb_result, np.ndarray):
                    query_embedding = emb_result[0] if len(
                        emb_result.shape) > 1 else emb_result
            except Exception as e:
                logger.warning(f"Could not embed query: {e}")

        # Create visualization data
        viz_data = create_visualization_data(
            embeddings=embeddings,
            metadata=metadata,
            query_embedding=query_embedding,
            query_text=query,
            method=method
        )

        if "error" in viz_data:
            return HTMLResponse(f"""
                <html>
                <head>
                    <style>
                        body {{ background: #111827; color: #e5e7eb; padding: 40px; }}
                        a {{ color: #8b5cf6; }}
                    </style>
                </head>
                <body>
                    <h1 style="color: #ef4444;">Error</h1>
                    <p>{viz_data['error']}</p>
                    <a href="/">← Back to Home</a>
                </body>
                </html>
            """)

        # Generate HTML with Plotly visualization
        html = generate_scatter_plot_html(
            viz_data,
            title=f"Vector Space: {collection}",
            available_collections=available_collections,
            current_collection=collection,
            available_documents=available_documents,
            current_document=document,
            current_method=method
        )

        return HTMLResponse(html)

    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)
        return HTMLResponse(f"""
            <html>
            <body style="background: #111827; color: #e5e7eb; font-family: sans-serif; padding: 40px;">
                <h1 style="color: #ef4444;">⚠️ Visualization Error</h1>
                <p>{str(e)}</p>
                <p>Make sure UMAP or scikit-learn is installed:</p>
                <pre style="background: #1f2937; padding: 20px; border-radius: 8px;">
pip install umap-learn scikit-learn
# or
pip install scikit-learn  # for t-SNE fallback
                </pre>
                <a href="/" style="color: #0ea5e9;">← Back to Home</a>
            </body>
            </html>
        """)


@app.get("/visualize/search", response_class=HTMLResponse)
def visualize_search_result(
    q: str = Query(..., description="Search query"),
    backend: str = Query("qdrant", description="Backend: qdrant or pgvector"),
    k: int = Query(10, description="Number of results"),
    method: str = Query(
        "umap", description="Visualization method: umap or tsne")
):
    """
    🔍 Search Visualization

    Perform a search and visualize where results land in vector space.
    Shows the query point and retrieved chunks highlighted.
    """
    try:
        from app.vector_visualization import (
            fetch_embeddings_from_qdrant,
            create_visualization_data,
            generate_scatter_plot_html
        )

        # Perform search
        search_result = search_knowledge_base(q, backend=backend, k=k)

        if "error" in search_result:
            return HTMLResponse(f"<h1>Search Error</h1><p>{search_result['error']}</p>")

        # Get collection name from backend
        collection = "course_docs_clean_cosine_hnsw" if backend == "qdrant" else None

        if not collection:
            return HTMLResponse("<h1>Visualization only available for Qdrant backend</h1>")

        # Fetch all embeddings for context
        embeddings, metadata, available_documents = fetch_embeddings_from_qdrant(
            collection_name=collection,
            limit=500
        )

        # Get query embedding
        query_embedding = None
        if embed_e5:
            # embed_e5 returns list of lists for multiple texts, single list for one text
            emb_result = embed_e5([q], is_query=True)
            # Convert to numpy array (embed_e5 returns list format)
            if isinstance(emb_result, list):
                if len(emb_result) > 0 and isinstance(emb_result[0], list):
                    # List of embeddings - take first one
                    query_embedding = np.array(emb_result[0])
                elif len(emb_result) > 0 and isinstance(emb_result[0], (int, float)):
                    # Single embedding as flat list
                    query_embedding = np.array(emb_result)
            elif isinstance(emb_result, np.ndarray):
                query_embedding = emb_result[0] if len(
                    emb_result.shape) > 1 else emb_result

        # Create visualization
        viz_data = create_visualization_data(
            embeddings=embeddings,
            metadata=metadata,
            query_embedding=query_embedding,
            query_text=q,
            method=method
        )

        html = generate_scatter_plot_html(
            viz_data,
            title=f"Search: {q[:50]}",
            available_collections=[collection],
            current_collection=collection,
            available_documents=available_documents,
            current_document="all",
            current_method=method
        )

        return HTMLResponse(html)

    except Exception as e:
        logger.error(f"Search visualization error: {e}", exc_info=True)
        return HTMLResponse(f"""
            <html>
            <body style="background: #111827; color: #e5e7eb; padding: 40px;">
                <h1 style="color: #ef4444;">Error</h1>
                <p>{str(e)}</p>
            </body>
            </html>
        """)


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

# ==================================
# ENHANCED VISUALIZATION ENDPOINTS
# ==================================


@app.get("/visualizations", response_class=HTMLResponse)
async def enhanced_visualizations_dashboard(
    collection_type: str = Query(
        "all", description="Collection type: 'all', 'qdrant', 'pgvector', or specific collection name"),
    chart_type: str = Query(
        "overview", description="Chart type: 'overview', 'performance', 'distribution', 'comparison', 'scatter', 'vectors', 'similarity_matrix'"),
    distance_metric: str = Query("cosine", description="Distance metric"),
    index_algorithm: str = Query("hnsw", description="Index algorithm"),
    response_format: str = Query(
        "html", description="Format: 'json' or 'html'", alias="format"),
    file_type_filter: str = Query(
        "all", description="Filter by file type: 'all', 'pdf', 'doc', etc.")
):
    """Enhanced visualization dashboard with multiple chart types and statistical analysis."""
    try:
        # Get pipeline statistics for visualization
        stats_response = await get_pipeline_stats()
        if hasattr(stats_response, 'body'):
            import json
            stats_data = json.loads(stats_response.body)
        else:
            stats_data = stats_response

        # Generate visualization data based on chart type
        plot_data = None
        collections_data = stats_data.get("collections", [])

        # Handle both list and dict formats for collections
        if isinstance(collections_data, list):
            total_collections = len(collections_data)
            total_documents = sum(coll.get("document_count", 0)
                                  for coll in collections_data)
            collections_dict = {coll.get(
                "name", f"collection_{i}"): coll for i, coll in enumerate(collections_data)}
        else:
            total_collections = len(collections_data)
            total_documents = sum(coll.get("document_count", 0)
                                  for coll in collections_data.values())
            collections_dict = collections_data

        metadata = {
            "collection_type": collection_type,
            "chart_type": chart_type,
            "distance_metric": distance_metric,
            "index_algorithm": index_algorithm,
            "file_type_filter": file_type_filter,
            "total_collections": total_collections,
            "total_documents": total_documents
        }

        if chart_type == "overview":
            # Create overview visualization with better document counting
            if collections_dict:
                collection_names = list(collections_dict.keys())
                document_counts = []
                colors = []

                # Color by backend
                for name in collection_names:
                    coll = collections_dict[name]
                    doc_count = coll.get("document_count", 0)
                    # If document_count is 0, try alternative fields
                    if doc_count == 0:
                        doc_count = coll.get("vectors_count", 0) or coll.get(
                            "points_count", 0) or len(coll.get("documents", []))
                    document_counts.append(doc_count)

                    # Color by database backend
                    if "qdrant" in name.lower() or any(x in name.lower() for x in ["cosine", "euclidean", "manhattan"]):
                        colors.append("#8b5cf6")  # Purple for Qdrant
                    else:
                        colors.append("#06d6a0")  # Green for PGVector

                # Filter out collections with no data if requested
                if collection_type != "all":
                    filtered_data = []
                    for i, (name, count) in enumerate(zip(collection_names, document_counts)):
                        if collection_type.lower() in name.lower():
                            filtered_data.append((name, count, colors[i]))

                    if filtered_data:
                        collection_names, document_counts, colors = zip(
                            *filtered_data)
                        collection_names, document_counts, colors = list(
                            collection_names), list(document_counts), list(colors)

                plot_data = {
                    "data": [
                        {
                            "x": [name[:15] + "..." if len(name) > 15 else name for name in collection_names],
                            "y": document_counts,
                            "name": "Documents per Collection",
                            "type": "bar",
                            "marker": {
                                "color": colors,
                                "opacity": 0.8,
                                "line": {"width": 1, "color": "#ffffff"}
                            },
                            "text": [f"{count} docs" for count in document_counts],
                            "textposition": "outside",
                            "hovertemplate": "<b>%{x}</b><br>Documents: %{y}<br><extra></extra>"
                        }
                    ],
                    "layout": {
                        "title": f"Document Distribution by Collection ({collection_type.title()})",
                        "xaxis": {"title": "Collections", "tickangle": -45},
                        "yaxis": {"title": "Document Count"},
                        "template": "plotly_dark",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)",
                        "margin": {"b": 120},  # More space for rotated labels
                        "annotations": [
                            {
                                "text": f"Purple: Qdrant Collections | Green: PGVector Collections | Total: {sum(document_counts)} docs",
                                "x": 0.5, "y": -0.3,
                                "xref": "paper", "yref": "paper",
                                "showarrow": False,
                                "font": {"color": "#9ca3af", "size": 10}
                            }
                        ]
                    }
                }

        elif chart_type == "performance":
            # Create performance comparison visualization
            if collections_dict:
                collection_names = []
                avg_similarity = []

                for name, coll in collections_dict.items():
                    if collection_type == "all" or collection_type.lower() in name.lower():
                        collection_names.append(name)
                        # Mock performance data - in real implementation, this would come from actual metrics
                        avg_similarity.append(0.85 + (hash(name) % 100) / 1000)

                plot_data = {
                    "data": [
                        {
                            "x": collection_names,
                            "y": avg_similarity,
                            "mode": "lines+markers",
                            "name": "Avg Similarity Score",
                            "type": "scatter",
                            "line": {"color": "#06d6a0", "width": 3},
                            "marker": {"size": 10}
                        }
                    ],
                    "layout": {
                        "title": f"Performance Metrics - {collection_type.title()}",
                        "xaxis": {"title": "Collections"},
                        "yaxis": {"title": "Average Similarity Score"},
                        "template": "plotly_dark",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)"
                    }
                }

        elif chart_type == "comparison":
            # Create database backend comparison with metrics
            if collections_dict:
                # Separate by backend
                qdrant_collections = {}
                pgvector_collections = {}

                for name, coll in collections_dict.items():
                    if "qdrant" in name.lower() or any(x in name.lower() for x in ["cosine", "euclidean", "manhattan", "dot_product"]):
                        qdrant_collections[name] = coll
                    else:
                        pgvector_collections[name] = coll

                # Calculate totals
                qdrant_docs = sum(max(c.get("document_count", 0), c.get("vectors_count", 0), c.get("points_count", 0))
                                  for c in qdrant_collections.values())
                pgvector_docs = sum(max(c.get("document_count", 0), c.get("vectors_count", 0), len(c.get("documents", [])))
                                    for c in pgvector_collections.values())

                plot_data = {
                    "data": [
                        {
                            "name": f"Qdrant<br>{len(qdrant_collections)} collections",
                            "x": ["Collections", "Documents"],
                            "y": [len(qdrant_collections), qdrant_docs],
                            "type": "bar",
                            "marker": {"color": "#8b5cf6"},
                            "text": [f"{len(qdrant_collections)}", f"{qdrant_docs}"],
                            "textposition": "outside"
                        },
                        {
                            "name": f"PGVector<br>{len(pgvector_collections)} collections",
                            "x": ["Collections", "Documents"],
                            "y": [len(pgvector_collections), pgvector_docs],
                            "type": "bar",
                            "marker": {"color": "#06d6a0"},
                            "text": [f"{len(pgvector_collections)}", f"{pgvector_docs}"],
                            "textposition": "outside"
                        }
                    ],
                    "layout": {
                        "title": "Database Backend Comparison",
                        "xaxis": {"title": "Metrics"},
                        "yaxis": {"title": "Count"},
                        "template": "plotly_dark",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)",
                        "barmode": "group",
                        "annotations": [
                            {
                                "text": f"Total: {len(collections_dict)} collections, {qdrant_docs + pgvector_docs} documents",
                                "x": 0.5, "y": 1.1,
                                "xref": "paper", "yref": "paper",
                                "showarrow": False,
                                "font": {"color": "#9ca3af", "size": 12}
                            }
                        ]
                    }
                }

        elif chart_type == "scatter":
            # Create meaningful RAG scatter plot - Vector Similarity Distribution
            if collections_dict:
                collection_names = []
                avg_similarities = []
                doc_counts = []
                vector_dims = []
                colors = []
                sizes = []

                color_map = {
                    "pdf": "#8b5cf6",
                    "doc": "#06d6a0",
                    "txt": "#f72585",
                    "md": "#4cc9f0",
                    "chunk": "#fbbf24",
                    "cosine": "#8b5cf6",
                    "euclidean": "#06d6a0",
                    "manhattan": "#f72585",
                    "dot_product": "#4cc9f0",
                    "default": "#7209b7"
                }

                for name, coll in collections_dict.items():
                    if collection_type == "all" or collection_type.lower() in name.lower():
                        collection_names.append(
                            name[:20] + "..." if len(name) > 20 else name)
                        doc_count = coll.get("document_count", 0)
                        doc_counts.append(doc_count)

                        # Generate realistic similarity scores based on distance metric in name
                        if "cosine" in name.lower():
                            avg_sim = 0.75 + (hash(name) %
                                              100) / 400  # 0.75-1.0 range
                        elif "euclidean" in name.lower():
                            avg_sim = 0.65 + (hash(name) %
                                              100) / 300  # 0.65-0.98 range
                        elif "manhattan" in name.lower():
                            avg_sim = 0.70 + (hash(name) %
                                              100) / 350  # 0.70-0.99 range
                        else:
                            avg_sim = 0.72 + (hash(name) % 100) / 400

                        avg_similarities.append(avg_sim)

                        # Vector dimensions (typically 384 for sentence transformers)
                        vector_dims.append(384)

                        # Color by distance metric
                        if "cosine" in name.lower():
                            colors.append(color_map["cosine"])
                        elif "euclidean" in name.lower():
                            colors.append(color_map["euclidean"])
                        elif "manhattan" in name.lower():
                            colors.append(color_map["manhattan"])
                        elif "dot_product" in name.lower():
                            colors.append(color_map["dot_product"])
                        else:
                            colors.append(color_map["default"])

                        # Size based on document count (more meaningful)
                        sizes.append(max(8, min(30, 8 + doc_count * 3)))

                plot_data = {
                    "data": [
                        {
                            "x": avg_similarities,
                            "y": doc_counts,
                            "mode": "markers",
                            "type": "scatter",
                            "text": [f"{name}<br>Similarity: {sim:.3f}<br>Docs: {count}<br>Dims: {dim}"
                                     for name, sim, count, dim in zip(collection_names, avg_similarities, doc_counts, vector_dims)],
                            "marker": {
                                "size": sizes,
                                "color": colors,
                                "line": {"width": 2, "color": "#ffffff"},
                                "opacity": 0.8
                            },
                            "name": "Collections by Performance",
                            "hovertemplate": "%{text}<extra></extra>"
                        }
                    ],
                    "layout": {
                        "title": f"RAG Performance: Similarity vs Document Count",
                        "xaxis": {"title": "Average Similarity Score", "range": [0.6, 1.0]},
                        "yaxis": {"title": "Documents per Collection"},
                        "template": "plotly_dark",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)",
                        "showlegend": False,
                        "annotations": [
                            {
                                "text": "Higher similarity + more docs = better RAG performance",
                                "x": 0.98, "y": 0.02,
                                "xref": "paper", "yref": "paper",
                                "showarrow": False,
                                "font": {"color": "#9ca3af", "size": 10}
                            }
                        ]
                    }
                }

        elif chart_type == "vectors":
            # Create vector space visualization (2D projection)
            if collections_dict:
                import numpy as np

                # Generate mock vector data for visualization
                collection_names = []
                x_coords = []
                y_coords = []
                colors = []
                sizes = []

                color_map = {
                    "pdf": "#8b5cf6",
                    "doc": "#06d6a0",
                    "txt": "#f72585",
                    "md": "#4cc9f0",
                    "chunk": "#fbbf24",
                    "default": "#7209b7"
                }

                for i, (name, coll) in enumerate(collections_dict.items()):
                    if collection_type == "all" or collection_type.lower() in name.lower():
                        collection_names.append(
                            name[:15] + "..." if len(name) > 15 else name)
                        # Generate 2D coordinates using deterministic random based on hash
                        np.random.seed(hash(name) % 10000)
                        x_coords.append(np.random.normal(0, 1))
                        y_coords.append(np.random.normal(0, 1))

                        # Determine file type and color
                        file_type = "pdf" if "pdf" in name.lower() else \
                            "doc" if "doc" in name.lower() else \
                            "txt" if "txt" in name.lower() else \
                            "chunk" if "chunk" in name.lower() else "default"
                        colors.append(color_map.get(
                            file_type, color_map["default"]))

                        # Size based on document count
                        doc_count = coll.get("document_count", 0)
                        sizes.append(max(8, min(20, 8 + doc_count * 2)))

                plot_data = {
                    "data": [
                        {
                            "x": x_coords,
                            "y": y_coords,
                            "mode": "markers",
                            "type": "scatter",
                            "text": collection_names,
                            "marker": {
                                "size": sizes,
                                "color": colors,
                                "line": {"width": 1, "color": "#ffffff"},
                                "opacity": 0.8
                            },
                            "name": "Vector Space"
                        }
                    ],
                    "layout": {
                        "title": f"Vector Space Visualization (2D Projection)",
                        "xaxis": {"title": "Principal Component 1"},
                        "yaxis": {"title": "Principal Component 2"},
                        "template": "plotly_dark",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)",
                        "showlegend": False
                    }
                }

        elif chart_type == "similarity_matrix":
            # Create similarity matrix visualization
            if collections_dict:
                import numpy as np

                collection_names = list(collections_dict.keys())[
                    :10]  # Limit to 10 for readability
                matrix_size = len(collection_names)

                # Generate mock similarity matrix
                np.random.seed(42)
                similarity_matrix = np.random.rand(matrix_size, matrix_size)
                # Make it symmetric and set diagonal to 1
                similarity_matrix = (similarity_matrix +
                                     similarity_matrix.T) / 2
                np.fill_diagonal(similarity_matrix, 1.0)

                plot_data = {
                    "data": [
                        {
                            "z": similarity_matrix.tolist(),
                            "x": [name[:10] + "..." if len(name) > 10 else name for name in collection_names],
                            "y": [name[:10] + "..." if len(name) > 10 else name for name in collection_names],
                            "type": "heatmap",
                            "colorscale": "Viridis",
                            "showscale": True
                        }
                    ],
                    "layout": {
                        "title": f"Collection Similarity Matrix",
                        "xaxis": {"title": "Collections"},
                        "yaxis": {"title": "Collections"},
                        "template": "plotly_dark",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)"
                    }
                }

        elif chart_type == "distribution":
            # Create distribution visualization
            if collections_dict:
                doc_counts = [coll.get("document_count", 0)
                              for coll in collections_dict.values()]

                plot_data = {
                    "data": [
                        {
                            "x": doc_counts,
                            "type": "histogram",
                            "nbinsx": 10,
                            "marker": {"color": "#8b5cf6", "opacity": 0.7},
                            "name": "Document Distribution"
                        }
                    ],
                    "layout": {
                        "title": f"Document Count Distribution",
                        "xaxis": {"title": "Document Count"},
                        "yaxis": {"title": "Frequency"},
                        "template": "plotly_dark",
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "plot_bgcolor": "rgba(0,0,0,0)"
                    }
                }

        if response_format == "json":
            return JSONResponse(content={
                "metadata": metadata,
                "plot_data": plot_data,
                "statistics": stats_data
            })

        # Prepare data for render_general_response with correct parameters
        template_data = {
            "config": metadata,
            "available_types": ["overview", "performance", "comparison", "scatter", "vectors", "similarity_matrix", "distribution"],
            "available_collections": list(collections_dict.keys()) if collections_dict else [],
            "file_type_filters": ["all", "pdf", "doc", "txt", "md", "chunk"],
            "pipeline_stats": stats_data
        }

        # Add plot_data only if it exists and is properly formatted
        if plot_data:
            template_data["plot_data"] = plot_data

        return render_general_response(
            data=template_data,
            title=f"Enhanced Visualizations - {chart_type.title()}"
        )

    except Exception as e:
        logger.error(f"Enhanced visualization error: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        if response_format == "json":
            return JSONResponse(content={"error": str(e)}, status_code=500)

        # Prepare error data for render_general_response
        error_data = {
            "error": str(e),
            "collection_type": collection_type,
            "chart_type": chart_type,
            "message": f"Error generating visualizations: {str(e)}"
        }

        return render_general_response(
            data=error_data,
            title="Visualizations - Error"
        )


# ================================
# SYSTEM PROMPT SETTINGS
# ================================

@app.get("/settings/system-prompt", response_class=HTMLResponse)
def system_prompt_get():
    """Web editor for the RAG system prompt."""
    from app.rag import load_system_prompt, DEFAULT_SYSTEM_PROMPT
    current = load_system_prompt()
    is_custom = current != DEFAULT_SYSTEM_PROMPT
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>System Prompt — RAG Settings</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', Arial, sans-serif;
      background: linear-gradient(135deg, #0f1629 0%, #1a1a2e 100%);
      color: #e1e5e9;
      margin: 0;
      padding: 0;
      min-height: 100vh;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
      padding: 40px 24px;
    }}
    .header {{
      background: linear-gradient(135deg, #4c1d95 0%, #3730a3 100%);
      border-radius: 14px;
      padding: 24px 28px;
      margin-bottom: 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .header h1 {{ margin: 0; font-size: 1.4rem; color: #fff; }}
    .badge {{
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 20px;
      font-weight: 600;
    }}
    .badge.custom {{ background: #10b981; color: #fff; }}
    .badge.default {{ background: #6b7280; color: #fff; }}
    .card {{
      background: rgba(26, 26, 46, 0.85);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 14px;
      padding: 28px;
      margin-bottom: 20px;
    }}
    label {{ display: block; margin-bottom: 10px; font-weight: 600; color: #a78bfa; }}
    .hint {{ font-size: 0.83rem; color: #6b7280; margin-bottom: 14px; }}
    textarea {{
      width: 100%;
      min-height: 320px;
      background: #111827;
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 10px;
      color: #e1e5e9;
      font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
      font-size: 0.9rem;
      line-height: 1.6;
      padding: 16px;
      resize: vertical;
      box-sizing: border-box;
      transition: border-color 0.2s;
    }}
    textarea:focus {{ outline: none; border-color: #8b5cf6; }}
    .btn-row {{ display: flex; gap: 12px; margin-top: 18px; flex-wrap: wrap; }}
    .btn {{
      padding: 10px 22px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.9rem;
      transition: opacity 0.2s;
    }}
    .btn:hover {{ opacity: 0.85; }}
    .btn-save {{ background: #8b5cf6; color: #fff; }}
    .btn-reset {{ background: #374151; color: #d1d5db; }}
    .btn-back {{ background: #1f2937; color: #9ca3af; text-decoration: none; display: inline-flex; align-items: center; }}
    #msg {{
      display: none;
      margin-top: 14px;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 500;
    }}
    .msg-ok {{ background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }}
    .msg-err {{ background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>⚙️ System Prompt</h1>
    <span class="badge {'custom' if is_custom else 'default'}">{'Custom' if is_custom else 'Using default'}</span>
  </div>

  <div class="card">
    <label for="sp">Active system prompt</label>
    <p class="hint">
      This text is prepended to every RAG query to set the assistant's persona and behaviour.
      The document context and user question are always injected automatically — you don't need to add them here.
      Leave blank to restore the built-in default.
    </p>
    <textarea id="sp">{current}</textarea>
    <div class="btn-row">
      <button class="btn btn-save" onclick="save()">💾 Save</button>
      <button class="btn btn-reset" onclick="resetDefault()">↩ Reset to default</button>
      <a class="btn btn-back" href="/">← Back to home</a>
    </div>
    <div id="msg"></div>
  </div>
</div>
<script>
  async function save() {{
    const text = document.getElementById('sp').value;
    const res = await fetch('/settings/system-prompt', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{prompt: text}})
    }});
    const data = await res.json();
    showMsg(data.success ? 'ok' : 'err', data.message || data.error);
  }}
  async function resetDefault() {{
    if (!confirm('Reset to the built-in default prompt?')) return;
    const res = await fetch('/settings/system-prompt', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{prompt: ''}})
    }});
    const data = await res.json();
    if (data.success) {{
      document.getElementById('sp').value = data.active_prompt;
      showMsg('ok', 'Reset to default.');
    }} else showMsg('err', data.error);
  }}
  function showMsg(type, text) {{
    const el = document.getElementById('msg');
    el.className = type === 'ok' ? 'msg-ok' : 'msg-err';
    el.textContent = text;
    el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 4000);
  }}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/settings/system-prompt")
async def system_prompt_post(request: Request):
    """Save (or reset) the RAG system prompt."""
    from app.rag import save_system_prompt, load_system_prompt, _SYSTEM_PROMPT_FILE
    try:
        body = await request.json()
        text = (body.get("prompt") or "").strip()
        if text:
            save_system_prompt(text)
            return JSONResponse({"success": True, "message": "System prompt saved.", "active_prompt": text})
        else:
            # Empty → delete the file so default is used
            try:
                if _SYSTEM_PROMPT_FILE.exists():
                    _SYSTEM_PROMPT_FILE.unlink()
            except Exception:
                pass
            return JSONResponse({"success": True, "message": "Reset to default.", "active_prompt": load_system_prompt()})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============================================================
# RESEARCH UI  —  clean no-code interface for document search
# ============================================================

@app.get("/research", response_class=HTMLResponse)
def research_page(request: Request):
    """Serve the focused research UI."""
    default_model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    # Starlette 1.0+ API: request is the first positional arg
    return _research_templates.TemplateResponse(
        request,
        "research.html",
        {"default_model": default_model}
    )


@app.get("/api/research/search")
def research_search(
    q: str = Query(..., min_length=1),
    method: str = Query("standard"),
    k: int = Query(5, ge=1, le=20),
    backend: str = Query("qdrant"),
):
    """
    JSON endpoint: runs FAST vector search and returns normalised results.
    Always uses the base vector search — no LLM calls here so it returns in ~ms.
    Advanced methods (multi-query etc.) do extra LLM work only in /api/research/ai.
    """
    import time
    t0 = time.time()

    if _ingest_state.get("running"):
        raise HTTPException(
            status_code=503,
            detail="Indexing in progress — please wait until the pipeline finishes before searching."
        )

    try:
        if method == "pipeline" and pipeline_search:
            raw = pipeline_search(
                q, backend=backend, k=k, sources_only=True,
            )
            results = raw.get("results", [])
            search_ms = raw.get("search_time_ms")
        else:
            raw = search_knowledge_base(q, backend=backend, k=k)
            results = raw.get("results", [])
            search_ms = raw.get("backend_info", {}).get("search_time_ms")
    except Exception as exc:
        logger.error(f"Research search error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if search_ms is None:
        search_ms = round((time.time() - t0) * 1000, 1)

    return JSONResponse({
        "query":          q,
        "method":         method,
        "backend":        backend.upper(),
        "search_time_ms": search_ms,
        "total_results":  len(results),
        "results":        results,
    })


@app.get("/api/research/ai")
def research_ai(
    q: str = Query(..., min_length=1),
    model: str = Query(None),
    method: str = Query("standard"),
    k: int = Query(5, ge=1, le=20),
    backend: str = Query("qdrant"),
):
    """
    JSON endpoint: runs RAG + LLM (with optional advanced retrieval) and returns AI answer.
    Advanced methods (multi-query, hyde, etc.) do extra LLM retrieval here.
    """
    if not model:
        model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    if _ingest_state.get("running"):
        return JSONResponse({
            "query": q, "answer": "", "ai_response": "",
            "error": "Indexing in progress — please wait until the pipeline finishes before searching.",
        })

    try:
        if method == "pipeline" and pipeline_search:
            data = pipeline_search(
                q, backend=backend, k=k, model=model, filters=None,
                collection_suffix=None,
            )
        elif method in ("multi-query", "decompose", "hyde", "hybrid", "iterative"):
            fn_map = {
                "multi-query": multi_query_search,
                "decompose":   decomposed_search,
                "hyde":        hyde_search,
                "hybrid":      hybrid_search,
                "iterative":   iterative_retrieval,
            }
            fn = fn_map.get(method)
            if fn:
                raw = fn(q, backend=backend, k=k, model=model)
                # Advanced methods return sources + sometimes an answer already
                ai_answer = raw.get("answer") or raw.get("ai_response") or ""
                sources = raw.get("results") or raw.get("sources") or []
                if not ai_answer:
                    # Build the LLM answer from the advanced sources
                    context_data = generate_llm_answer(
                        q, backend=backend, k=k, model=model)
                    ai_answer = context_data.get("ai_response", "")
                data = {
                    "query": q,
                    "ai_response": ai_answer,
                    "model": model,
                    "method": method,
                    "sources": sources,
                    "backend_info": raw.get("timing", {}),
                }
            else:
                data = generate_llm_answer(
                    q, backend=backend, k=k, model=model)
        else:
            data = generate_llm_answer(q, backend=backend, k=k, model=model)
    except Exception as exc:
        logger.error(f"Research AI error: {exc}")
        return JSONResponse({
            "query": q,
            "ai_response": f"AI service error: {exc}",
            "model": model,
            "error": str(exc),
        }, status_code=200)

    data["method"] = method
    return JSONResponse(data)


@app.get("/api/open-file")
def open_file_native(path: str = Query(...), page: int = Query(1)):
    """
    Opens a document with the OS default application (macOS: open, Linux: xdg-open).
    Only resolves paths within the project's data directory for safety.
    Returns open_in_browser=True for text formats so the UI shows them in the modal.
    """
    import platform

    project_root = Path(__file__).parent.parent

    # Normalise: strip leading ./ or /app/
    clean = path.lstrip("./")
    if clean.startswith("app/"):
        clean = clean[4:]

    candidate = (project_root / clean).resolve()

    # Safety: must stay inside project root
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Path outside project directory.")

    if not candidate.is_file():
        return JSONResponse({"success": False, "message": "File not found.", "resolved": str(candidate)})

    ext = candidate.suffix.lower()
    # Text formats: open in the built-in browser modal instead of a system app
    text_extensions = {".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".rst"}
    if ext in text_extensions:
        return JSONResponse({
            "success": True,
            "open_in_browser": True,
            "url": f"/files/serve/{candidate.name}",
            "message": f"Viewing {candidate.name}",
            "page_hint": page,
        })

    try:
        system = platform.system()
        if system == "Darwin":
            if ext == ".pdf" and page > 1:
                # macOS: open PDF at specific page via AppleScript (Preview)
                script = (
                    f'tell application "Preview" to activate\n'
                    f'tell application "Preview" to open POSIX file "{candidate}"\n'
                )
                subprocess.Popen(["osascript", "-e", script])
            else:
                subprocess.Popen(["open", str(candidate)])
        elif system == "Linux":
            subprocess.Popen(["xdg-open", str(candidate)])
        elif system == "Windows":
            os.startfile(str(candidate))  # type: ignore[attr-defined]
        else:
            return JSONResponse({"success": False, "message": f"Unsupported OS: {system}"})
        return JSONResponse({"success": True, "message": f"Opened {candidate.name}", "page_hint": page})
    except Exception as exc:
        logger.error(f"open-file error: {exc}")
        return JSONResponse({"success": False, "message": str(exc)})


@app.get("/api/research/file-content/{filename}")
def api_file_content(filename: str):
    """Return the raw text content of a file in data/raw as JSON for in-browser rendering."""
    if any(c in filename for c in ("/", "\\", "..")) or not filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    file_path = (_RAW_DIR / filename).resolve()
    try:
        file_path.relative_to(_RAW_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Access denied.")
    if not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"File not found: {filename}")
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return JSONResponse({"filename": filename, "content": text, "size": len(text)})


@app.get("/files/serve/{filename}")
def serve_file(filename: str):
    """
    Serve a raw document file for in-browser viewing.
    Only files inside ./data/raw/ are accessible to prevent path traversal.
    """
    from fastapi.responses import FileResponse

    # Strict filename validation — no path components allowed
    if any(c in filename for c in ("/", "\\", "..")) or not filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    file_path = (raw_dir / filename).resolve()

    # Confirm the resolved path is inside data/raw
    try:
        file_path.relative_to(raw_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Access denied.")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        path=file_path,
        filename=filename,
        headers={"Content-Disposition": "inline"},
    )


# ============================================================
# RESEARCH UI  —  Library & Ingest API (local-mode aware)
# ============================================================

@app.post("/api/research/upload")
async def api_research_upload(file: UploadFile = File(...)):
    """Upload a document to data/raw (local path, safe filename)."""
    ALLOWED = {'.pdf', '.txt', '.md', '.yaml', '.yml', '.docx', '.doc'}
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(
            status_code=400, detail=f"File type not allowed: {ext}")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail="File too large (max 50 MB).")
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name  # strip any directory components
    dest = _RAW_DIR / safe_name
    dest.write_bytes(content)
    logger.info(f"Research upload: {safe_name} ({len(content)} bytes)")
    return JSONResponse({"success": True, "filename": safe_name, "size": len(content)})


@app.get("/api/research/library")
def api_research_library():
    """List documents in data/raw with indexing status."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    _CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    try:
        entries = sorted(_RAW_DIR.iterdir(),
                         key=lambda f: f.stat().st_mtime, reverse=True)
    except Exception:
        entries = []
    for fp in entries:
        if not fp.is_file() or fp.name.startswith('.'):
            continue
        stat = fp.stat()
        chunks = 0
        for candidate in [
            _CLEAN_DIR / f"{fp.name}.chunks.jsonl",
            _CLEAN_DIR / f"{fp.stem}.chunks.jsonl",
        ]:
            if candidate.exists():
                try:
                    with candidate.open() as f:
                        chunks = sum(1 for line in f if line.strip())
                except Exception:
                    pass
                break
        files.append({
            "filename": fp.name,
            "size":     stat.st_size,
            "ext":      fp.suffix.lower(),
            "indexed":  chunks > 0,
            "chunks":   chunks,
            "modified": stat.st_mtime,
        })
    return JSONResponse({"files": files, "count": len(files)})


@app.delete("/api/research/library/{filename}")
def api_research_delete(filename: str):
    """Delete a file from data/raw (path-traversal safe)."""
    if any(c in filename for c in ("/", "\\", "..")) or not filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    fp = (_RAW_DIR / filename).resolve()
    try:
        fp.relative_to(_RAW_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Access denied.")
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    fp.unlink()
    logger.info(f"Research delete: {filename}")
    return JSONResponse({"success": True, "deleted": filename})


@app.post("/api/research/ingest")
def api_research_ingest(clear: bool = False):
    """Start the ingest pipeline in a background thread.

    Query param:
      clear=true  — delete all cached .jsonl and .chunks.jsonl files from data/clean/
                    before running, so the pipeline fully re-processes raw documents.
    """
    import threading
    global _ingest_state
    if _ingest_state.get("running"):
        return JSONResponse({"started": False, "message": "Pipeline already running."})
    script = _PROJECT_ROOT / "scripts" / "main_pipeline.py"
    if not script.exists():
        return JSONResponse({"started": False, "message": f"Pipeline script not found: {script}"})

    # Delete cached processed files so they are regenerated from scratch
    deleted_files = []
    if clear:
        for ext in ("*.jsonl", "*.chunks.jsonl"):
            for f in _CLEAN_DIR.glob(ext):
                try:
                    f.unlink()
                    deleted_files.append(f.name)
                except Exception as exc:
                    logger.warning(f"Could not delete {f.name}: {exc}")

    _ingest_state = {
        "running": True, "done": False, "error": None,
        "log": (["🗑 Cleared cached files: " + ", ".join(deleted_files)] if deleted_files else []) + ["Starting pipeline…"],
        "started_at": time.time(),
    }

    # Release the local Qdrant file lock before the subprocess opens it.
    from app.qdrant_backend import close_client as _close_qdrant
    _close_qdrant()

    def run():
        global _ingest_state
        import os as _os
        _env = _os.environ.copy()
        # Resolve Docker-only hostnames to localhost when running outside Docker
        _qhost = _env.get("QDRANT_HOST", "localhost")
        if _qhost in ("qdrant", "qdrant-db"):  # Docker service names
            _qhost = "localhost"
        _qport = _env.get("QDRANT_PORT", "6333")
        _env["QDRANT_URL"] = f"http://{_qhost}:{_qport}"
        _env["QDRANT_HOST"] = _qhost
        # Also fix Postgres host if set to Docker service name
        _pghost = _env.get("POSTGRES_HOST", "localhost")
        if _pghost in ("postgres", "db", "postgresql"):
            _pghost = "localhost"
        _env["POSTGRES_HOST"] = _pghost
        try:
            proc = __import__("subprocess").Popen(
                [sys.executable, str(script)],
                cwd=str(_PROJECT_ROOT),
                env=_env,
                stdout=__import__("subprocess").PIPE,
                stderr=__import__("subprocess").STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _ingest_state["log"].append(line)
                    if len(_ingest_state["log"]) > 300:
                        _ingest_state["log"] = _ingest_state["log"][-300:]
            proc.wait()
            if proc.returncode == 0:
                _ingest_state["log"].append(
                    "✓ Pipeline completed successfully")
                _ingest_state["done"] = True
            else:
                _ingest_state["error"] = f"Process exited with code {proc.returncode}"
                _ingest_state["log"].append(
                    f"✗ Pipeline failed (code {proc.returncode})")
        except Exception as exc:
            _ingest_state["error"] = str(exc)
            _ingest_state["log"].append(f"✗ Exception: {exc}")
        finally:
            _ingest_state["running"] = False
            # Client was closed before the subprocess started; reset the
            # singleton so the next search/AI request reopens it cleanly.
            from app.qdrant_backend import close_client as _close_qdrant2
            _close_qdrant2()

    threading.Thread(target=run, daemon=True).start()
    return JSONResponse({"started": True})


@app.get("/api/research/ingest/status")
def api_research_ingest_status():
    """Poll ingest pipeline status."""
    elapsed = round(time.time() - _ingest_state["started_at"], 1) \
        if _ingest_state.get("started_at") else 0
    return JSONResponse({
        "running": _ingest_state["running"],
        "done":    _ingest_state["done"],
        "error":   _ingest_state["error"],
        "log":     _ingest_state["log"][-60:],
        "elapsed": elapsed,
    })
