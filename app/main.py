from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import json
from app.rag import rag_answer, generate_llm_answer

# Import query utilities from consolidated module
try:
    from scripts.query_embed import embed_e5, expand_query
except ImportError:
    # Fallback to utils if query_embed not available
    try:
        from scripts.utils import embed_e5, expand_query
    except ImportError:
        embed_e5 = None
        def expand_query(q): return q

from app.qdrant_backend import search_qdrant
from app.pgvector_backend import search_pgvector
app = FastAPI(title="RAG Demo - Qdrant vs PGvector Postgres")


def enhanced_ai_response_html(data: dict, query: str) -> str:
    """Enhanced HTML for AI responses with better UX"""
    
    # Extract key information
    ai_response = data.get("ai_response", "")
    sources = data.get("sources", [])
    total_results = data.get("total_results", 0)
    backend = data.get("backend", "")
    model = data.get("model", "")
    
    # Build sources section with page references
    sources_html = ""
    if sources:
        sources_html = "<div class='sources-section'><h3>📚 Fuentes:</h3>"
        for i, source in enumerate(sources, 1):
            doc_name = source.get("document", "").replace("data/raw/", "").replace(".pdf", "")
            similarity = source.get("similarity", "0.000")
            preview = source.get("preview", "")
            reference = source.get("reference", "")
            page = source.get("page")
            chapter = source.get("chapter")
            
            # Build detailed reference info
            ref_info = []
            if reference:
                ref_info.append(reference)
            elif page:
                ref_info.append(f"{doc_name} - página {page}")
            else:
                ref_info.append(doc_name)
            
            if chapter:
                ref_info.append(chapter)
            
            full_reference = ", ".join(ref_info)
            
            sources_html += f"""
            <div class='source-item'>
                <div class='source-header'>
                    <span class='source-number'>{i}.</span>
                    <span class='source-doc'>{full_reference}</span>
                    <span class='source-similarity'>Similitud: {similarity}</span>
                </div>
                <div class='source-preview'>{preview}</div>
            </div>
            """
        sources_html += "</div>"

    # JSON details (expandable)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 AI Response: {query}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                background: #0f0f0f; 
                color: #e1e1e1; 
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: #1a1a1a; 
                border-radius: 12px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2d5aa0 0%, #1e3a5f 100%);
                padding: 20px 30px;
                color: white;
            }}
            .search-bar {{
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }}
            .search-input {{
                flex: 1;
                padding: 12px 15px;
                border: none;
                border-radius: 8px;
                background: rgba(255,255,255,0.1);
                color: white;
                font-size: 16px;
            }}
            .search-input::placeholder {{ color: rgba(255,255,255,0.7); }}
            .search-btn, .nav-btn {{
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                background: rgba(255,255,255,0.2);
                color: white;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s;
            }}
            .search-btn:hover, .nav-btn:hover {{
                background: rgba(255,255,255,0.3);
                transform: translateY(-1px);
            }}
            .title {{ 
                font-size: 28px; 
                margin: 0;
                font-weight: 600;
            }}
            .subtitle {{ 
                opacity: 0.9; 
                margin-top: 5px;
                font-size: 16px;
            }}
            .content {{
                padding: 30px;
            }}
            .question-section {{
                background: #242424;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                border-left: 4px solid #4a90e2;
            }}
            .question-label {{
                font-weight: 600;
                color: #4a90e2;
                font-size: 18px;
                margin-bottom: 10px;
            }}
            .question-text {{
                font-size: 18px;
                color: #f0f0f0;
            }}
            .answer-section {{
                background: #1f2937;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 25px;
                border-left: 4px solid #10b981;
            }}
            .answer-label {{
                font-weight: 600;
                color: #10b981;
                font-size: 18px;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .answer-text {{
                font-size: 18px;
                line-height: 1.7;
                color: #f9fafb;
            }}
            .sources-section {{
                margin-top: 30px;
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
            }}
            .sources-section h3 {{
                margin: 0 0 20px 0;
                color: #f59e0b;
                font-size: 20px;
            }}
            .source-item {{
                background: #333;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border: 1px solid #444;
            }}
            .source-header {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 10px;
                flex-wrap: wrap;
            }}
            .source-number {{
                background: #4a90e2;
                color: white;
                padding: 4px 8px;
                border-radius: 50%;
                font-weight: bold;
                font-size: 14px;
            }}
            .source-doc {{
                font-weight: 600;
                color: #60a5fa;
                flex: 1;
                min-width: 200px;
            }}
            .source-similarity {{
                background: #065f46;
                color: #10b981;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
            }}
            .source-preview {{
                color: #d1d5db;
                font-style: italic;
                border-left: 3px solid #4b5563;
                padding-left: 12px;
                margin-left: 10px;
            }}
            .details-toggle {{
                margin-top: 30px;
                text-align: center;
            }}
            .toggle-btn {{
                background: #374151;
                color: #e5e7eb;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s;
            }}
            .toggle-btn:hover {{
                background: #4b5563;
                transform: translateY(-1px);
            }}
            .json-details {{
                display: none;
                margin-top: 20px;
                background: #111827;
                padding: 20px;
                border-radius: 8px;
                border: 1px solid #374151;
            }}
            .json-details.show {{ display: block; }}
            .json-content {{
                background: #0f172a;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.4;
            }}
            .navigation {{
                display: flex;
                gap: 15px;
                justify-content: space-between;
                margin-top: 30px;
                flex-wrap: wrap;
            }}
            .nav-group {{
                display: flex;
                gap: 10px;
            }}
            .nav-link {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 20px;
                background: #374151;
                color: #e5e7eb;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 500;
                transition: all 0.3s;
                border: 1px solid #4b5563;
            }}
            .nav-link:hover {{
                background: #4b5563;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .nav-link.primary {{
                background: #1d4ed8;
                border-color: #2563eb;
            }}
            .nav-link.primary:hover {{
                background: #2563eb;
            }}
            .model-info {{
                font-size: 14px;
                color: #9ca3af;
                margin-top: 10px;
            }}
            
            /* JSON Syntax Highlighting */
            .string {{ color: #fbbf24; }}
            .number {{ color: #34d399; }}
            .boolean {{ color: #60a5fa; }}
            .null {{ color: #f87171; }}
            .key {{ color: #a78bfa; }}
            
            @media (max-width: 768px) {{
                .container {{ margin: 10px; }}
                .header {{ padding: 15px 20px; }}
                .content {{ padding: 20px; }}
                .search-bar {{ flex-direction: column; }}
                .source-header {{ flex-direction: column; align-items: flex-start; }}
                .navigation {{ flex-direction: column; }}
                .nav-group {{ justify-content: center; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="search-bar">
                    <input type="text" class="search-input" placeholder="Nueva búsqueda..." value="{query}" id="searchInput">
                    <button class="search-btn" onclick="newSearch()">🔍 Buscar</button>
                    <button class="nav-btn" onclick="window.location.href='/ask?q=bases+de+datos+vectoriales'">🎯 Solo /ask</button>
                    <button class="nav-btn" onclick="window.location.href='/'">🏠 Menu Principal</button>
                </div>
                <div class="title">🤖 Respuesta AI</div>
                <div class="subtitle">Resultados: {total_results} | Motor: {backend} | Modelo: {model}</div>
            </div>
            
            <div class="content">
                <div class="question-section">
                    <div class="question-label">Q:</div>
                    <div class="question-text">{query}</div>
                </div>
                
                <div class="answer-section">
                    <div class="answer-label">
                        <span>Answer AI:</span>
                        <span style="font-size: 12px; background: rgba(16,185,129,0.2); padding: 2px 6px; border-radius: 3px;">{model}</span>
                    </div>
                    <div class="answer-text">{ai_response}</div>
                    <div class="model-info">Respuesta generada por {model} usando {total_results} fuentes relevantes</div>
                </div>
                
                {sources_html}
                
                <div class="details-toggle">
                    <button class="toggle-btn" onclick="toggleDetails()">🔧 Ver detalles técnicos JSON</button>
                    <div class="json-details" id="jsonDetails">
                        <div class="json-content" id="jsonContent">{json_str}</div>
                    </div>
                </div>
                
                <div class="navigation">
                    <div class="nav-group">
                        <a href="/" class="nav-link primary">🏠 Inicio</a>
                        <a href="/ask?q=evaluación+del+curso" class="nav-link">📋 Búsquedas Rápidas</a>
                    </div>
                    <div class="nav-group">
                        <a href="/docs" class="nav-link">📖 API Docs</a>
                        <a href="/compare?q={query}" class="nav-link">⚖️ Comparar Motores</a>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function toggleDetails() {{
                const details = document.getElementById('jsonDetails');
                const btn = document.querySelector('.toggle-btn');
                
                if (details.classList.contains('show')) {{
                    details.classList.remove('show');
                    btn.textContent = '🔧 Ver detalles técnicos JSON';
                }} else {{
                    details.classList.add('show');
                    btn.textContent = '🔧 Ocultar detalles JSON';
                    
                    // Apply syntax highlighting
                    highlightJSON();
                }}
            }}
            
            function highlightJSON() {{
                const jsonContent = document.getElementById('jsonContent');
                let html = jsonContent.innerHTML;
                
                // Highlight strings
                html = html.replace(/"([^"]+)":/g, '<span class="key">"$1"</span>:');
                html = html.replace(/: "([^"]*)"/g, ': <span class="string">"$1"</span>');
                
                // Highlight numbers
                html = html.replace(/: (\\d+\\.?\\d*)/g, ': <span class="number">$1</span>');
                
                // Highlight booleans and null
                html = html.replace(/: (true|false)/g, ': <span class="boolean">$1</span>');
                html = html.replace(/: (null)/g, ': <span class="null">$1</span>');
                
                jsonContent.innerHTML = html;
            }}
            
            function newSearch() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    window.location.href = `/ai?q=${{encodeURIComponent(query.trim())}}`;
                }}
            }}
            
            // Allow Enter key to search
            document.getElementById('searchInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    newSearch();
                }}
            }});
        </script>
    </body>
    </html>
    """


def pretty_json_html(data: dict, title: str = "API Response") -> str:
    """Convert JSON to pretty HTML with syntax highlighting"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ 
                font-family: 'Courier New', monospace; 
                background: #1e1e1e; 
                color: #d4d4d4; 
                margin: 20px; 
                line-height: 1.4;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: #252526; 
                padding: 20px; 
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            .title {{ 
                color: #569cd6; 
                font-size: 24px; 
                margin-bottom: 20px; 
                border-bottom: 2px solid #569cd6;
                padding-bottom: 10px;
            }}
            pre {{ 
                background: #1e1e1e; 
                padding: 15px; 
                border-radius: 5px; 
                overflow-x: auto;
                border: 1px solid #3e3e42;
            }}
            .string {{ color: #ce9178; }}
            .number {{ color: #b5cea8; }}
            .boolean {{ color: #569cd6; }}
            .null {{ color: #569cd6; }}
            .key {{ color: #9cdcfe; }}
            .back-link {{ 
                display: inline-block; 
                margin-top: 20px; 
                color: #569cd6; 
                text-decoration: none;
                padding: 8px 16px;
                border: 1px solid #569cd6;
                border-radius: 4px;
                transition: background 0.3s;
            }}
            .back-link:hover {{ 
                background: #569cd6; 
                color: #1e1e1e; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="title">🚀 {title}</div>
            <pre id="json-content">{json_str}</pre>
            <a href="/" class="back-link">← Volver al inicio</a>
            <a href="/docs" class="back-link">📖 Documentación API</a>
        </div>
        <script>
            // Simple JSON syntax highlighting
            const jsonContent = document.getElementById('json-content');
            let html = jsonContent.innerHTML;
            
            // Highlight strings
            html = html.replace(/"([^"]+)":/g, '<span class="key">"$1"</span>:');
            html = html.replace(/: "([^"]*)"/g, ': <span class="string">"$1"</span>');
            
            // Highlight numbers
            html = html.replace(/: (\d+\.?\d*)/g, ': <span class="number">$1</span>');
            
            // Highlight booleans and null
            html = html.replace(/: (true|false)/g, ': <span class="boolean">$1</span>');
            html = html.replace(/: (null)/g, ': <span class="null">$1</span>');
            
            jsonContent.innerHTML = html;
        </script>
    </body>
    </html>
    """


@app.get("/ask", response_class=HTMLResponse)
def ask(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    format: str = Query("html", description="Formato: 'json' o 'html'"),
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

        result = rag_answer(q, backend=backend, k=k, filters=filters or None)

        if format == "json":
            return result
        else:
            return pretty_json_html(result, f"🔍 Búsqueda: {q}")
    except ValueError as e:
        error_data = {"error": str(e), "status": 400}
        if format == "json":
            raise HTTPException(status_code=400, detail=str(e))
        return pretty_json_html(error_data, "❌ Error")
    except Exception as e:
        error_data = {
            "error": f"Internal server error: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}")
        return pretty_json_html(error_data, "❌ Error")


@app.get("/ai", response_class=HTMLResponse)
def ai(
    q: str = Query(..., description="Pregunta"),
    backend: str = "qdrant",
    k: int = 5,
    model: str = "phi3:mini",
    format: str = Query("html", description="Formato: 'json' o 'html'"),
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

        if format == "json":
            return JSONResponse(content=result)
        else:
            return enhanced_ai_response_html(result, q)
    except ImportError as e:
        error_data = {
            "error": f"LLM service not available: {str(e)}", "status": 503}
        if format == "json":
            raise HTTPException(
                status_code=503, detail=f"LLM service not available: {str(e)}")
        return pretty_json_html(error_data, "❌ Error LLM")
    except Exception as e:
        error_data = {"error": f"AI error: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
        return pretty_json_html(error_data, "❌ Error AI")


@app.get("/compare", response_class=HTMLResponse)
def compare(
    q: str = Query(..., description="Pregunta"),
    k: int = 3,
    format: str = Query("html", description="Formato: 'json' o 'html'")
):
    """Compare results from both Qdrant and pgvector backends"""
    try:
        qdrant_results = rag_answer(q, backend="qdrant", k=k)
        pgvector_results = rag_answer(q, backend="pgvector", k=k)

        result = {
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

        if format == "json":
            return result
        else:
            return pretty_json_html(result, f"⚖️ Comparación: {q}")
    except Exception as e:
        error_data = {"error": f"Comparison error: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Comparison error: {str(e)}")
        return pretty_json_html(error_data, "❌ Error Comparación")


@app.get("/manual/embed", response_class=HTMLResponse)
def manual_embed_demo(
    q: str = Query("bases de datos vectoriales",
                   description="Texto a vectorizar"),
    format: str = Query("html", description="Formato: 'json' o 'html'")
):
    """🔧 Demostración manual del proceso de vectorización paso a paso"""
    try:
        # Paso 1: Mostrar query original
        step1 = {
            "paso": 1,
            "descripcion": "📝 Consulta original del usuario",
            "query_original": q,
            "explicacion": "Esta es la consulta en lenguaje natural que quiere hacer el usuario"
        }

        # Paso 2: Expansión de consulta
        expanded = expand_query(q)
        step2 = {
            "paso": 2,
            "descripcion": "🔄 Expansión y normalización de la consulta",
            "query_expandida": expanded,
            "cambios": "Se mejora la consulta para mejor recuperación" if expanded != q else "No se necesitan cambios",
            "explicacion": "Se procesan sinónimos y términos relacionados para mejor búsqueda"
        }

        # Paso 3: Vectorización
        embedding = embed_e5([expanded], is_query=True)[0]
        step3 = {
            "paso": 3,
            "descripcion": "🧮 Conversión a vector numérico (embedding)",
            "vector_dimensiones": len(embedding),
            "modelo_usado": "intfloat/multilingual-e5-base",
            "primeros_10_valores": embedding[:10],
            "ultimos_10_valores": embedding[-10:],
            "explicacion": "El texto se convierte en un vector de 768 números que representa su significado semántico",
            "como_funciona": "El modelo E5 fue entrenado para que textos similares tengan vectores similares"
        }

        result = {
            "titulo": "🔧 PROCESO MANUAL DE VECTORIZACIÓN",
            "resumen": "Conversión de texto natural a vector numérico para búsqueda semántica",
            "pasos": [step1, step2, step3],
            "vector_completo_disponible": f"Vector completo de {len(embedding)} dimensiones generado exitosamente",
            "siguiente_paso": "Usar este vector para buscar documentos similares con /manual/search"
        }

        if format == "json":
            return result
        else:
            return pretty_json_html(result, f"🔧 Vectorización: {q}")

    except Exception as e:
        error_data = {
            "error": f"Error en demostración: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en demostración: {str(e)}")
        return pretty_json_html(error_data, "❌ Error Demo")


@app.get("/manual/search", response_class=HTMLResponse)
def manual_search_demo(
    q: str = Query("bases de datos vectoriales",
                   description="Consulta para buscar"),
    backend: str = Query("qdrant", description="Motor de búsqueda"),
    k: int = Query(3, description="Número de resultados"),
    format: str = Query("html", description="Formato: 'json' o 'html'")
):
    """🔍 Demostración manual del proceso de búsqueda vectorial paso a paso"""
    try:

        # Paso 1: Vectorización de la consulta
        expanded = expand_query(q)
        embedding = embed_e5([expanded], is_query=True)[0]

        step1 = {
            "paso": 1,
            "descripcion": "🧮 Vectorización de la consulta",
            "query_original": q,
            "query_expandida": expanded,
            "vector_generado": f"Vector de {len(embedding)} dimensiones"
        }

        # Paso 2: Búsqueda en la base de datos
        if backend == "qdrant":
            results = search_qdrant(embedding, k=k)
            motor_info = {
                "nombre": "Qdrant",
                "tipo": "Base de datos vectorial especializada",
                "algoritmo": "HNSW (Hierarchical Navigable Small World)",
                "metrica": "Similaridad coseno"
            }
        else:
            results = search_pgvector(embedding, k=k)
            motor_info = {
                "nombre": "PostgreSQL + pgvector",
                "tipo": "Extensión vectorial para PostgreSQL",
                "algoritmo": "Búsqueda de vecinos más cercanos",
                "metrica": "Distancia coseno"
            }

        step2 = {
            "paso": 2,
            "descripcion": "🔍 Búsqueda en la base de datos vectorial",
            "motor_usado": motor_info,
            "proceso": "Se compara el vector de consulta con todos los vectores de documentos",
            "resultados_encontrados": len(results)
        }

        # Paso 3: Cálculo de similaridad y ranking
        resultados_detallados = []
        for i, result in enumerate(results, 1):
            score = float(result.get('score', 0))
            content_preview = result.get('content', '')[
                :150] + "..." if len(result.get('content', '')) > 150 else result.get('content', '')

            resultados_detallados.append({
                "posicion": i,
                "documento": result.get('path', '').replace('./data/raw/', ''),
                "pagina": result.get('page'),
                "chunk_id": result.get('chunk_id'),
                "similaridad": f"{score:.4f}",
                "similaridad_porcentaje": f"{score * 100:.1f}%",
                "contenido_preview": content_preview,
                "explicacion_score": "Más cercano a 1.0 = más similar al texto consultado"
            })

        step3 = {
            "paso": 3,
            "descripcion": "📊 Ranking por similaridad",
            "explicacion": "Los resultados se ordenan por similaridad coseno (0.0 a 1.0)",
            "metrica_usada": "Similaridad coseno: mide el ángulo entre vectores",
            "interpretacion": "0.9+ = Muy similar, 0.7-0.9 = Similar, 0.5-0.7 = Algo relacionado, <0.5 = Poco relacionado",
            "resultados_ordenados": resultados_detallados
        }

        result = {
            "titulo": "🔍 PROCESO MANUAL DE BÚSQUEDA VECTORIAL",
            "resumen": f"Búsqueda semántica usando {motor_info['nombre']} con {len(results)} resultados",
            "pasos": [step1, step2, step3],
            "conclusion": "Los vectores permiten encontrar documentos por significado, no solo por palabras exactas"
        }

        if format == "json":
            return result
        else:
            return pretty_json_html(result, f"🔍 Búsqueda Manual: {q}")

    except Exception as e:
        error_data = {
            "error": f"Error en búsqueda manual: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en búsqueda manual: {str(e)}")
        return pretty_json_html(error_data, "❌ Error Búsqueda")


@app.get("/manual/demo", response_class=HTMLResponse)
def manual_complete_demo(
    q: str = Query("bases de datos vectoriales",
                   description="Consulta de ejemplo"),
    backend: str = Query("qdrant", description="Motor de búsqueda"),
    format: str = Query("html", description="Formato: 'json' o 'html'")
):
    """🎓 Demostración completa del proceso RAG para la clase"""
    try:

        # Obtener proceso de embedding
        embed_response = manual_embed_demo(q)

        # Obtener proceso de búsqueda
        search_response = manual_search_demo(q, backend, k=3)

        result = {
            "titulo": "🎓 DEMOSTRACIÓN COMPLETA: BÚSQUEDA VECTORIAL SEMÁNTICA",
            "subtitulo": "Proceso completo de cómo funcionan las bases de datos vectoriales",
            "introduccion": {
                "que_es": "Un sistema que convierte texto en números (vectores) para buscar por significado",
                "por_que_funciona": "Textos similares en significado tienen vectores similares",
                "ventaja": "Puede encontrar documentos relevantes aunque no compartan palabras exactas"
            },
            "proceso_embedding": embed_response,
            "proceso_busqueda": search_response,
            "resumen_tecnico": {
                "modelo_embedding": "intfloat/multilingual-e5-base (768 dimensiones)",
                "base_datos": f"{backend.upper()} - Motor de búsqueda vectorial",
                "metrica": "Similaridad coseno (0.0 a 1.0)",
                "ventajas": [
                    "Búsqueda semántica (por significado)",
                    "Funciona en múltiples idiomas",
                    "No requiere palabras exactas",
                    "Escalable a millones de documentos"
                ]
            },
            "ejemplo_practico": {
                "consulta": q,
                "resultados_encontrados": len(search_response["pasos"][2]["resultados_ordenados"]),
                "mejor_resultado": search_response["pasos"][2]["resultados_ordenados"][0] if search_response["pasos"][2]["resultados_ordenados"] else None
            }
        }

        if format == "json":
            return result
        else:
            return pretty_json_html(result, f"🎓 Demo Completa: {q}")

    except Exception as e:
        error_data = {
            "error": f"Error en demo completa: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en demo completa: {str(e)}")
        return pretty_json_html(error_data, "❌ Error Demo")


@app.get("/filters/examples", response_class=HTMLResponse)
def filter_examples(format: str = Query("html", description="Formato: 'json' o 'html'")):
    """📋 Ejemplos de filtros de metadata disponibles"""
    result = {
        "titulo": "📋 EJEMPLOS DE FILTROS DE METADATA",
        "descripcion": "Cómo usar filtros para búsquedas más específicas",
        "filtros_disponibles": {
            "document_type": {
                "descripcion": "Tipo de documento",
                "valores": ["pdf", "txt", "md"],
                "ejemplo": "/ask?q=vectores&document_type=pdf"
            },
            "section": {
                "descripcion": "Sección del curso",
                "valores": ["objetivos", "cronograma", "evaluacion", "proyectos"],
                "ejemplo": "/ask?q=evaluacion&section=objetivos"
            },
            "topic": {
                "descripcion": "Tema específico",
                "valores": ["nosql", "vectorial", "sql", "mongodb", "qdrant"],
                "ejemplo": "/ask?q=bases de datos&topic=vectorial"
            },
            "page": {
                "descripcion": "Página específica (solo PDFs)",
                "valores": "número entero",
                "ejemplo": "/ask?q=proyecto&page=5"
            },
            "contains": {
                "descripcion": "Debe contener esta palabra",
                "valores": "cualquier texto",
                "ejemplo": "/ask?q=mongodb&contains=NoSQL"
            }
        },
        "ejemplos_combinados": [
            {
                "descripcion": "Buscar información de evaluación solo en PDFs",
                "url": "/ask?q=evaluacion&document_type=pdf&section=evaluacion"
            },
            {
                "descripcion": "Buscar sobre bases vectoriales en objetivos del curso",
                "url": "/ask?q=vectoriales&section=objetivos&topic=vectorial"
            },
            {
                "descripcion": "Buscar proyectos que mencionen NoSQL",
                "url": "/ask?q=proyecto&contains=NoSQL&section=proyectos"
            }
        ],
        "filtros_qdrant": {
            "descripcion": "Filtros específicos para Qdrant (formato interno)",
            "ejemplo_simple": {"must": [{"key": "document_type", "match": {"value": "pdf"}}]},
            "ejemplo_combinado": {
                "must": [
                    {"key": "section", "match": {"value": "objetivos"}},
                    {"key": "document_type", "match": {"value": "pdf"}}
                ]
            }
        }
    }

    if format == "json":
        return result
    else:
        return pretty_json_html(result, "📋 Ejemplos de Filtros")


@app.get("/", response_class=HTMLResponse)
def root(format: str = Query("html", description="Formato: 'json' o 'html'")):
    """Enhanced home page with search interface and quick actions"""
    if format == "json":
        result = {
            "message": "🚀 RAG Demo API - Qdrant vs pgvector + AI",
            "version": "3.0",
            "features": [
                "✨ Enhanced UI with search bar and navigation",
                "🔍 Semantic search with E5 multilingual embeddings", 
                "🤖 AI-powered responses with Ollama LLMs",
                "📊 Page and chapter references in results",
                "📋 Metadata filtering (document_type, section, topic, page, contains)",
                "⚖️ Backend comparison (Qdrant vs PostgreSQL+pgvector)",
                "🎓 Educational demos for classroom demonstrations",
                "✂️ Smart chunking (200 tokens, preserves context)"
            ],
            "endpoints": {
                "/": "Enhanced home with search interface",
                "/ask": "RAG search with metadata filtering (HTML/JSON)",
                "/ai": "AI-powered RAG with LLM generation (requires Ollama)",
                "/compare": "Side-by-side comparison of Qdrant vs pgvector",
                "/docs": "� OpenAPI/Swagger documentation"
            }
        }
        return result
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 RAG Demo - Main Menu</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
                color: #e1e1e1; 
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .container {{ 
                max-width: 900px; 
                margin: 20px;
                background: rgba(26, 26, 26, 0.95); 
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.5);
                overflow: hidden;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #2d5aa0 0%, #1e3a5f 100%);
                padding: 40px 30px;
                text-align: center;
                color: white;
            }}
            .title {{ 
                font-size: 36px; 
                margin: 0 0 10px 0;
                font-weight: 700;
                background: linear-gradient(45deg, #ffffff, #a8e6cf);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            .subtitle {{ 
                font-size: 18px;
                opacity: 0.9;
                margin: 0;
            }}
            .search-section {{
                padding: 40px 30px;
                text-align: center;
            }}
            .search-title {{
                font-size: 24px;
                margin-bottom: 20px;
                color: #4a90e2;
                font-weight: 600;
            }}
            .search-bar {{
                display: flex;
                gap: 15px;
                max-width: 600px;
                margin: 0 auto 30px auto;
            }}
            .search-input {{
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #374151;
                border-radius: 12px;
                background: #1f2937;
                color: #e5e7eb;
                font-size: 16px;
                transition: all 0.3s;
            }}
            .search-input:focus {{
                outline: none;
                border-color: #4a90e2;
                box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.2);
            }}
            .search-input::placeholder {{ color: #9ca3af; }}
            .search-btn {{
                padding: 15px 25px;
                border: none;
                border-radius: 12px;
                background: linear-gradient(45deg, #4a90e2, #357abd);
                color: white;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s;
                min-width: 120px;
            }}
            .search-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
            }}
            .quick-searches {{
                margin-top: 20px;
            }}
            .quick-title {{
                font-size: 16px;
                color: #9ca3af;
                margin-bottom: 15px;
            }}
            .quick-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
            }}
            .quick-btn {{
                padding: 8px 16px;
                background: rgba(74, 144, 226, 0.1);
                border: 1px solid rgba(74, 144, 226, 0.3);
                border-radius: 20px;
                color: #60a5fa;
                text-decoration: none;
                font-size: 14px;
                transition: all 0.3s;
            }}
            .quick-btn:hover {{
                background: rgba(74, 144, 226, 0.2);
                transform: translateY(-1px);
            }}
            .features-section {{
                padding: 30px;
                background: #1f2937;
            }}
            .features-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .feature-card {{
                background: #374151;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #4b5563;
                transition: all 0.3s;
            }}
            .feature-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                border-color: #60a5fa;
            }}
            .feature-icon {{
                font-size: 24px;
                margin-bottom: 10px;
            }}
            .feature-title {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 8px;
                color: #e5e7eb;
            }}
            .feature-desc {{
                color: #9ca3af;
                font-size: 14px;
                line-height: 1.5;
            }}
            .action-card {{
                background: linear-gradient(45deg, #1e40af, #1e3a8a);
                cursor: pointer;
            }}
            .action-card:hover {{
                background: linear-gradient(45deg, #2563eb, #1e40af);
            }}
            .nav-section {{
                padding: 30px;
                background: #111827;
                text-align: center;
            }}
            .nav-buttons {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                justify-content: center;
            }}
            .nav-link {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 12px 20px;
                background: #374151;
                color: #e5e7eb;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 500;
                transition: all 0.3s;
                border: 1px solid #4b5563;
            }}
            .nav-link:hover {{
                background: #4b5563;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .nav-link.primary {{
                background: linear-gradient(45deg, #059669, #047857);
                border-color: #10b981;
            }}
            .nav-link.primary:hover {{
                background: linear-gradient(45deg, #10b981, #059669);
            }}
            
            @media (max-width: 768px) {{
                .container {{ margin: 10px; }}
                .header {{ padding: 30px 20px; }}
                .search-section {{ padding: 30px 20px; }}
                .search-bar {{ flex-direction: column; }}
                .quick-buttons {{ flex-direction: column; align-items: center; }}
                .nav-buttons {{ flex-direction: column; }}
                .features-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">🚀 RAG Demo</div>
                <div class="subtitle">Sistema de Búsqueda Inteligente con IA</div>
            </div>
            
            <div class="search-section">
                <div class="search-title">🔍 Buscar en Documentos</div>
                <div class="search-bar">
                    <input type="text" class="search-input" placeholder="Ej: ¿Cuáles son las vacunas recomendadas para embarazadas?" id="searchInput">
                    <button class="search-btn" onclick="searchWithAI()">🤖 AI Search</button>
                </div>
                
                <div class="quick-searches">
                    <div class="quick-title">🎯 Búsquedas Rápidas:</div>
                    <div class="quick-buttons">
                        <a href="/ai?q=vacunas+embarazadas" class="quick-btn">💉 Vacunas</a>
                        <a href="/ai?q=hipertensión+embarazo" class="quick-btn">🩺 Hipertensión</a>
                        <a href="/ai?q=diabetes+gestacional" class="quick-btn">🍯 Diabetes</a>
                        <a href="/ai?q=parto+cesárea" class="quick-btn">👶 Parto</a>
                        <a href="/ai?q=lactancia+medicamentos" class="quick-btn">🤱 Lactancia</a>
                        <a href="/ai?q=ultrasonido+embarazo" class="quick-btn">📊 Ultrasonido</a>
                    </div>
                </div>
            </div>
            
            <div class="features-section">
                <div class="features-grid">
                    <div class="feature-card action-card" onclick="window.location.href='/ask?q=bases+de+datos+vectoriales'">
                        <div class="feature-icon">🎯</div>
                        <div class="feature-title">Solo Búsqueda (/ask)</div>
                        <div class="feature-desc">Búsqueda semántica sin respuesta de IA</div>
                    </div>
                    
                    <div class="feature-card action-card" onclick="window.location.href='/ai?q=qué+es+la+preeclampsia'">
                        <div class="feature-icon">🤖</div>
                        <div class="feature-title">Respuesta con IA (/ai)</div>
                        <div class="feature-desc">Búsqueda + respuesta generada por IA</div>
                    </div>
                    
                    <div class="feature-card action-card" onclick="window.location.href='/compare?q=embarazo+diabetes'">
                        <div class="feature-icon">⚖️</div>
                        <div class="feature-title">Comparar Motores</div>
                        <div class="feature-desc">Qdrant vs PostgreSQL+pgvector</div>
                    </div>
                    
                    <div class="feature-card action-card" onclick="window.location.href='/docs'">
                        <div class="feature-icon">📖</div>
                        <div class="feature-title">Documentación API</div>
                        <div class="feature-desc">OpenAPI/Swagger docs completas</div>
                    </div>
                </div>
            </div>
            
            <div class="nav-section">
                <div class="nav-buttons">
                    <a href="/ai?q=evaluación+del+curso" class="nav-link primary">🎓 Búsquedas Académicas</a>
                    <a href="/filters/examples" class="nav-link">📋 Ejemplos de Filtros</a>
                    <a href="/manual/demo" class="nav-link">🎓 Demo Educativa</a>
                </div>
            </div>
        </div>
        
        <script>
            function searchWithAI() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    window.location.href = `/ai?q=${{encodeURIComponent(query.trim())}}`;
                }}
            }}
            
            function searchOnly() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    window.location.href = `/ask?q=${{encodeURIComponent(query.trim())}}`;
                }}
            }}
            
            // Allow Enter key to search
            document.getElementById('searchInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    searchWithAI();
                }}
            }});
            
            // Add some example text on page load
            document.addEventListener('DOMContentLoaded', function() {{
                const examples = [
                    "¿Cuáles son las vacunas recomendadas para embarazadas?",
                    "¿Qué es la preeclampsia?",
                    "Tratamiento de diabetes gestacional",
                    "Complicaciones del parto prematuro",
                    "Medicamentos seguros en el embarazo"
                ];
                
                const input = document.getElementById('searchInput');
                let currentExample = 0;
                
                function cycleExamples() {{
                    input.placeholder = examples[currentExample];
                    currentExample = (currentExample + 1) % examples.length;
                }}
                
                setInterval(cycleExamples, 3000);
            }});
        </script>
    </body>
    </html>
    """
