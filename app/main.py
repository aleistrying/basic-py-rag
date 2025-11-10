from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import json
import logging
from app.rag import rag_answer, generate_llm_answer

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

# Setup logger
logger = logging.getLogger(__name__)

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
            doc_name = source.get("document", "").replace(
                "data/raw/", "").replace(".pdf", "")
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
        <title>AI Response: {query}</title>
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
            .controls-row {{
                display: flex;
                gap: 20px;
                margin-top: 15px;
                margin-bottom: 15px;
                align-items: center;
            }}
            .control-group {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .control-group label {{
                color: rgba(255,255,255,0.9);
                font-size: 14px;
                font-weight: 500;
                min-width: 50px;
            }}
            .select-control {{
                padding: 8px 12px;
                border: none;
                border-radius: 6px;
                background: rgba(255,255,255,0.1);
                color: white;
                font-size: 14px;
                min-width: 120px;
                cursor: pointer;
            }}
            .select-control option {{
                background: #1a1a1a;
                color: white;
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
                white-space: pre-wrap;
                word-wrap: break-word;
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
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'Courier New', 'Monaco', 'Menlo', monospace;
                font-size: 13px;
                line-height: 1.5;
                text-align: left;
                white-space: pre-wrap;
                word-wrap: break-word;
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
                    <button class="search-btn" onclick="newSearch()">{get_svg_icon("search", "16", "#ffffff")} Buscar</button>
                    <button class="nav-btn" onclick="window.location.href='/ask?q=bases+de+datos+vectoriales'">{get_svg_icon("target", "16", "#ffffff")} Solo /ask</button>
                    <button class="nav-btn" onclick="window.location.href='/'">{get_svg_icon("home", "16", "#ffffff")} Menu Principal</button>
                </div>
                <div class="controls-row">
                    <div class="control-group">
                        <label for="backendSelect">Motor:</label>
                        <select id="backendSelect" class="select-control">
                            <option value="qdrant" {"selected" if backend.lower() == "qdrant" else ""}>Qdrant</option>
                            <option value="pgvector" {"selected" if backend.lower() == "pgvector" else ""}>PostgreSQL</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="modelSelect">Modelo:</label>
                        <select id="modelSelect" class="select-control">
                            <option value="phi3:mini" {"selected" if model == "phi3:mini" else ""}>Phi3 Mini</option>
                            <option value="llama3.1:8b" {"selected" if model == "llama3.1:8b" else ""}>Llama 3.1 8B</option>
                            <option value="gemma2:2b" {"selected" if model == "gemma2:2b" else ""}>Gemma2 2B</option>
                            <option value="qwen2.5:3b" {"selected" if model == "qwen2.5:3b" else ""}>Qwen2.5 3B</option>
                        </select>
                    </div>
                </div>
                <div class="title">{get_svg_icon("robot", "32", "#ffffff")} Respuesta AI</div>
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
                    <button class="toggle-btn" onclick="toggleDetails()">{get_svg_icon("settings", "16", "#ffffff")} Ver detalles técnicos JSON</button>
                    <div class="json-details" id="jsonDetails">
                        <div class="json-content" id="jsonContent">{json_str}</div>
                    </div>
                </div>
                
                <div class="navigation">
                    <div class="nav-group">
                        <a href="/" class="nav-link primary">{get_svg_icon("home", "16", "#3b82f6")} Inicio</a>
                        <a href="/ask?q=evaluación+del+curso" class="nav-link">{get_svg_icon("clipboard", "16", "#3b82f6")} Búsquedas Rápidas</a>
                    </div>
                    <div class="nav-group">
                        <a href="/docs" class="nav-link">{get_svg_icon("book", "16", "#3b82f6")} API Docs</a>
                        <a href="/compare?q={query}" class="nav-link">{get_svg_icon("balance", "16", "#3b82f6")} Comparar Motores</a>
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
                    btn.innerHTML = `{get_svg_icon("settings", "16", "#ffffff")} Ver detalles técnicos JSON`;
                }} else {{
                    details.classList.add('show');
                    btn.innerHTML = `{get_svg_icon("settings", "16", "#ffffff")} Ocultar detalles JSON`;
                    
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
                
                // Highlight numbers (simplified to avoid f-string backslash issues)
                html = html.replace(/: ([0-9]+\.?[0-9]*)/g, ': <span class="number">$1</span>');
                
                // Highlight booleans and null
                html = html.replace(/: (true|false)/g, ': <span class="boolean">$1</span>');
                html = html.replace(/: (null)/g, ': <span class="null">$1</span>');
                
                jsonContent.innerHTML = html;
            }}
            
            function newSearch() {{
                const query = document.getElementById('searchInput').value;
                const backend = document.getElementById('backendSelect').value;
                const model = document.getElementById('modelSelect').value;
                
                if (query.trim()) {{
                    const params = new URLSearchParams({{
                        q: query.trim(),
                        backend: backend,
                        model: model
                    }});
                    window.location.href = `/ai?${{params.toString()}}`;
                }}
            }}
            
            // Handle backend/model changes
            function updateSettings() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    newSearch();
                }}
            }}
            
            // Add event listeners
            document.getElementById('backendSelect').addEventListener('change', updateSettings);
            document.getElementById('modelSelect').addEventListener('change', updateSettings);
            
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


def enhanced_general_response_html(data: dict, title: str, theme_color: str = "#2563eb") -> str:
    """Enhanced HTML response for any type of data with customizable theme"""

    # Build content based on data type
    content_html = ""

    # If it's a demonstration or has structured data
    if "pasos" in data or "proceso" in data or "titulo" in data:
        content_html = build_demo_content(data)
    elif "sources" in data:
        content_html = build_sources_content(data)
    elif "error" in data:
        content_html = build_error_content(data)
    else:
        content_html = build_generic_content(data)

    # JSON details (expandable)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
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
                background: linear-gradient(135deg, {theme_color} 0%, {adjust_color(theme_color)} 100%);
                padding: 20px 30px;
                color: white;
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
            .section {{
                background: #242424;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid {theme_color};
            }}
            .section-title {{
                font-weight: 600;
                color: {theme_color};
                margin-bottom: 12px;
                font-size: 18px;
            }}
            .step-card {{
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 15px;
                transition: all 0.3s;
            }}
            .step-card:hover {{
                border-color: {theme_color};
                transform: translateY(-2px);
            }}
            .error-section {{
                background: #7f1d1d;
                color: #fef2f2;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid #dc2626;
            }}
            .details-toggle {{
                margin-top: 30px;
            }}
            .toggle-btn {{
                background: #374151;
                color: #e5e7eb;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s;
                width: 100%;
            }}
            .toggle-btn:hover {{
                background: #4b5563;
                transform: translateY(-1px);
            }}
            .json-details {{
                display: none;
                margin-top: 15px;
                border: 1px solid #374151;
            }}
            .json-details.show {{ display: block; }}
            .json-content {{
                background: #0f172a;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'Courier New', 'Monaco', 'Menlo', monospace;
                font-size: 13px;
                line-height: 1.5;
                text-align: left;
                white-space: pre-wrap;
                word-wrap: break-word;
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
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s;
            }}
            .nav-link:hover {{
                background: #4b5563;
                transform: translateY(-1px);
            }}
            .nav-link.primary {{
                background: {theme_color};
                color: white;
            }}
            .nav-link.primary:hover {{
                background: {adjust_color(theme_color)};
            }}
            
            /* Enhanced demo content styles */
            .summary {{
                font-size: 16px;
                opacity: 0.9;
                margin: 10px 0;
                font-style: italic;
            }}
            .step-header {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 12px;
            }}
            .step-number {{
                background: {theme_color};
                color: white;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 14px;
            }}
            .step-title {{
                margin: 0;
                color: #e5e7eb;
                font-size: 16px;
            }}
            .step-content {{
                margin-left: 45px;
            }}
            .query-display {{
                background: #1f2937;
                padding: 12px;
                border-radius: 6px;
                margin: 8px 0;
                border-left: 3px solid {theme_color};
            }}
            .info-note {{
                background: #065f46;
                color: #d1fae5;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
                margin: 8px 0;
            }}
            .vector-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 10px 0;
            }}
            .stat {{
                background: #374151;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 14px;
            }}
            .vector-sample {{
                background: #0f172a;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 12px;
                overflow-x: auto;
                margin: 8px 0;
            }}
            .results-list {{
                margin-top: 15px;
            }}
            .result-item {{
                background: #1f2937;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 10px;
                border-left: 3px solid {theme_color};
            }}
            .result-score {{
                color: {theme_color};
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 8px;
            }}
            .result-content {{
                margin: 8px 0;
                line-height: 1.4;
            }}
            .result-source {{
                font-size: 12px;
                opacity: 0.7;
                margin-top: 8px;
            }}
            .info-highlight {{
                background: linear-gradient(135deg, #065f46 0%, #047857 100%);
                color: #d1fae5;
                padding: 15px 20px;
                border-radius: 8px;
                font-weight: 500;
            }}
            .next-step {{
                background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                font-weight: 500;
            }}
            
            /* Enhanced demo content styles for comprehensive display */
            .explanation {{
                background: #1f2937;
                padding: 12px 15px;
                border-radius: 6px;
                margin: 10px 0;
                font-style: italic;
                border-left: 3px solid {theme_color};
            }}
            .info-box {{
                background: #2d3748;
                border: 1px solid #4a5568;
                border-radius: 8px;
                padding: 15px;
                margin: 12px 0;
                transition: all 0.3s;
            }}
            .info-box:hover {{
                border-color: {theme_color};
                transform: translateX(5px);
            }}
            .info-label {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 600;
                color: {theme_color};
                margin-bottom: 8px;
                font-size: 14px;
            }}
            .info-value {{
                background: #1a202c;
                padding: 10px 12px;
                border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 13px;
                border-left: 3px solid #48bb78;
            }}
            .info-explanation {{
                background: #065f46;
                color: #d1fae5;
                padding: 10px 12px;
                border-radius: 6px;
                margin: 8px 0;
                font-size: 13px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .vector-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 10px;
                margin-top: 10px;
            }}
            .stat-item {{
                background: #374151;
                padding: 8px 12px;
                border-radius: 4px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .stat-label {{
                font-size: 12px;
                color: #9ca3af;
            }}
            .stat-value {{
                font-weight: 600;
                color: #f3f4f6;
            }}
            .vector-preview {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 15px;
                margin: 12px 0;
                font-family: 'Monaco', 'Menlo', monospace;
            }}
            .vector-section {{
                margin: 8px 0;
            }}
            .vector-label {{
                color: #64748b;
                font-size: 12px;
                margin-bottom: 4px;
            }}
            .vector-values {{
                color: #38bdf8;
                font-size: 11px;
                background: #0c1620;
                padding: 4px 8px;
                border-radius: 4px;
                display: block;
                margin: 4px 0;
                word-break: break-all;
            }}
            .vector-dots {{
                text-align: center;
                color: #64748b;
                margin: 8px 0;
                font-style: italic;
                font-size: 12px;
            }}
            .engine-details {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 8px;
                margin-top: 10px;
            }}
            .results-summary {{
                background: #065f46;
                color: #d1fae5;
                padding: 10px 15px;
                border-radius: 6px;
                margin: 10px 0;
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 500;
            }}
            .results-count {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .similarity-guide {{
                background: #1e1b4b;
                color: #c7d2fe;
                padding: 10px;
                border-radius: 4px;
                margin-top: 8px;
                font-size: 12px;
                border-left: 3px solid #6366f1;
            }}
            .results-section {{
                margin: 15px 0;
            }}
            .results-header {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 600;
                color: {theme_color};
                margin-bottom: 12px;
                font-size: 16px;
            }}
            .result-card {{
                background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
                border: 1px solid #4a5568;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 12px;
                transition: all 0.3s;
            }}
            .result-card:hover {{
                border-color: {theme_color};
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }}
            .result-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 1px solid #4a5568;
            }}
            .result-position {{
                background: {theme_color};
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
            }}
            .result-similarity {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .similarity-score {{
                font-size: 18px;
                font-weight: bold;
                color: #48bb78;
            }}
            .similarity-percent {{
                font-size: 14px;
                color: #9ca3af;
            }}
            .content-preview {{
                background: #1a202c;
                padding: 12px;
                border-radius: 6px;
                margin: 10px 0;
                border-left: 3px solid #48bb78;
                font-style: italic;
                line-height: 1.5;
            }}
            .result-metadata {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 8px;
                margin: 10px 0;
            }}
            .metadata-item {{
                background: #374151;
                padding: 6px 10px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
            }}
            .metadata-label {{
                color: #9ca3af;
                display: flex;
                align-items: center;
                gap: 4px;
            }}
            .metadata-value {{
                color: #f3f4f6;
                font-weight: 500;
            }}
            .score-explanation {{
                background: #1e3a8a;
                color: #bfdbfe;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 11px;
                margin-top: 8px;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">{title}</div>
            </div>
            
            <div class="content">
                {content_html}
                
                <div class="details-toggle">
                    <button class="toggle-btn" onclick="toggleDetails()">{get_svg_icon("settings", "16", "#ffffff")} Ver detalles técnicos JSON</button>
                    <div class="json-details" id="jsonDetails">
                        <div class="json-content" id="jsonContent">{json_str}</div>
                    </div>
                </div>
                
                <div class="navigation">
                    <div class="nav-group">
                        <a href="/" class="nav-link primary">{get_svg_icon("home", "16", "#3b82f6")} Inicio</a>
                        <a href="/docs" class="nav-link">{get_svg_icon("book", "16", "#3b82f6")} API Docs</a>
                    </div>
                    <div class="nav-group">
                        <a href="/ask?q=bases+de+datos" class="nav-link">{get_svg_icon("clipboard", "16", "#3b82f6")} Búsqueda</a>
                        <a href="/ai?q=que+es+nosql" class="nav-link">{get_svg_icon("robot", "16", "#3b82f6")} Con IA</a>
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
                    btn.innerHTML = `{get_svg_icon("settings", "16", "#ffffff")} Ver detalles técnicos JSON`;
                }} else {{
                    details.classList.add('show');
                    btn.innerHTML = `{get_svg_icon("settings", "16", "#ffffff")} Ocultar detalles JSON`;
                }}
            }}
        </script>
    </body>
    </html>
    """


def adjust_color(color: str) -> str:
    """Adjust hex color to be slightly darker"""
    try:
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, c - 30) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
    except:
        return "#1e3a5f"


def get_svg_icon(name: str, size: str = "20", color: str = "currentColor") -> str:
    """Generate SVG icons"""
    icons = {
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>',
        "robot": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="m12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>',
        "target": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "compare": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m8 3 4 8 5-5v11H5l3-7 4 4"/></svg>',
        "rocket": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>',
        "brain": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.07 2.07 0 0 1-2.44-2.44 2.07 2.07 0 0 1-2.44-2.44 2.07 2.07 0 0 1 0-4.12A2.5 2.5 0 0 1 2.5 8 2.5 2.5 0 0 1 5 5.5a2.5 2.5 0 0 1 4.5-3.5z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.07 2.07 0 0 0 2.44-2.44 2.07 2.07 0 0 0 2.44-2.44 2.07 2.07 0 0 0 0-4.12A2.5 2.5 0 0 0 21.5 8 2.5 2.5 0 0 0 19 5.5a2.5 2.5 0 0 0-4.5-3.5z"/></svg>',
        "docs": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14,2 14,8 20,8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10,9 9,9 8,9"/></svg>',
        "gear": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
        "graduation": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m22 10-8-4-8 4 8 4 8-4v6c0 1-1 2-3 2s-3-1-3-2"/><path d="M6 10v9a1 1 0 0 0 1.6.8L12 18l4.4 1.8a1 1 0 0 0 1.6-.8v-9"/></svg>',
        "filter": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polygon points="22,3 2,3 10,12.46 10,19 14,21 14,12.46"/></svg>',
        "home": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9,22 9,12 15,12 15,22"/></svg>',
        "medical": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.29 1.51 4.04 3 5.5l7 7 7-7z"/></svg>',
        "chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
        "book": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
        "clipboard": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="m16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>',
        "balance": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m9 12 2 2 4-4"/><path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"/><path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"/><path d="m3 12h6m6 0h6"/></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="m12 1v6m0 6v6"/><path d="m20.49 9l-1.73 1-1.73-1 1.73-1z"/><path d="m20.49 15l-1.73 1-1.73-1 1.73-1z"/><path d="m3.51 9l1.73 1 1.73-1-1.73-1z"/><path d="m3.51 15l1.73 1 1.73-1-1.73-1z"/></svg>',
        "tools": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
        "lightbulb": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M9 21h6"/><path d="M12 3a6 6 0 0 1 6 6c0 3-2 5.5-2 8H8c0-2.5-2-5-2-8a6 6 0 0 1 6-6z"/><path d="M12 3V1"/></svg>',
        "calculator": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><rect x="4" y="2" width="16" height="20" rx="2"/><line x1="8" y1="6" x2="16" y2="6"/><line x1="8" y1="10" x2="8" y2="10.01"/><line x1="12" y1="10" x2="12" y2="10.01"/><line x1="16" y1="10" x2="16" y2="10.01"/><line x1="8" y1="14" x2="8" y2="14.01"/><line x1="12" y1="14" x2="12" y2="14.01"/><line x1="16" y1="14" x2="16" y2="14.01"/><line x1="8" y1="18" x2="8" y2="18.01"/><line x1="12" y1="18" x2="16" y2="18"/></svg>',
        "experiment": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M7 13v8a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-8"/><polyline points="7,9 12,15 17,9"/><polyline points="9,5 9,9"/><polyline points="15,5 15,9"/><circle cx="12" cy="5" r="2"/></svg>',
        "ruler": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="m21.73 6.27-10 10a1 1 0 0 1-1.41 0l-4.05-4.05a1 1 0 0 1 0-1.41l10-10a1 1 0 0 1 1.41 0l4.05 4.05a1 1 0 0 1 0 1.41z"/><line x1="7" y1="17" x2="17" y2="7"/></svg>',
        "edit": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="m18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
    }
    return icons.get(name, f'<span style="font-size: {size}px;">•</span>')


def adjust_color(color: str) -> str:
    """Adjust color for gradient effect"""
    color_map = {
        "#2563eb": "#1d4ed8",  # blue
        "#059669": "#047857",  # green
        "#7c3aed": "#6d28d9",  # purple
        "#dc2626": "#b91c1c",  # red
        "#ea580c": "#c2410c",  # orange
    }
    return color_map.get(color, "#1e40af")


def build_demo_content(data: dict) -> str:
    """Build comprehensive content for demonstration pages with detailed step information"""
    content = ""

    if "titulo" in data:
        content += f'<div class="section"><div class="section-title">{data["titulo"]}</div>'
        if "resumen" in data:
            content += f'<p class="summary">{data["resumen"]}</p>'
        content += '</div>'

    if "pasos" in data:
        content += f'<div class="section"><div class="section-title">{get_svg_icon("clipboard", "20", "#3b82f6")} Proceso Paso a Paso</div>'
        for i, paso in enumerate(data["pasos"], 1):
            paso_num = paso.get("paso", i)
            descripcion = paso.get("descripcion", "")
            explicacion = paso.get("explicacion", "")

            content += f'''
            <div class="step-card">
                <div class="step-header">
                    <div class="step-number">{paso_num}</div>
                    <h4 class="step-title">{descripcion}</h4>
                </div>
                <div class="step-content">
                    '''

            # Add explanation if available
            if explicacion:
                content += f'<div class="explanation">{explicacion}</div>'

            # Query information
            if "query_original" in paso:
                content += f'''
                <div class="info-box">
                    <div class="info-label">{get_svg_icon("search", "16", "#059669")} Consulta Original:</div>
                    <div class="info-value">"{paso["query_original"]}"</div>
                </div>
                '''

            if "query_expandida" in paso:
                content += f'''
                <div class="info-box">
                    <div class="info-label">{get_svg_icon("target", "16", "#7c3aed")} Consulta Expandida:</div>
                    <div class="info-value">"{paso["query_expandida"]}"</div>
                '''
                if paso.get("cambios"):
                    content += f'<div class="info-note">{get_svg_icon("gear", "14", "#6b7280")} {paso["cambios"]}</div>'
                content += '</div>'

            # Vector information
            if "vector_dimensiones" in paso:
                content += f'''
                <div class="info-box">
                    <div class="info-label">{get_svg_icon("brain", "16", "#dc2626")} Vector Generado:</div>
                    <div class="vector-stats">
                        <div class="stat-item">
                            <span class="stat-label">Dimensiones:</span>
                            <span class="stat-value">{paso["vector_dimensiones"]}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Modelo:</span>
                            <span class="stat-value">{paso.get("modelo_usado", "N/A")}</span>
                        </div>
                    </div>
                </div>
                '''

                if "primeros_10_valores" in paso and "ultimos_10_valores" in paso:
                    primeros = ", ".join(
                        [f"{v:.4f}" for v in paso["primeros_10_valores"]])
                    ultimos = ", ".join(
                        [f"{v:.4f}" for v in paso["ultimos_10_valores"]])
                    content += f'''
                    <div class="vector-preview">
                        <div class="vector-section">
                            <div class="vector-label">Primeros 10 valores:</div>
                            <code class="vector-values">[{primeros}]</code>
                        </div>
                        <div class="vector-dots">... {paso["vector_dimensiones"] - 20} valores más ...</div>
                        <div class="vector-section">
                            <div class="vector-label">Últimos 10 valores:</div>
                            <code class="vector-values">[{ultimos}]</code>
                        </div>
                    </div>
                    '''

                if paso.get("como_funciona"):
                    content += f'<div class="info-explanation">{get_svg_icon("graduation", "14", "#059669")} {paso["como_funciona"]}</div>'

            # Search engine information
            if "motor_usado" in paso:
                motor = paso["motor_usado"]
                content += f'''
                <div class="info-box">
                    <div class="info-label">{get_svg_icon("gear", "16", "#ea580c")} Motor de Búsqueda:</div>
                    <div class="engine-details">
                        <div class="stat-item">
                            <span class="stat-label">Nombre:</span>
                            <span class="stat-value">{motor.get("nombre", "")}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Tipo:</span>
                            <span class="stat-value">{motor.get("tipo", "")}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Algoritmo:</span>
                            <span class="stat-value">{motor.get("algoritmo", "")}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Métrica:</span>
                            <span class="stat-value">{motor.get("metrica", "")}</span>
                        </div>
                    </div>
                </div>
                '''

            if "proceso" in paso:
                content += f'<div class="info-explanation">{get_svg_icon("search", "14", "#3b82f6")} {paso["proceso"]}</div>'

            # Results information
            if "resultados_encontrados" in paso:
                content += f'''
                <div class="results-summary">
                    <span class="results-count">{get_svg_icon("chart", "16", "#059669")} {paso["resultados_encontrados"]} resultados encontrados</span>
                </div>
                '''

            # Similarity ranking details
            if "metrica_usada" in paso:
                content += f'''
                <div class="info-box">
                    <div class="info-label">{get_svg_icon("chart", "16", "#7c3aed")} Métrica de Similaridad:</div>
                    <div class="info-value">{paso["metrica_usada"]}</div>
                '''
                if paso.get("interpretacion"):
                    content += f'<div class="similarity-guide">{paso["interpretacion"]}</div>'
                content += '</div>'

            # Detailed results
            if "resultados_ordenados" in paso:
                content += f'''
                <div class="results-section">
                    <div class="results-header">{get_svg_icon("filter", "16", "#dc2626")} Resultados Ordenados por Similaridad:</div>
                    <div class="results-list">
                '''
                for resultado in paso["resultados_ordenados"]:
                    content += f'''
                    <div class="result-card">
                        <div class="result-header">
                            <div class="result-position">#{resultado.get("posicion", "")}</div>
                            <div class="result-similarity">
                                <span class="similarity-score">{resultado.get("similaridad", "0.000")}</span>
                                <span class="similarity-percent">({resultado.get("similaridad_porcentage", "0%")})</span>
                            </div>
                        </div>
                        <div class="result-content">
                            <div class="content-preview">{resultado.get("contenido_preview", "")}</div>
                            <div class="result-metadata">
                                <div class="metadata-item">
                                    <span class="metadata-label">{get_svg_icon("docs", "12", "#6b7280")} Documento:</span>
                                    <span class="metadata-value">{resultado.get("documento", "").split('/')[-1]}</span>
                                </div>
                                <div class="metadata-item">
                                    <span class="metadata-label">Página:</span>
                                    <span class="metadata-value">{resultado.get("pagina", "")}</span>
                                </div>
                                <div class="metadata-item">
                                    <span class="metadata-label">Chunk ID:</span>
                                    <span class="metadata-value">{resultado.get("chunk_id", "")}</span>
                                </div>
                            </div>
                            {f'<div class="score-explanation">{resultado.get("explicacion_score", "")}</div>' if resultado.get("explicacion_score") else ""}
                        </div>
                    </div>
                    '''
                content += '</div></div>'

            content += '</div></div>'
        content += '</div>'

    # Add additional sections
    if "vector_completo_disponible" in data:
        content += f'<div class="section"><div class="info-highlight">{data["vector_completo_disponible"]}</div></div>'

    if "siguiente_paso" in data:
        content += f'<div class="section"><div class="next-step">{get_svg_icon("lightbulb", "16", "#fbbf24")} {data["siguiente_paso"]}</div></div>'

    return content


def build_sources_content(data: dict) -> str:
    """Build content for search results"""
    content = ""
    sources = data.get("sources", [])

    if sources:
        content += f'<div class="section"><div class="section-title">{get_svg_icon("clipboard", "16", "#3b82f6")} Resultados Encontrados</div>'
        for i, source in enumerate(sources, 1):
            content += f'''
            <div class="step-card">
                <h4>{i}. {source.get("reference", "")}</h4>
                <p>{source.get("preview", "")}</p>
            </div>
            '''
        content += '</div>'

    return content


def build_error_content(data: dict) -> str:
    """Build content for error pages"""
    return f'''
    <div class="error-section">
        <h3>❌ Error</h3>
        <p>{data.get("error", "Error desconocido")}</p>
    </div>
    '''


def build_generic_content(data: dict) -> str:
    """Build content for generic data"""
    content = ""

    # Try to display key information nicely
    for key, value in data.items():
        if key not in ['error', 'status']:
            content += f'''
            <div class="section">
                <div class="section-title">{key.replace('_', ' ').title()}</div>
                <p>{str(value)[:200]}{'...' if len(str(value)) > 200 else ''}</p>
            </div>
            '''

    return content


def enhanced_search_response_html(data: dict, query: str) -> str:
    """Enhanced HTML for search responses (similar to AI but without LLM response)"""

    # Extract key information
    sources = data.get("sources", [])
    total_results = data.get("total_results", 0)
    backend = data.get("backend", "")

    # Build sources section with page references
    sources_html = ""
    if sources:
        sources_html = '<div class="sources-section">'
        sources_html += f'<div class="sources-title">{get_svg_icon("clipboard", "16", "#3b82f6")} Documentos Encontrados:</div>'
        sources_html += '<div class="sources-grid">'

        for i, source in enumerate(sources, 1):
            reference = source.get("reference", "")
            preview = source.get("preview", "")
            similarity = source.get("similarity", "")

            sources_html += f'''
            <div class="source-card">
                <div class="source-header">
                    <span class="source-number">{i}</span>
                    <span class="source-title">{reference}</span>
                    <span class="similarity-score">Similitud: {similarity}</span>
                </div>
                <div class="source-preview">{preview}</div>
            </div>
            '''

        sources_html += '</div></div>'

    # JSON details (expandable)
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results: {query}</title>
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
                background: linear-gradient(135deg, #059669 0%, #047857 100%);
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
                border-left: 4px solid #059669;
            }}
            .question-label {{
                font-weight: 600;
                color: #059669;
                margin-bottom: 8px;
                font-size: 16px;
            }}
            .question-text {{
                font-size: 18px;
                line-height: 1.5;
            }}
            .sources-section {{
                margin-bottom: 30px;
            }}
            .sources-title {{
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 20px;
                color: #e1e1e1;
            }}
            .sources-grid {{
                display: grid;
                gap: 15px;
            }}
            .source-card {{
                background: #2a2a2a;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 16px;
                transition: all 0.3s;
            }}
            .source-card:hover {{
                border-color: #059669;
                transform: translateY(-2px);
            }}
            .source-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 10px;
                flex-wrap: wrap;
            }}
            .source-number {{
                background: #059669;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: 600;
                flex-shrink: 0;
            }}
            .source-title {{
                font-weight: 600;
                color: #e1e1e1;
                flex: 1;
                min-width: 200px;
            }}
            .similarity-score {{
                background: #374151;
                color: #9ca3af;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            .source-preview {{
                color: #b1b1b1;
                font-size: 14px;
                line-height: 1.5;
                border-left: 3px solid #404040;
                padding-left: 12px;
            }}
            .details-toggle {{
                margin-top: 30px;
            }}
            .toggle-btn {{
                background: #374151;
                color: #e5e7eb;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s;
                width: 100%;
            }}
            .toggle-btn:hover {{
                background: #4b5563;
                transform: translateY(-1px);
            }}
            .json-details {{
                display: none;
                margin-top: 15px;
                border: 1px solid #374151;
            }}
            .json-details.show {{ display: block; }}
            .json-content {{
                background: #0f172a;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'Courier New', 'Monaco', 'Menlo', monospace;
                font-size: 13px;
                line-height: 1.5;
                text-align: left;
                white-space: pre-wrap;
                word-wrap: break-word;
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
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s;
            }}
            .nav-link:hover {{
                background: #4b5563;
                transform: translateY(-1px);
            }}
            .nav-link.primary {{
                background: #059669;
                color: white;
            }}
            .nav-link.primary:hover {{
                background: #047857;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="search-bar">
                    <input type="text" class="search-input" placeholder="Nueva búsqueda..." value="{query}" id="searchInput">
                    <button class="search-btn" onclick="newSearch()">{get_svg_icon("search", "16", "#ffffff")} Buscar</button>
                    <button class="nav-btn" onclick="window.location.href='/ai?q={query}'">{get_svg_icon("robot", "16", "#ffffff")} Con IA</button>
                    <button class="nav-btn" onclick="window.location.href='/'">{get_svg_icon("home", "16", "#ffffff")} Menu Principal</button>
                </div>
                <div class="title">{get_svg_icon("clipboard", "32", "#ffffff")} Búsqueda Semántica</div>
                <div class="subtitle">Resultados: {total_results} | Motor: {backend}</div>
            </div>
            
            <div class="content">
                <div class="question-section">
                    <div class="question-label">Q:</div>
                    <div class="question-text">{query}</div>
                </div>
                
                {sources_html}
                
                <div class="details-toggle">
                    <button class="toggle-btn" onclick="toggleDetails()">{get_svg_icon("settings", "16", "#ffffff")} Ver detalles técnicos JSON</button>
                    <div class="json-details" id="jsonDetails">
                        <div class="json-content" id="jsonContent">{json_str}</div>
                    </div>
                </div>
                
                <div class="navigation">
                    <div class="nav-group">
                        <a href="/" class="nav-link primary">{get_svg_icon("home", "16", "#3b82f6")} Inicio</a>
                        <a href="/ai?q={query}" class="nav-link">{get_svg_icon("robot", "16", "#3b82f6")} Respuesta con IA</a>
                    </div>
                    <div class="nav-group">
                        <a href="/docs" class="nav-link">{get_svg_icon("book", "16", "#3b82f6")} API Docs</a>
                        <a href="/compare?q={query}" class="nav-link">{get_svg_icon("balance", "16", "#3b82f6")} Comparar Motores</a>
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
                    btn.innerHTML = `{get_svg_icon("tools", "16", "#ffffff")} Ver detalles técnicos JSON`;
                }} else {{
                    details.classList.add('show');
                    btn.innerHTML = `{get_svg_icon("tools", "16", "#ffffff")} Ocultar detalles JSON`;
                }}
            }}
            
            function newSearch() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    window.location.href = `/ask?q=${{encodeURIComponent(query.trim())}}`;
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
            <div class="title">{get_svg_icon("rocket", "24", "#569cd6")} {title}</div>
            <pre id="json-content">{json_str}</pre>
            <a href="/" class="back-link">← Volver al inicio</a>
            <a href="/docs" class="back-link">{get_svg_icon("book", "16", "#569cd6")} Documentación API</a>
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
            return enhanced_search_response_html(result, q)
    except ValueError as e:
        error_data = {"error": str(e), "status": 400}
        if format == "json":
            raise HTTPException(status_code=400, detail=str(e))
        return enhanced_general_response_html(error_data, "❌ Error", "#dc2626")
    except Exception as e:
        error_data = {
            "error": f"Internal server error: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}")
        return enhanced_general_response_html(error_data, "❌ Error", "#dc2626")


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
        return enhanced_general_response_html(error_data, "❌ Error LLM", "#dc2626")
    except Exception as e:
        error_data = {"error": f"AI error: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
        return enhanced_general_response_html(error_data, "❌ Error AI", "#dc2626")


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
            return enhanced_general_response_html(result, f"{get_svg_icon('balance', '20', '#059669')} Comparación: {q}", "#059669")
    except Exception as e:
        error_data = {"error": f"Comparison error: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Comparison error: {str(e)}")
        return enhanced_general_response_html(error_data, "❌ Error Comparación", "#dc2626")


@app.get("/manual/embed", response_class=HTMLResponse)
def manual_embed_demo(
    q: str = Query("bases de datos vectoriales",
                   description="Texto a vectorizar"),
    format: str = Query("html", description="Formato: 'json' o 'html'")
):
    """Demostración manual del proceso de vectorización paso a paso"""
    try:
        # Paso 1: Mostrar query original
        step1 = {
            "paso": 1,
            "descripcion": f"{get_svg_icon('edit', '16', '#059669')} Consulta original del usuario",
            "query_original": q,
            "explicacion": "Esta es la consulta en lenguaje natural que quiere hacer el usuario"
        }

        # Paso 2: Expansión de consulta
        expanded = expand_query(q)
        step2 = {
            "paso": 2,
            "descripcion": f"{get_svg_icon('target', '16', '#7c3aed')} Expansión y normalización de la consulta",
            "query_expandida": expanded,
            "cambios": "Se mejora la consulta para mejor recuperación" if expanded != q else "No se necesitan cambios",
            "explicacion": "Se procesan sinónimos y términos relacionados para mejor búsqueda"
        }

        # Paso 3: Vectorización
        embedding = embed_e5([expanded], is_query=True)[0]
        step3 = {
            "paso": 3,
            "descripcion": f"{get_svg_icon('calculator', '16', '#2563eb')} Conversión a vector numérico (embedding)",
            "vector_dimensiones": len(embedding),
            "modelo_usado": "intfloat/multilingual-e5-base",
            "primeros_10_valores": embedding[:10],
            "ultimos_10_valores": embedding[-10:],
            "explicacion": "El texto se convierte en un vector de 768 números que representa su significado semántico",
            "como_funciona": "El modelo E5 fue entrenado para que textos similares tengan vectores similares"
        }

        result = {
            "titulo": f"{get_svg_icon('tools', '20', '#ea580c')} PROCESO MANUAL DE VECTORIZACIÓN",
            "resumen": "Conversión de texto natural a vector numérico para búsqueda semántica",
            "pasos": [step1, step2, step3],
            "vector_completo_disponible": f"Vector completo de {len(embedding)} dimensiones generado exitosamente",
            "siguiente_paso": "Usar este vector para buscar documentos similares con /manual/search"
        }

        if format == "json":
            return result
        else:
            return enhanced_general_response_html(result, f"{get_svg_icon('tools', '20', '#ea580c')} Vectorización: {q}", "#ea580c")

    except Exception as e:
        error_data = {
            "error": f"Error en demostración: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en demostración: {str(e)}")
        return enhanced_general_response_html(error_data, "❌ Error Demo", "#dc2626")


@app.get("/manual/search", response_class=HTMLResponse)
def manual_search_demo(
    q: str = Query("bases de datos vectoriales",
                   description="Consulta para buscar"),
    backend: str = Query("qdrant", description="Motor de búsqueda"),
    k: int = Query(3, description="Número de resultados"),
    format: str = Query("html", description="Formato: 'json' o 'html'")
):
    """Demostración manual del proceso de búsqueda vectorial paso a paso"""
    try:

        # Paso 1: Vectorización de la consulta
        expanded = expand_query(q)
        embedding = embed_e5([expanded], is_query=True)[0]

        step1 = {
            "paso": 1,
            "descripcion": f"{get_svg_icon('calculator', '16', '#2563eb')} Vectorización de la consulta",
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
            "descripcion": f"{get_svg_icon('search', '16', '#059669')} Búsqueda en la base de datos vectorial",
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
            "descripcion": f"{get_svg_icon('chart', '16', '#7c3aed')} Ranking por similaridad",
            "explicacion": "Los resultados se ordenan por similaridad coseno (0.0 a 1.0)",
            "metrica_usada": "Similaridad coseno: mide el ángulo entre vectores",
            "interpretacion": "0.9+ = Muy similar, 0.7-0.9 = Similar, 0.5-0.7 = Algo relacionado, <0.5 = Poco relacionado",
            "resultados_ordenados": resultados_detallados
        }

        result = {
            "titulo": f"{get_svg_icon('search', '20', '#2563eb')} PROCESO MANUAL DE BÚSQUEDA VECTORIAL",
            "resumen": f"Búsqueda semántica usando {motor_info['nombre']} con {len(results)} resultados",
            "pasos": [step1, step2, step3],
            "conclusion": "Los vectores permiten encontrar documentos por significado, no solo por palabras exactas"
        }

        if format == "json":
            return result
        else:
            return enhanced_general_response_html(result, f"{get_svg_icon('search', '20', '#2563eb')} Búsqueda Manual: {q}", "#2563eb")

    except Exception as e:
        error_data = {
            "error": f"Error en búsqueda manual: {str(e)}", "status": 500}
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Error en búsqueda manual: {str(e)}")
        return enhanced_general_response_html(error_data, "❌ Error Búsqueda", "#dc2626")


@app.get("/filters/examples", response_class=HTMLResponse)
def filter_examples(format: str = Query("html", description="Formato: 'json' o 'html'")):
    """Ejemplos de filtros de metadata disponibles"""
    result = {
        "titulo": f"{get_svg_icon('clipboard', '20', '#059669')} EJEMPLOS DE FILTROS DE METADATA",
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
        return enhanced_general_response_html(result, f"{get_svg_icon('clipboard', '20', '#059669')} Ejemplos de Filtros", "#059669")


@app.get("/", response_class=HTMLResponse)
def root(format: str = Query("html", description="Formato: 'json' o 'html'")):
    """Enhanced home page with search interface and quick actions"""
    if format == "json":
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
                "Smart chunking (200 tokens, preserves context)"
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
        <title>RAG Demo - Main Menu</title>
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
            
            /* Loading Animations */
            .loading-overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                z-index: 9999;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }}
            
            .loading-overlay.active {{
                display: flex;
            }}
            
            .spinner {{
                width: 50px;
                height: 50px;
                border: 4px solid #374151;
                border-top: 4px solid #2563eb;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .loading-text {{
                color: #e5e7eb;
                font-size: 16px;
                margin-top: 10px;
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ opacity: 0.7; }}
                50% {{ opacity: 1; }}
            }}
            
            /* Button loading states */
            .btn-loading {{
                position: relative;
                pointer-events: none;
                opacity: 0.7;
            }}
            
            .btn-loading::after {{
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 16px;
                height: 16px;
                margin: -8px 0 0 -8px;
                border: 2px solid transparent;
                border-top: 2px solid currentColor;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            
            /* Form loading animation */
            .search-loading {{
                display: none;
                align-items: center;
                gap: 8px;
                margin-top: 10px;
                color: #2563eb;
                font-size: 14px;
            }}
            
            .search-loading.active {{
                display: flex;
            }}
            
            .dots {{
                display: inline-flex;
                gap: 2px;
            }}
            
            .dot {{
                width: 4px;
                height: 4px;
                background: #2563eb;
                border-radius: 50%;
                animation: dot-bounce 1.4s infinite ease-in-out;
            }}
            
            .dot:nth-child(1) {{ animation-delay: -0.32s; }}
            .dot:nth-child(2) {{ animation-delay: -0.16s; }}
            .dot:nth-child(3) {{ animation-delay: 0s; }}
            
            @keyframes dot-bounce {{
                0%, 80%, 100% {{ 
                    transform: scale(0);
                }} 
                40% {{ 
                    transform: scale(1);
                }}
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
            
            .api-routes-section {{
                padding: 30px;
                background: #0f172a;
                border-top: 1px solid #1e293b;
            }}
            .section-title {{
                text-align: center;
                font-size: 24px;
                color: #e1e7ef;
                margin-bottom: 30px;
                font-weight: 600;
            }}
            .api-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
                max-width: 1000px;
                margin: 0 auto;
            }}
            .api-group {{
                background: #1e293b;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
                display: flex;
                flex-direction: column;
            }}
            .api-group h4 {{
                color: #3b82f6;
                margin: 0 0 15px 0;
                font-size: 16px;
                border-bottom: 2px solid #3b82f6;
                padding-bottom: 8px;
            }}
            .api-btn {{
                display: block;
                padding: 12px 16px;
                margin-bottom: 8px;
                text-decoration: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s;
                border: 1px solid transparent;
            }}
            .api-btn:last-child {{ margin-bottom: 0; }}
            
            .search-btn {{ background: #059669; color: white; }}
            .search-btn:hover {{ background: #047857; transform: translateY(-1px); }}
            
            .ai-btn {{ background: #2563eb; color: white; }}
            .ai-btn:hover {{ background: #1d4ed8; transform: translateY(-1px); }}
            
            .compare-btn {{ background: #7c3aed; color: white; }}
            .compare-btn:hover {{ background: #6d28d9; transform: translateY(-1px); }}
            
            .demo-btn {{ background: #ea580c; color: white; }}
            .demo-btn:hover {{ background: #c2410c; transform: translateY(-1px); }}
            
            .embed-btn {{ background: #dc2626; color: white; }}
            .embed-btn:hover {{ background: #b91c1c; transform: translateY(-1px); }}
            
            .manual-btn {{ background: #0891b2; color: white; }}
            .manual-btn:hover {{ background: #0e7490; transform: translateY(-1px); }}
            
            .filter-btn {{ background: #16a34a; color: white; }}
            .filter-btn:hover {{ background: #15803d; transform: translateY(-1px); }}
            
            .docs-btn {{ background: #6366f1; color: white; }}
            .docs-btn:hover {{ background: #4f46e5; transform: translateY(-1px); }}
            
            .json-btn {{ background: #374151; color: #e5e7eb; }}
            .json-btn:hover {{ background: #4b5563; transform: translateY(-1px); }}
            
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
        <!-- Global loading overlay -->
        <div class="loading-overlay" id="loadingOverlay">
            <div class="spinner"></div>
            <div class="loading-text">Procesando consulta...</div>
        </div>
        
        <div class="container">
            <div class="header">
                <div class="title">{get_svg_icon("rocket", "32", "#ffffff")} RAG Demo</div>
                <div class="subtitle">Sistema de Búsqueda Inteligente con IA</div>
            </div>
            
            <div class="search-section">
                <div class="search-title">{get_svg_icon("search", "24")} Buscar en Documentos</div>
                <div class="search-bar">
                    <input type="text" class="search-input" placeholder="Ej: ¿Cuáles son las vacunas recomendadas para embarazadas?" id="searchInput">
                    <button class="search-btn" onclick="searchWithAI()" id="searchButton">{get_svg_icon("robot", "18")} AI Search</button>
                </div>
                
                <!-- Loading indicator -->
                <div class="search-loading" id="searchLoading">
                    <span>Buscando</span>
                    <div class="dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
                
                <div class="quick-searches">
                    <div class="quick-title">{get_svg_icon("target", "20")} Búsquedas Rápidas:</div>
                    <div class="quick-buttons">
                        <a href="/ai?q=vacunas+embarazadas" class="quick-btn">{get_svg_icon("medical", "16")} Vacunas</a>
                        <a href="/ai?q=hipertensión+embarazo" class="quick-btn">{get_svg_icon("medical", "16")} Hipertensión</a>
                        <a href="/ai?q=diabetes+gestacional" class="quick-btn">{get_svg_icon("medical", "16")} Diabetes</a>
                        <a href="/ai?q=parto+cesárea" class="quick-btn">{get_svg_icon("medical", "16")} Parto</a>
                        <a href="/ai?q=lactancia+medicamentos" class="quick-btn">{get_svg_icon("medical", "16")} Lactancia</a>
                        <a href="/ai?q=ultrasonido+embarazo" class="quick-btn">{get_svg_icon("chart", "16")} Ultrasonido</a>
                    </div>
                </div>
            </div>
            
            <div class="features-section">
                <div class="features-grid">
                    <div class="feature-card action-card" onclick="window.location.href='/ask?q=bases+de+datos+vectoriales'">
                        <div class="feature-icon">{get_svg_icon("target", "32")}</div>
                        <div class="feature-title">Solo Búsqueda (/ask)</div>
                        <div class="feature-desc">Búsqueda semántica sin respuesta de IA</div>
                    </div>
                    
                    <div class="feature-card action-card" onclick="window.location.href='/ai?q=qué+es+la+preeclampsia'">
                        <div class="feature-icon">{get_svg_icon("robot", "32")}</div>
                        <div class="feature-title">Respuesta con IA (/ai)</div>
                        <div class="feature-desc">Búsqueda + respuesta generada por IA</div>
                    </div>
                    
                    <div class="feature-card action-card" onclick="window.location.href='/compare?q=embarazo+diabetes'">
                        <div class="feature-icon">{get_svg_icon("compare", "32")}</div>
                        <div class="feature-title">Comparar Motores</div>
                        <div class="feature-desc">Qdrant vs PostgreSQL+pgvector</div>
                    </div>
                    
                    <div class="feature-card action-card" onclick="window.location.href='/docs'">
                        <div class="feature-icon">{get_svg_icon("book", "32", "#2563eb")}</div>
                        <div class="feature-title">Documentación API</div>
                        <div class="feature-desc">OpenAPI/Swagger docs completas</div>
                    </div>
                </div>
            </div>
            
            <div class="api-routes-section">
                <div class="section-title">{get_svg_icon("settings", "24", "#ffffff")} Todas las Rutas API</div>
                <div class="api-grid">
                    <div class="api-group">
                        <h4>{get_svg_icon("clipboard", "20", "#3b82f6")} Búsqueda y Consultas</h4>
                        <a href="/ask?q=bases+de+datos" class="api-btn search-btn">{get_svg_icon("clipboard", "16", "#ffffff")} /ask - Búsqueda Semántica</a>
                        <a href="/ai?q=que+es+nosql" class="api-btn ai-btn">{get_svg_icon("robot", "16", "#ffffff")} /ai - Búsqueda + IA</a>
                        <a href="/compare?q=vectores" class="api-btn compare-btn">{get_svg_icon("balance", "16", "#ffffff")} /compare - Comparar Motores</a>
                    </div>
                    
                    <div class="api-group">
                        <h4>{get_svg_icon("graduation", "20", "#3b82f6")} Demos Educativas</h4>
                        <a href="/demo/pipeline?q=pgvector" class="api-btn demo-btn">{get_svg_icon("experiment", "16", "#ffffff")} /demo/pipeline - Pipeline Completo</a>
                        <a href="/demo/embedding?text=PostgreSQL" class="api-btn embed-btn">{get_svg_icon("brain", "16", "#ffffff")} /demo/embedding - Crear Embeddings</a>
                        <a href="/demo/similarity?text1=pgvector&text2=vectorial" class="api-btn manual-btn">{get_svg_icon("ruler", "16", "#ffffff")} /demo/similarity - Calcular Similitud</a>
                        <a href="/demo/pipeline?query=bases+de+datos+vectoriales" class="api-btn demo-btn">{get_svg_icon("graduation", "16", "#ffffff")} /demo/pipeline - Demo Educativa Completa</a>
                        <a href="/manual/embed?q=ejemplo" class="api-btn embed-btn">{get_svg_icon("settings", "16", "#ffffff")} /manual/embed - Vectorización</a>
                        <a href="/manual/search?q=ejemplo" class="api-btn manual-btn">{get_svg_icon("search", "16", "#ffffff")} /manual/search - Búsqueda Manual</a>
                    </div>
                    
                    <div class="api-group">
                        <h4>{get_svg_icon("book", "20", "#3b82f6")} Documentación y Filtros</h4>
                        <a href="/filters/examples" class="api-btn filter-btn">{get_svg_icon("clipboard", "16", "#ffffff")} /filters/examples - Ejemplos</a>
                        <a href="/docs" class="api-btn docs-btn">{get_svg_icon("book", "16", "#ffffff")} /docs - Swagger/OpenAPI</a>
                        <a href="/?format=json" class="api-btn json-btn">{get_svg_icon("chart", "16", "#ffffff")} JSON Response</a>
                    </div>
                </div>
            </div>
            
            <div class="nav-section">
                <div class="nav-buttons">
                    <a href="/ai?q=evaluación+del+curso" class="nav-link primary">{get_svg_icon("graduation", "16")} Búsquedas Académicas</a>
                    <a href="/filters/examples" class="nav-link">{get_svg_icon("filter", "16")} Ejemplos de Filtros</a>
                    <a href="/demo/pipeline?query=bases+de+datos+vectoriales" class="nav-link">{get_svg_icon("book", "16")} Demo Pipeline RAG</a>
                </div>
            </div>
        </div>
        
        <script>
            function showLoading(message = 'Procesando consulta...') {{
                document.getElementById('loadingOverlay').classList.add('active');
                document.querySelector('.loading-text').textContent = message;
            }}
            
            function showSearchLoading() {{
                document.getElementById('searchLoading').classList.add('active');
                document.getElementById('searchButton').classList.add('btn-loading');
                document.getElementById('searchButton').disabled = true;
            }}
            
            function searchWithAI() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    showSearchLoading();
                    showLoading('Generando respuesta con IA...');
                    // Small delay to show animation before redirect
                    setTimeout(() => {{
                        window.location.href = `/ai?q=${{encodeURIComponent(query.trim())}}`;
                    }}, 300);
                }}
            }}
            
            function searchOnly() {{
                const query = document.getElementById('searchInput').value;
                if (query.trim()) {{
                    showSearchLoading();
                    showLoading('Buscando documentos...');
                    setTimeout(() => {{
                        window.location.href = `/ask?q=${{encodeURIComponent(query.trim())}}`;
                    }}, 300);
                }}
            }}
            
            // Add loading to feature cards
            function navigateWithLoading(url, message) {{
                showLoading(message);
                setTimeout(() => {{
                    window.location.href = url;
                }}, 300);
            }}
            
            // Allow Enter key to search
            document.getElementById('searchInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    searchWithAI();
                }}
            }});
            
            // Add loading to quick search buttons
            document.addEventListener('DOMContentLoaded', function() {{
                // Add loading to quick search buttons
                const quickButtons = document.querySelectorAll('.quick-btn');
                quickButtons.forEach(btn => {{
                    btn.addEventListener('click', function(e) {{
                        e.preventDefault();
                        showLoading('Ejecutando búsqueda rápida...');
                        setTimeout(() => {{
                            window.location.href = this.href;
                        }}, 300);
                    }});
                }});
                
                // Add loading to feature cards
                const featureCards = document.querySelectorAll('.action-card');
                featureCards.forEach(card => {{
                    const originalOnclick = card.getAttribute('onclick');
                    if (originalOnclick) {{
                        card.removeAttribute('onclick');
                        card.addEventListener('click', function() {{
                            if (originalOnclick.includes('/ask')) {{
                                showLoading('Preparando búsqueda...');
                            }} else if (originalOnclick.includes('/ai')) {{
                                showLoading('Iniciando IA...');
                            }} else if (originalOnclick.includes('/compare')) {{
                                showLoading('Comparando backends...');
                            }}
                            setTimeout(() => {{
                                eval(originalOnclick);
                            }}, 300);
                        }});
                    }}
                }});
                
                // Example text cycling
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

# ================================
# COMPREHENSIVE RAG PIPELINE DEMO ROUTES
# ================================


@app.get("/demo/pipeline")
def comprehensive_pipeline_demo(
    q: str = Query("¿Qué es pgvector?",
                   description="Consulta para demostrar pipeline"),
    format: str = Query("html", description="Formato: 'json' o 'html'")
):
    """Demo paso a paso completo del pipeline RAG"""
    try:
        from app.demo_pipeline import RAGPipelineDemo, create_demo_html

        demo = RAGPipelineDemo()

        # Execute all pipeline steps
        steps = []

        # Step 1: Parse text
        step1 = demo.step_1_parse_text()
        steps.append(step1)

        # Step 2: Clean text
        step2 = demo.step_2_clean_text(step1["output"]["sentences"])
        steps.append(step2)

        # Step 3: Create embeddings
        step3 = demo.step_3_create_embeddings(step2["output"]["chunks"])
        steps.append(step3)

        # Step 4: Process query
        step4 = demo.step_4_query_processing(q)
        steps.append(step4)

        # Step 5: Qdrant search
        step5 = demo.step_5_vector_search_qdrant(
            step4["output"]["full_embedding"])
        steps.append(step5)

        # Step 6: pgvector search
        step6 = demo.step_6_vector_search_pgvector(
            step4["output"]["full_embedding"])
        steps.append(step6)

        # Step 7: Math explanation
        if step3["output"]["full_embeddings"]:
            step7 = demo.step_7_similarity_math(
                step4["output"]["full_embedding"],
                step3["output"]["full_embeddings"]
            )
            steps.append(step7)

        # Step 8: Result ranking
        step8 = demo.step_8_result_ranking(step5["output"])
        steps.append(step8)

        if format == "json":
            from fastapi import Response
            import json
            return Response(
                content=json.dumps({
                    "query": q,
                    "pipeline_steps": steps,
                    "total_steps": len(steps),
                    "demo_type": "comprehensive_rag_pipeline"
                }),
                media_type="application/json"
            )

        # Create comprehensive HTML
        html = create_demo_html(steps, q)
        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Error in pipeline demo: {str(e)}")
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Demo error: {str(e)}")

        error_html = f"""
        <div style="color: #ff6b6b; padding: 20px; text-align: center;">
            <h2>❌ Error en Demo</h2>
            <p>Error: {str(e)}</p>
            <a href="/demo/pipeline?q=test" style="color: #4CAF50;">Reintentar con consulta de prueba</a>
        </div>
        """
        return HTMLResponse(content=error_html)


@app.get("/demo/test")
def test_demo():
    """Simple test endpoint to debug container issues"""
    return {"message": "test works", "status": "ok"}


@app.get("/demo/embedding")
def embedding_demo(
    text: str = Query("PostgreSQL es una base de datos",
                      description="Texto para convertir a embedding"),
    format: str = Query("html", description="Formato de respuesta")
):
    """Demo específico de creación de embeddings"""
    try:
        from app.demo_pipeline import RAGPipelineDemo

        demo = RAGPipelineDemo()

        # Create embedding for the text
        chunks = [text]
        embedding_result = demo.step_3_create_embeddings(chunks)

        if format == "json":
            from fastapi import Response
            import json
            return Response(
                content=json.dumps(embedding_result),
                media_type="application/json"
            )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Embedding Demo</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                    background: #0f0f0f; color: #e1e1e1; margin: 0; padding: 20px; line-height: 1.6;
                }}
                .container {{ max-width: 1000px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .embedding-viz {{ 
                    background: rgba(255,255,255,0.03); padding: 20px; border-radius: 10px; margin: 20px 0;
                }}
                .dimensions {{ 
                    display: grid; grid-template-columns: repeat(auto-fill, minmax(60px, 1fr)); gap: 5px; margin: 15px 0;
                }}
                .dim {{ 
                    background: #2196F3; color: white; padding: 5px; text-align: center; border-radius: 3px; font-size: 0.8em;
                }}
                .code {{ 
                    background: #1e1e1e; padding: 15px; border-radius: 8px; font-family: monospace; margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{get_svg_icon("brain", "32", "#2563eb")} Demo de Generación de Embeddings</h1>
                    <p><strong>Texto de entrada:</strong> "{text}"</p>
                </div>
                
                <div class="embedding-viz">
                    <h3>{get_svg_icon("chart", "24", "#2563eb")} Representación Vectorial</h3>
                    <p><strong>Dimensiones:</strong> {len(embedding_result['output']['sample_embedding'])} mostradas (de {len(embedding_result['output']['full_embeddings'][0])} totales)</p>
                    
                    <div class="dimensions">
        """

        # Show first 50 dimensions as visual representation
        for i, val in enumerate(embedding_result['output']['sample_embedding'][:50]):
            html += f'<div class="dim" title="Dim {i}: {val:.4f}">{val:.3f}</div>'

        html += f"""
                    </div>
                    
                    <p style="color: #999;">Cada número representa una dimensión de significado en el espacio vectorial de 768 dimensiones.</p>
                </div>
                
                <div class="code">
                    <h3>💻 Código Utilizado</h3>
                    <pre>{embedding_result['code_example']}</pre>
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="/demo/pipeline?q={text}" style="color: #4CAF50; text-decoration: none; padding: 10px 20px; border: 1px solid #4CAF50; border-radius: 5px;">
                        {get_svg_icon("experiment", "16", "#4CAF50")} Ver Demo Completo del Pipeline
                    </a>
                    <a href="/" style="color: #4CAF50; text-decoration: none; padding: 10px 20px; margin-left: 10px;">
                        {get_svg_icon("home", "16", "#4CAF50")} Volver al Inicio
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Error in embedding demo: {str(e)}")
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Embedding demo error: {str(e)}")
        return HTMLResponse(content=f"<div style='color: red; text-align: center; padding: 20px;'>Error: {str(e)}</div>")


@app.get("/demo/similarity")
def similarity_demo(
    text1: str = Query("PostgreSQL con pgvector", description="Primer texto"),
    text2: str = Query("Base de datos vectorial", description="Segundo texto"),
    format: str = Query("html", description="Formato de respuesta")
):
    """Demo de cálculo de similitud entre textos"""
    try:
        from app.demo_pipeline import RAGPipelineDemo
        import numpy as np

        demo = RAGPipelineDemo()

        # Create embeddings for both texts
        embedding1_result = demo.step_3_create_embeddings([text1])
        embedding2_result = demo.step_3_create_embeddings([text2])

        emb1 = np.array(embedding1_result['output']['full_embeddings'][0])
        emb2 = np.array(embedding2_result['output']['full_embeddings'][0])

        # Calculate similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        cosine_similarity = dot_product / (norm1 * norm2)
        cosine_distance = 1 - cosine_similarity

        result = {
            "text1": text1,
            "text2": text2,
            "calculations": {
                "dot_product": float(dot_product),
                "norm1": float(norm1),
                "norm2": float(norm2),
                "cosine_similarity": float(cosine_similarity),
                "cosine_distance": float(cosine_distance),
                "similarity_percentage": float(cosine_similarity * 100)
            }
        }

        if format == "json":
            from fastapi import Response
            import json
            return Response(
                content=json.dumps(result),
                media_type="application/json"
            )

        similarity_color = "#4CAF50" if cosine_similarity > 0.7 else "#FF9800" if cosine_similarity > 0.3 else "#F44336"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Similarity Demo</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                    background: #0f0f0f; color: #e1e1e1; margin: 0; padding: 20px; line-height: 1.6;
                }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .text-comparison {{ 
                    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;
                }}
                .text-box {{ 
                    background: rgba(255,255,255,0.03); padding: 20px; border-radius: 10px; 
                }}
                .similarity-result {{ 
                    background: {similarity_color}22; border: 2px solid {similarity_color}; 
                    padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;
                }}
                .calculation {{ 
                    background: rgba(255,255,255,0.02); padding: 15px; border-radius: 8px; margin: 10px 0; font-family: monospace;
                }}
                .progress-bar {{ 
                    background: #333; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0;
                }}
                .progress-fill {{ 
                    background: {similarity_color}; height: 100%; width: {cosine_similarity * 100:.1f}%; 
                    transition: width 0.5s ease;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 style="text-align: center;">{get_svg_icon("ruler", "32", "#2563eb")} Calculadora de Similitud de Textos</h1>
                
                <div class="text-comparison">
                    <div class="text-box">
                        <h3>{get_svg_icon("edit", "20", "#2563eb")} Texto 1</h3>
                        <p>"{text1}"</p>
                    </div>
                    <div class="text-box">
                        <h3>{get_svg_icon("edit", "20", "#2563eb")} Texto 2</h3>
                        <p>"{text2}"</p>
                    </div>
                </div>
                
                <div class="similarity-result">
                    <h2 style="color: {similarity_color};">Similitud: {cosine_similarity:.3f}</h2>
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <p>{cosine_similarity * 100:.1f}% similar</p>
                </div>
                
                <h3>🔢 Cálculos Matemáticos</h3>
                <div class="calculation">
                    <strong>Producto Punto:</strong> {dot_product:.6f}
                </div>
                <div class="calculation">
                    <strong>Normas Vectoriales:</strong> ||texto1|| = {norm1:.6f}, ||texto2|| = {norm2:.6f}
                </div>
                <div class="calculation">
                    <strong>Similitud de Coseno:</strong> {dot_product:.6f} / ({norm1:.6f} × {norm2:.6f}) = {cosine_similarity:.6f}
                </div>
                <div class="calculation">
                    <strong>Distancia de Coseno:</strong> 1 - {cosine_similarity:.6f} = {cosine_distance:.6f}
                </div>
                
                <div style="text-align: center; margin-top: 30px;">
                    <a href="/demo/pipeline?q=comparación de similitud" style="color: #4CAF50; text-decoration: none; padding: 10px 20px; border: 1px solid #4CAF50; border-radius: 5px;">
                        {get_svg_icon("experiment", "16", "#4CAF50")} Demo Completo del Pipeline
                    </a>
                    <a href="/" style="color: #4CAF50; text-decoration: none; padding: 10px 20px; margin-left: 10px;">
                        {get_svg_icon("home", "16", "#4CAF50")} Volver al Inicio
                    </a>
                </div>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Error in similarity demo: {str(e)}")
        if format == "json":
            raise HTTPException(
                status_code=500, detail=f"Similarity demo error: {str(e)}")
        return HTMLResponse(content=f"<div style='color: red; text-align: center; padding: 20px;'>Error: {str(e)}</div>")
