"""
Response templates for RAG Demo
Contains all HTML response templates with dynamic variables
"""
import json


def enhanced_ai_response_html(data: dict, query: str) -> str:
    """Enhanced HTML for AI responses with complete UX and functionality"""

    # Extract key information
    ai_response = data.get("ai_response", "")
    sources = data.get("sources", [])
    total_results = data.get("total_results", 0)
    backend = data.get("backend", "").upper()
    model = data.get("model", "")

    # Build sources section with page references
    sources_html = ""
    if sources:
        sources_html = "<div class='sources-section'><h3>📚 Fuentes:</h3>"
        for i, source in enumerate(sources, 1):
            similarity = source.get("similarity", "0.000")
            preview = source.get("preview", "")
            reference = source.get("reference", "")

            sources_html += f"""
            <div class="source-item">
                <div class="source-header">
                    <span class="source-number">{i}.</span>
                    <span class="source-doc">{reference}</span>
                    <span class="source-similarity">Similitud: {similarity}</span>
                </div>
                <div class="source-preview">{preview}</div>
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
                border-radius: 6px;
                background: #4CAF50;
                color: white;
                cursor: pointer;
                font-weight: 500;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: all 0.2s;
            }}
            .search-btn:hover, .nav-btn:hover {{ background: #45a049; }}
            .controls-row {{
                display: flex;
                gap: 20px;
                margin-bottom: 15px;
                flex-wrap: wrap;
            }}
            .control-group {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .control-group label {{
                font-weight: 500;
                min-width: 50px;
            }}
            .select-control {{
                padding: 8px 12px;
                border: none;
                border-radius: 6px;
                background: rgba(255,255,255,0.1);
                color: white;
                cursor: pointer;
            }}
            .title {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 24px;
                font-weight: 600;
                margin: 15px 0 5px 0;
            }}
            .subtitle {{
                opacity: 0.9;
                font-size: 14px;
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
            .model-info {{
                font-size: 14px;
                color: #9ca3af;
                margin-top: 10px;
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
                display: inline-flex;
                align-items: center;
                gap: 8px;
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
                    <button class="search-btn" onclick="newSearch()"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg> Buscar</button>
                    <button class="nav-btn" onclick="window.location.href='/ask?q=bases+de+datos+vectoriales'"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg> Solo /ask</button>
                    <button class="nav-btn" onclick="window.location.href='/'"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9,22 9,12 15,12 15,22"></polyline></svg> Menu Principal</button>
                </div>
                <div class="controls-row">
                    <div class="control-group">
                        <label for="backendSelect">Motor:</label>
                        <select id="backendSelect" class="select-control">
                            <option value="qdrant" {'selected' if backend == 'QDRANT' else ''}>Qdrant</option>
                            <option value="pgvector" {'selected' if backend == 'PGVECTOR' else ''}>PostgreSQL</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="modelSelect">Modelo:</label>
                        <select id="modelSelect" class="select-control">
                            <option value="phi3:mini" {'selected' if model == 'phi3:mini' else ''}>Phi3 Mini</option>
                            <option value="llama3.1:8b" {'selected' if model == 'llama3.1:8b' else ''}>Llama 3.1 8B</option>
                            <option value="gemma2:2b" {'selected' if model == 'gemma2:2b' else ''}>Gemma2 2B</option>
                            <option value="qwen2.5:3b" {'selected' if model == 'qwen2.5:3b' else ''}>Qwen2.5 3B</option>
                        </select>
                    </div>
                </div>
                <div class="title"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><rect x="3" y="11" width="18" height="10" rx="2"></rect><circle cx="12" cy="5" r="2"></circle><path d="m12 7v4"></path><line x1="8" y1="16" x2="8" y2="16"></line><line x1="16" y1="16" x2="16" y2="16"></line></svg> Respuesta AI</div>
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
                    <button class="toggle-btn" onclick="toggleDetails()"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="m12 1v6m0 6v6"></path><path d="m20.49 9l-1.73 1-1.73-1 1.73-1z"></path><path d="m20.49 15l-1.73 1-1.73-1 1.73-1z"></path><path d="m3.51 9l1.73 1 1.73-1-1.73-1z"></path><path d="m3.51 15l1.73 1 1.73-1-1.73-1z"></path></svg> Ver detalles técnicos JSON</button>
                    <div class="json-details" id="jsonDetails">
                        <div class="json-content" id="jsonContent">{json_str}</div>
                    </div>
                </div>
                
                <div class="navigation">
                    <div class="nav-group">
                        <a href="/" class="nav-link primary"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9,22 9,12 15,12 15,22"></polyline></svg> Inicio</a>
                        <a href="/ask?q=evaluación+del+curso" class="nav-link"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect><path d="m16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path></svg> Búsquedas Rápidas</a>
                    </div>
                    <div class="nav-group">
                        <a href="/docs" class="nav-link"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg> API Docs</a>
                        <a href="/compare?q={query}" class="nav-link"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><path d="m9 12 2 2 4-4"></path><path d="M21 12c-1 0-3-1-3-3s2-3 3-3 3 1 3 3-2 3-3 3"></path><path d="M3 12c1 0 3-1 3-3s-2-3-3-3-3 1-3 3 2 3 3 3"></path><path d="m3 12h6m6 0h6"></path></svg> Comparar Motores</a>
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
                    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="m12 1v6m0 6v6"/><path d="m20.49 9l-1.73 1-1.73-1 1.73-1z"/><path d="m20.49 15l-1.73 1-1.73-1 1.73-1z"/><path d="m3.51 9l1.73 1 1.73-1-1.73-1z"/><path d="m3.51 15l1.73 1 1.73-1-1.73-1z"/></svg> Ver detalles técnicos JSON`;
                }} else {{
                    details.classList.add('show');
                    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="m12 1v6m0 6v6"/><path d="m20.49 9l-1.73 1-1.73-1 1.73-1z"/><path d="m20.49 15l-1.73 1-1.73-1 1.73-1z"/><path d="m3.51 9l1.73 1 1.73-1-1.73-1z"/><path d="m3.51 15l1.73 1 1.73-1-1.73-1z"/></svg> Ocultar detalles JSON`;
                    
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
                html = html.replace(/: ([0-9]+\\.?[0-9]*)/g, ': <span class="number">$1</span>');
                
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
    """Enhanced HTML for general responses with better styling"""

    # Extract common data
    query = data.get("query", "")
    results = data.get("results", [])
    total_results = data.get("total_results", 0)
    backend = data.get("backend", "")
    search_time = data.get("search_time_ms", 0)

    # Build results section
    results_html = ""
    if results:
        for i, result in enumerate(results, 1):
            doc = result.get("document", "").replace(
                "data/raw/", "").replace(".pdf", "")
            similarity = result.get("similarity", "0.000")
            content = result.get("content", "")
            page = result.get("page", "")
            chapter = result.get("chapter", "")

            # Build reference
            reference = f"{doc}"
            if page:
                reference += f" - Página {page}"
            if chapter:
                reference += f" ({chapter})"

            results_html += f"""
            <div class="result-item">
                <div class="result-header">
                    <span class="result-number">{i}</span>
                    <span class="result-doc">{reference}</span>
                    <span class="result-similarity">Similitud: {similarity}</span>
                </div>
                <div class="result-content">{content}</div>
            </div>
            """

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
                background: linear-gradient(135deg, {theme_color} 0%, #1e3a5f 100%);
                padding: 20px 30px;
                color: white;
            }}
            .search-bar {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
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
                border-radius: 6px;
                background: #4CAF50;
                color: white;
                cursor: pointer;
                font-weight: 500;
                text-decoration: none;
                display: inline-block;
                transition: all 0.2s;
            }}
            .search-btn:hover, .nav-btn:hover {{ background: #45a049; }}
            .metadata {{
                background: #2a2a2a;
                margin: 20px 30px;
                padding: 15px 20px;
                border-radius: 8px;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }}
            .metadata-item {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .metadata-label {{
                color: #888;
                font-size: 13px;
            }}
            .metadata-value {{
                color: #fff;
                font-weight: 500;
            }}
            .results-section {{
                background: #252525;
                margin: 20px 30px;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid {theme_color};
            }}
            .result-item {{
                background: #2f2f2f;
                margin: 15px 0;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #3a3a3a;
            }}
            .result-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }}
            .result-number {{
                background: {theme_color};
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }}
            .result-doc {{
                flex: 1;
                font-weight: 500;
                color: #61dafb;
            }}
            .result-similarity {{
                background: #4CAF50;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 11px;
            }}
            .result-content {{
                color: #b3b3b3;
                font-size: 14px;
                line-height: 1.4;
                margin-left: 20px;
            }}
            .json-section {{
                background: #1f1f1f;
                margin: 20px 30px;
                border-radius: 8px;
                overflow: hidden;
            }}
            .json-header {{
                background: #333;
                padding: 15px 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .json-content {{
                padding: 20px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 13px;
                line-height: 1.4;
                white-space: pre-wrap;
                display: none;
                background: #1a1a1a;
            }}
            .json-content.show {{ display: block; }}
            h1 {{ margin: 0; font-size: 28px; }}
            h3 {{ margin: 10px 0; color: {theme_color}; }}
            .nav-buttons {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }}
            @media (max-width: 768px) {{
                body {{ padding: 10px; }}
                .container {{ margin: 0; }}
                .header {{ padding: 15px 20px; }}
                .nav-buttons {{ flex-direction: column; }}
                .metadata {{ flex-direction: column; gap: 10px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{title}</h1>
                <div class="search-bar">
                    <input type="text" class="search-input" placeholder="Nueva consulta..." value="{query}">
                    <button class="search-btn" onclick="newSearch()">🔍 Buscar</button>
                </div>
                <div class="nav-buttons">
                    <a href="/" class="nav-btn">🏠 Inicio</a>
                    <a href="/ask?q={query}" class="nav-btn">📖 Búsqueda</a>
                    <a href="/ai?q={query}" class="nav-btn">🤖 IA</a>
                    <a href="/docs" class="nav-btn">📚 API Docs</a>
                </div>
            </div>

            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Backend:</span>
                    <span class="metadata-value">{backend}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Tiempo:</span>
                    <span class="metadata-value">{search_time}ms</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Resultados:</span>
                    <span class="metadata-value">{total_results}</span>
                </div>
            </div>

            <div class="results-section">
                <h3>📋 Resultados</h3>
                {results_html}
            </div>

            <div class="json-section">
                <div class="json-header" onclick="toggleJson()">
                    <span>📄 Ver datos completos (JSON)</span>
                    <span id="toggle-icon">▼</span>
                </div>
                <div id="json-content" class="json-content">{json_str}</div>
            </div>
        </div>

        <script>
            function toggleJson() {{
                const content = document.getElementById('json-content');
                const icon = document.getElementById('toggle-icon');
                if (content.classList.contains('show')) {{
                    content.classList.remove('show');
                    icon.textContent = '▼';
                }} else {{
                    content.classList.add('show');
                    icon.textContent = '▲';
                }}
            }}

            function newSearch() {{
                const query = document.querySelector('.search-input').value;
                if (query.trim()) {{
                    window.location.href = `/ask?q=${{encodeURIComponent(query)}}`;
                }}
            }}

            document.querySelector('.search-input').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    newSearch();
                }}
            }});
        </script>
    </body>
    </html>
    """


def enhanced_search_response_html(data: dict, query: str) -> str:
    """Enhanced HTML for search responses"""
    # Add query to data if not present
    if 'query' not in data:
        data['query'] = query
    return enhanced_general_response_html(data, "🔍 Resultados de Búsqueda", "#059669")


def pretty_json_html(data: dict, title: str = "API Response") -> str:
    """Pretty JSON HTML template"""
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
                font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                background: #1a1a1a; 
                color: #e1e1e1; 
                margin: 0;
                padding: 20px;
                line-height: 1.4;
            }}
            .container {{ 
                max-width: 1000px; 
                margin: 0 auto; 
                background: #2a2a2a; 
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            }}
            .header {{
                color: #4CAF50;
                font-size: 24px;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #4CAF50;
            }}
            .json-content {{
                background: #1f1f1f;
                padding: 20px;
                border-radius: 6px;
                overflow-x: auto;
                white-space: pre-wrap;
                font-size: 14px;
                border: 1px solid #3a3a3a;
            }}
            .nav-link {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                transition: background 0.2s;
            }}
            .nav-link:hover {{
                background: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">{title}</div>
            <div class="json-content">{json_str}</div>
            <a href="/" class="nav-link">← Volver al inicio</a>
        </div>
    </body>
    </html>
    """


def format_color(hex_color: str) -> str:
    """Ensure color format is valid"""
    if not hex_color or not hex_color.startswith('#'):
        return "#2563eb"
    return hex_color


def adjust_color(color: str) -> str:
    """Adjust color brightness for better contrast"""
    try:
        # Remove # and convert to RGB
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

        # Darken the color slightly
        r = max(0, r - 30)
        g = max(0, g - 30)
        b = max(0, b - 30)

        return f"#{r:02x}{g:02x}{b:02x}"
    except (ValueError, IndexError):
        return "#1e3a5f"
