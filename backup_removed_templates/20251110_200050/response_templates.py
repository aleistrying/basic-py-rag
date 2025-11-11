"""
Response templates for RAG Demo
Contains all HTML response templates with dynamic variables
"""
import json
import re


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
            .ai-response {{
                background: #2a2a2a;
                margin: 20px 30px;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
            }}
            .sources-section {{
                background: #252525;
                margin: 20px 30px;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #2196F3;
            }}
            .source-item {{
                background: #2f2f2f;
                margin: 10px 0;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #3a3a3a;
            }}
            .source-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 8px;
            }}
            .source-number {{
                background: #2196F3;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }}
            .source-doc {{
                flex: 1;
                font-weight: 500;
                color: #61dafb;
            }}
            .source-similarity {{
                background: #4CAF50;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 11px;
            }}
            .source-preview {{
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
            h1 {{ margin: 0; font-size: 28px; }}
            h3 {{ margin: 10px 0; color: #4CAF50; }}
            .nav-buttons {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }}
            @media (max-width: 768px) {{
                body {{ padding: 10px; }}
                .container {{ margin: 0; }}
                .header {{ padding: 15px 20px; }}
                .search-bar {{ flex-direction: column; }}
                .nav-buttons {{ flex-direction: column; }}
                .metadata {{ flex-direction: column; gap: 10px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Respuesta con IA</h1>
                <div class="search-bar">
                    <input type="text" class="search-input" placeholder="Nueva consulta..." value="{query}">
                    <button class="search-btn" onclick="newSearch()">🔍 Buscar</button>
                </div>
                <div class="nav-buttons">
                    <a href="/" class="nav-btn">🏠 Inicio</a>
                    <a href="/ask?q={query}" class="nav-btn">📖 Solo Búsqueda</a>
                    <a href="/compare?q={query}" class="nav-btn">⚖️ Comparar</a>
                    <a href="/docs" class="nav-btn">📚 API Docs</a>
                </div>
            </div>

            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Backend:</span>
                    <span class="metadata-value">{backend}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Modelo:</span>
                    <span class="metadata-value">{model}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Fuentes:</span>
                    <span class="metadata-value">{len(sources)}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Total resultados:</span>
                    <span class="metadata-value">{total_results}</span>
                </div>
            </div>

            <div class="ai-response">
                <h3>🤖 Respuesta de IA</h3>
                <p>{ai_response}</p>
            </div>

            {sources_html}

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
                    window.location.href = `/ai?q=${{encodeURIComponent(query)}}`;
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


def enhanced_general_response_html(data: dict, title: str, theme_color: str = "#2563eb") -> str:
    """Enhanced HTML template for AI responses with better styling and interactivity"""

    # Extract information
    qdrant_results = data.get('results', {}).get('qdrant', {})
    postgres_results = data.get('results', {}).get('postgres', {})
    ai_response = data.get('ai_response', 'No response available')

    # Get search results for both databases
    qdrant_docs = qdrant_results.get('documents', [])
    postgres_docs = postgres_results.get('documents', [])

    qdrant_time = qdrant_results.get('search_time_ms', 0)
    postgres_time = postgres_results.get('search_time_ms', 0)

    # Create documents HTML for Qdrant
    qdrant_docs_html = ""
    for i, doc in enumerate(qdrant_docs[:3]):  # Show top 3
        score = doc.get('score', 0)
        content = doc.get('content', '')[
            :200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', '')
        metadata = doc.get('metadata', {})

        qdrant_docs_html += f"""
        <div class="result-item">
            <div class="result-header">
                <span class="result-rank">#{i+1}</span>
                <span class="result-score">Score: {score:.4f}</span>
            </div>
            <div class="result-content">{content}</div>
            <div class="result-metadata">
                <strong>File:</strong> {metadata.get('source_file', 'N/A')}<br>
                <strong>Page:</strong> {metadata.get('page_number', 'N/A')}
            </div>
        </div>
        """

    # Create documents HTML for PostgreSQL
    postgres_docs_html = ""
    for i, doc in enumerate(postgres_docs[:3]):  # Show top 3
        score = doc.get('score', 0)
        content = doc.get('content', '')[
            :200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', '')
        metadata = doc.get('metadata', {})

        postgres_docs_html += f"""
        <div class="result-item">
            <div class="result-header">
                <span class="result-rank">#{i+1}</span>
                <span class="result-score">Distance: {score:.4f}</span>
            </div>
            <div class="result-content">{content}</div>
            <div class="result-metadata">
                <strong>File:</strong> {metadata.get('source_file', 'N/A')}<br>
                <strong>Page:</strong> {metadata.get('page_number', 'N/A')}
            </div>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG AI Response - {query}</title>
        <style>
            :root {{
                --primary-color: #2563eb;
                --secondary-color: #1e40af;
                --success-color: #059669;
                --warning-color: #d97706;
                --danger-color: #dc2626;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
                --gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            
            * {{
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background: var(--light-bg);
                color: var(--text-color);
            }}
            
            .header {{
                background: var(--gradient);
                color: white;
                padding: 2rem 0;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
            }}
            
            .query-display {{
                background: rgba(255, 255, 255, 0.2);
                padding: 1rem;
                border-radius: 0.5rem;
                margin-top: 1rem;
                font-style: italic;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 1rem;
            }}
            
            .ai-response {{
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                border-left: 4px solid var(--primary-color);
            }}
            
            .ai-response h2 {{
                color: var(--primary-color);
                margin-top: 0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .comparison-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                margin-bottom: 2rem;
            }}
            
            @media (max-width: 768px) {{
                .comparison-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .db-section {{
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            
            .db-section.qdrant {{
                border-left: 4px solid var(--success-color);
            }}
            
            .db-section.postgres {{
                border-left: 4px solid var(--warning-color);
            }}
            
            .db-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-color);
            }}
            
            .db-title {{
                font-size: 1.25rem;
                font-weight: 600;
                margin: 0;
            }}
            
            .db-title.qdrant {{ color: var(--success-color); }}
            .db-title.postgres {{ color: var(--warning-color); }}
            
            .search-time {{
                background: var(--light-bg);
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.875rem;
                color: var(--text-muted);
            }}
            
            .result-item {{
                background: var(--light-bg);
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid var(--border-color);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            
            .result-item:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            
            .result-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }}
            
            .result-rank {{
                background: var(--primary-color);
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.875rem;
                font-weight: 600;
            }}
            
            .result-score {{
                background: var(--light-bg);
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.875rem;
                font-family: monospace;
            }}
            
            .result-content {{
                margin: 0.5rem 0;
                line-height: 1.6;
            }}
            
            .result-metadata {{
                font-size: 0.875rem;
                color: var(--text-muted);
                margin-top: 0.5rem;
                padding-top: 0.5rem;
                border-top: 1px solid var(--border-color);
            }}
            
            .navigation {{
                text-align: center;
                margin: 2rem 0;
            }}
            
            .nav-button {{
                background: var(--primary-color);
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 0.5rem;
                text-decoration: none;
                display: inline-block;
                margin: 0 0.5rem;
                transition: background 0.2s, transform 0.2s;
                font-weight: 500;
            }}
            
            .nav-button:hover {{
                background: var(--secondary-color);
                transform: translateY(-1px);
            }}
            
            .footer {{
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                border-top: 1px solid var(--border-color);
                margin-top: 2rem;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>🤖 RAG AI Response</h1>
                <div class="query-display">
                    <strong>Query:</strong> "{query}"
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="ai-response">
                <h2>🧠 AI Generated Response</h2>
                <div style="white-space: pre-wrap; line-height: 1.8;">{ai_response}</div>
            </div>
            
            <div class="comparison-grid">
                <div class="db-section qdrant">
                    <div class="db-header">
                        <h3 class="db-title qdrant">📊 Qdrant Vector DB</h3>
                        <span class="search-time">{qdrant_time:.2f}ms</span>
                    </div>
                    <div class="search-results">
                        {qdrant_docs_html if qdrant_docs_html else '<p>No results found</p>'}
                    </div>
                </div>
                
                <div class="db-section postgres">
                    <div class="db-header">
                        <h3 class="db-title postgres">🐘 PostgreSQL + pgvector</h3>
                        <span class="search-time">{postgres_time:.2f}ms</span>
                    </div>
                    <div class="search-results">
                        {postgres_docs_html if postgres_docs_html else '<p>No results found</p>'}
                    </div>
                </div>
            </div>
            
            <div class="navigation">
                <a href="/" class="nav-button">🏠 Home</a>
                <a href="/ask?query={query}" class="nav-button">🔍 Search Only</a>
                <a href="/compare?query={query}" class="nav-button">⚖️ Compare DBs</a>
                <a href="/demo/pipeline" class="nav-button">📚 Demo Pipeline</a>
            </div>
        </div>
        
        <div class="footer">
            <p>RAG System with Qdrant and PostgreSQL+pgvector</p>
        </div>
    </body>
    </html>
    """

    return html


def enhanced_search_response_html(data: dict, query: str) -> str:
    """Enhanced HTML template for search-only responses"""

    # Extract information
    qdrant_results = data.get('qdrant', {})
    postgres_results = data.get('postgres', {})

    # Get search results for both databases
    qdrant_docs = qdrant_results.get('documents', [])
    postgres_docs = postgres_results.get('documents', [])

    qdrant_time = qdrant_results.get('search_time_ms', 0)
    postgres_time = postgres_results.get('search_time_ms', 0)

    # Create documents HTML for Qdrant
    qdrant_docs_html = ""
    for i, doc in enumerate(qdrant_docs):
        score = doc.get('score', 0)
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})

        # Truncate content for display
        display_content = content[:300] + \
            "..." if len(content) > 300 else content

        qdrant_docs_html += f"""
        <div class="result-item">
            <div class="result-header">
                <span class="result-rank">#{i+1}</span>
                <span class="result-score">Score: {score:.4f}</span>
            </div>
            <div class="result-content">{display_content}</div>
            <div class="result-metadata">
                <strong>Source:</strong> {metadata.get('source_file', 'N/A')}<br>
                <strong>Page:</strong> {metadata.get('page_number', 'N/A')}<br>
                <strong>Chunk:</strong> {metadata.get('chunk_index', 'N/A')}
            </div>
        </div>
        """

    # Create documents HTML for PostgreSQL
    postgres_docs_html = ""
    for i, doc in enumerate(postgres_docs):
        score = doc.get('score', 0)
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})

        # Truncate content for display
        display_content = content[:300] + \
            "..." if len(content) > 300 else content

        postgres_docs_html += f"""
        <div class="result-item">
            <div class="result-header">
                <span class="result-rank">#{i+1}</span>
                <span class="result-score">Distance: {score:.4f}</span>
            </div>
            <div class="result-content">{display_content}</div>
            <div class="result-metadata">
                <strong>Source:</strong> {metadata.get('source_file', 'N/A')}<br>
                <strong>Page:</strong> {metadata.get('page_number', 'N/A')}<br>
                <strong>Chunk:</strong> {metadata.get('chunk_index', 'N/A')}
            </div>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vector Search Results - {query}</title>
        <style>
            :root {{
                --primary-color: #059669;
                --secondary-color: #047857;
                --qdrant-color: #6366f1;
                --postgres-color: #f59e0b;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
                --gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }}
            
            * {{
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background: var(--light-bg);
                color: var(--text-color);
            }}
            
            .header {{
                background: var(--gradient);
                color: white;
                padding: 2rem 0;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
            }}
            
            .query-display {{
                background: rgba(255, 255, 255, 0.2);
                padding: 1rem;
                border-radius: 0.5rem;
                margin-top: 1rem;
                font-style: italic;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 1rem;
            }}
            
            .comparison-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                margin-bottom: 2rem;
            }}
            
            @media (max-width: 768px) {{
                .comparison-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
            
            .db-section {{
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            
            .db-section.qdrant {{
                border-left: 4px solid var(--qdrant-color);
            }}
            
            .db-section.postgres {{
                border-left: 4px solid var(--postgres-color);
            }}
            
            .db-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid var(--border-color);
            }}
            
            .db-title {{
                font-size: 1.25rem;
                font-weight: 600;
                margin: 0;
            }}
            
            .db-title.qdrant {{ color: var(--qdrant-color); }}
            .db-title.postgres {{ color: var(--postgres-color); }}
            
            .search-time {{
                background: var(--light-bg);
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.875rem;
                color: var(--text-muted);
            }}
            
            .result-item {{
                background: var(--light-bg);
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid var(--border-color);
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            
            .result-item:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            
            .result-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }}
            
            .result-rank {{
                background: var(--primary-color);
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.875rem;
                font-weight: 600;
            }}
            
            .result-score {{
                background: var(--light-bg);
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.875rem;
                font-family: monospace;
            }}
            
            .result-content {{
                margin: 0.5rem 0;
                line-height: 1.6;
            }}
            
            .result-metadata {{
                font-size: 0.875rem;
                color: var(--text-muted);
                margin-top: 0.5rem;
                padding-top: 0.5rem;
                border-top: 1px solid var(--border-color);
            }}
            
            .navigation {{
                text-align: center;
                margin: 2rem 0;
            }}
            
            .nav-button {{
                background: var(--primary-color);
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 0.5rem;
                text-decoration: none;
                display: inline-block;
                margin: 0 0.5rem;
                transition: background 0.2s, transform 0.2s;
                font-weight: 500;
            }}
            
            .nav-button:hover {{
                background: var(--secondary-color);
                transform: translateY(-1px);
            }}
            
            .footer {{
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                border-top: 1px solid var(--border-color);
                margin-top: 2rem;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>🔍 Vector Search Results</h1>
                <div class="query-display">
                    <strong>Query:</strong> "{query}"
                </div>
            </div>
        </div>
        
        <div class="container">
            <div class="comparison-grid">
                <div class="db-section qdrant">
                    <div class="db-header">
                        <h3 class="db-title qdrant">📊 Qdrant Results</h3>
                        <span class="search-time">{qdrant_time:.2f}ms</span>
                    </div>
                    <div class="search-results">
                        {qdrant_docs_html if qdrant_docs_html else '<p>No results found</p>'}
                    </div>
                </div>
                
                <div class="db-section postgres">
                    <div class="db-header">
                        <h3 class="db-title postgres">🐘 PostgreSQL Results</h3>
                        <span class="search-time">{postgres_time:.2f}ms</span>
                    </div>
                    <div class="search-results">
                        {postgres_docs_html if postgres_docs_html else '<p>No results found</p>'}
                    </div>
                </div>
            </div>
            
            <div class="navigation">
                <a href="/" class="nav-button">🏠 Home</a>
                <a href="/ai?query={query}" class="nav-button">🤖 AI Response</a>
                <a href="/compare?query={query}" class="nav-button">⚖️ Compare DBs</a>
                <a href="/demo/pipeline" class="nav-button">📚 Demo Pipeline</a>
            </div>
        </div>
        
        <div class="footer">
            <p>Vector Search powered by Qdrant and PostgreSQL+pgvector</p>
        </div>
    </body>
    </html>
    """

    return html


def enhanced_general_response_html(data: dict, title: str, theme_color: str = "#2563eb") -> str:
    """General purpose enhanced HTML template"""

    import json

    # Format the data nicely
    formatted_data = json.dumps(data, ensure_ascii=False, indent=2)

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            :root {{
                --primary-color: {theme_color};
                --secondary-color: #1e40af;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
            }}
            
            * {{
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background: var(--light-bg);
                color: var(--text-color);
            }}
            
            .header {{
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
                color: white;
                padding: 2rem 0;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
            }}
            
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                padding: 0 1rem;
            }}
            
            .content-card {{
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border-left: 4px solid var(--primary-color);
            }}
            
            .json-container {{
                background: #1e293b;
                color: #e2e8f0;
                padding: 1.5rem;
                border-radius: 0.5rem;
                overflow-x: auto;
                font-family: 'Fira Code', 'Courier New', monospace;
                font-size: 0.875rem;
                line-height: 1.5;
            }}
            
            .navigation {{
                text-align: center;
                margin: 2rem 0;
            }}
            
            .nav-button {{
                background: var(--primary-color);
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 0.5rem;
                text-decoration: none;
                display: inline-block;
                margin: 0 0.5rem;
                transition: background 0.2s, transform 0.2s;
                font-weight: 500;
            }}
            
            .nav-button:hover {{
                background: var(--secondary-color);
                transform: translateY(-1px);
            }}
            
            .footer {{
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                border-top: 1px solid var(--border-color);
                margin-top: 2rem;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>{title}</h1>
            </div>
        </div>
        
        <div class="container">
            <div class="content-card">
                <div class="json-container">
                    <pre>{formatted_data}</pre>
                </div>
            </div>
            
            <div class="navigation">
                <a href="/" class="nav-button">🏠 Home</a>
                <a href="/docs" class="nav-button">📚 API Docs</a>
            </div>
        </div>
        
        <div class="footer">
            <p>RAG System API Response</p>
        </div>
    </body>
    </html>
    """

    return html


def pretty_json_html(data: dict, title: str = "API Response") -> str:
    """Pretty JSON HTML template with syntax highlighting"""

    import json

    # Format the JSON nicely
    formatted_json = json.dumps(data, ensure_ascii=False, indent=2)

    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 2rem;
                background: #f8fafc;
                color: #334155;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 2rem;
                padding: 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 1rem;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }}
            
            .header h1 {{
                margin: 0;
                font-size: 2rem;
                font-weight: 300;
            }}
            
            .json-container {{
                background: #1e293b;
                color: #e2e8f0;
                padding: 2rem;
                border-radius: 1rem;
                overflow-x: auto;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                font-family: 'Fira Code', 'Courier New', monospace;
                font-size: 0.875rem;
                line-height: 1.5;
            }}
            
            .json-container pre {{
                margin: 0;
                white-space: pre-wrap;
            }}
            
            .navigation {{
                text-align: center;
                margin: 2rem 0;
            }}
            
            .nav-button {{
                background: #667eea;
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 0.5rem;
                text-decoration: none;
                display: inline-block;
                margin: 0 0.5rem;
                transition: background 0.2s, transform 0.2s;
                font-weight: 500;
            }}
            
            .nav-button:hover {{
                background: #5a67d8;
                transform: translateY(-1px);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
        </div>
        
        <div class="json-container">
            <pre>{formatted_json}</pre>
        </div>
        
        <div class="navigation">
            <a href="/" class="nav-button">🏠 Home</a>
            <a href="/docs" class="nav-button">📚 API Docs</a>
        </div>
    </body>
    </html>
    """

    return html
