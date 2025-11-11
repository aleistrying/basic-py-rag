"""
HTML templates for utility and info routes
"""


def render_filter_examples_html() -> str:
    """HTML template for filter examples"""

    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Filter Examples</title>
        <style>
            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 2rem;
                background: #f8fafc;
                color: #334155;
            }
            
            .header {
                text-align: center;
                margin-bottom: 2rem;
                padding: 2rem;
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                color: white;
                border-radius: 1rem;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            
            .filter-card {
                background: white;
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #f59e0b;
            }
            
            .code-block {
                background: #1e293b;
                color: #e2e8f0;
                padding: 1.5rem;
                border-radius: 0.5rem;
                overflow-x: auto;
                font-family: 'Fira Code', 'Courier New', monospace;
                margin: 1rem 0;
            }
            
            .nav-button {
                background: #f59e0b;
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 0.5rem;
                text-decoration: none;
                display: inline-block;
                margin: 0 0.5rem;
                transition: background 0.2s;
            }
            
            .nav-button:hover {
                background: #d97706;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Metadata Filter Examples</h1>
            <p>Available filters for searching documents</p>
        </div>
        
        <div class="filter-card">
            <h2>📁 Available Metadata Fields</h2>
            <ul>
                <li><strong>source_file:</strong> Original document filename</li>
                <li><strong>page_number:</strong> Page number in the document</li>
                <li><strong>chunk_index:</strong> Position of text chunk</li>
                <li><strong>total_pages:</strong> Total pages in document</li>
                <li><strong>chunk_size:</strong> Size of the text chunk</li>
            </ul>
        </div>
        
        <div class="filter-card">
            <h2>🌐 URL Filter Examples</h2>
            <div class="code-block">
<pre># Filter by source file
/ask?query=pgvector&source_file=Tema 2

# Filter by page number
/ai?query=vectores&page_number=5

# Combined filters
/compare?query=base de datos&source_file=Williams&page_number=10
</pre>
            </div>
        </div>
        
        <div class="filter-card">
            <h2>🐍 Python Code Examples</h2>
            <div class="code-block">
<pre>import requests

# Search with file filter
response = requests.get("http://localhost:8000/ask", {
    "query": "¿Qué es pgvector?",
    "source_file": "Tema 2"
})

# Search with page filter  
response = requests.get("http://localhost:8000/ai", {
    "query": "vectores en bases de datos",
    "page_number": 5
})
</pre>
            </div>
        </div>
        
        <div class="filter-card">
            <h2>📊 Database Queries</h2>
            <div class="code-block">
<pre># PostgreSQL with metadata filter
SELECT content, metadata 
FROM documents 
WHERE metadata-&gt;&gt;'source_file' LIKE '%pgvector%'
ORDER BY embedding &lt;=&gt; %s::vector;

# Qdrant with payload filter
from qdrant_client.http.models import Filter, FieldCondition

client.search(
    collection_name="course_docs_clean",
    query_vector=embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source_file",
                match={"value": "Tema 2"}
            )
        ]
    )
)
</pre>
            </div>
        </div>
        
        <div style="text-align: center; margin: 2rem 0;">
            <a href="/" class="nav-button">🏠 Home</a>
            <a href="/ask?query=pgvector" class="nav-button">🔍 Try Search</a>
            <a href="/demo/pipeline" class="nav-button">📚 Demo</a>
        </div>
    </body>
    </html>
    """


def render_gpu_status_html(data: dict = None) -> str:
    """HTML template for GPU status"""

    # Default data if none provided
    if data is None:
        data = {
            "gpu_available": "N/A",
            "running_on": "CPU",
            "embedding_dimensions": 768,
            "model_type": "E5",
            "embedding_model": "intfloat/multilingual-e5-base",
            "qdrant_status": "🟢",
            "postgres_status": "🟢",
            "pgvector_status": "🟢",
            "ollama_status": "🟢"
        }

    gpu_available = data.get("gpu_available", "N/A")
    running_on = data.get("running_on", "CPU")
    embedding_dims = data.get("embedding_dimensions", 768)
    model_type = data.get("model_type", "E5")
    embedding_model = data.get(
        "embedding_model", "intfloat/multilingual-e5-base")
    qdrant_status = data.get("qdrant_status", "🟢")
    postgres_status = data.get("postgres_status", "🟢")
    pgvector_status = data.get("pgvector_status", "🟢")
    ollama_status = data.get("ollama_status", "🟢")

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GPU Status</title>
        <style>
            body {"{"}
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 2rem;
                background: #f8fafc;
                color: #334155;
            {"}"}
            
            .header {"{"}
                text-align: center;
                margin-bottom: 2rem;
                padding: 2rem;
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                border-radius: 1rem;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
           {"}"}
            
            .status-card {"{"}
                background: white;
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #10b981;
           {"}"}
            
            .status-grid {"{"}
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin: 2rem 0;
           {"}"}
            
            .metric-card {"{"}
                background: #f8fafc;
                border-radius: 0.5rem;
                padding: 1.5rem;
                border: 2px solid #e2e8f0;
                text-align: center;
           {"}"}
            
            .metric-value {"{"}
                font-size: 2rem;
                font-weight: 600;
                color: #10b981;
           {"}"}
            
            .metric-label {"{"}
                color: #64748b;
                margin-top: 0.5rem;
           {"}"}
            
            .warning {"{"}
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 1rem 0;
           {"}"}
            
            .nav-button {"{"}
                background: #10b981;
                color: white;
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 0.5rem;
                text-decoration: none;
                display: inline-block;
                margin: 0 0.5rem;
                transition: background 0.2s;
           {"}"}
            
            .nav-button:hover {"{"}
                background: #059669;
           {"}"}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ GPU Status & System Info</h1>
            <p>Hardware acceleration status for the RAG system</p>
        </div>
        
        <div class="status-card">
            <h2>🖥️ GPU Information</h2>
            <div class="warning">
                <strong>⚠️ Note:</strong> This is a development system. GPU information would be dynamically loaded in production.
            </div>
            
            <div class="status-grid">
                <div class="metric-card">
                    <div class="metric-value">{gpu_available}</div>
                    <div class="metric-label">GPU Available</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{running_on}</div>
                    <div class="metric-label">Running on</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{embedding_dims}</div>
                    <div class="metric-label">Embedding Dimensions</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{model_type}</div>
                    <div class="metric-label">Model Type</div>
                </div>
            </div>
        </div>
        
        <div class="status-card">
            <h2>🔧 Model Configuration</h2>
            <ul>
                <li><strong>Embedding Model:</strong> {embedding_model}</li>
                <li><strong>Vector Dimensions:</strong> {embedding_dims}</li>
                <li><strong>Language Support:</strong> Multilingual (Spanish, English, etc.)</li>
                <li><strong>Context Window:</strong> 512 tokens</li>
            </ul>
        </div>
        
        <div class="status-card">
            <h2>📊 Database Status</h2>
            <div class="status-grid">
                <div class="metric-card">
                    <div class="metric-value">{qdrant_status}</div>
                    <div class="metric-label">Qdrant Status</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{postgres_status}</div>
                    <div class="metric-label">PostgreSQL Status</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{pgvector_status}</div>
                    <div class="metric-label">pgvector Extension</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{ollama_status}</div>
                    <div class="metric-label">Ollama Service</div>
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin: 2rem 0;">
            <a href="/" class="nav-button">🏠 Home</a>
            <a href="/docs" class="nav-button">📚 API Docs</a>
            <a href="/demo/pipeline" class="nav-button">🔬 Demo</a>
        </div>
    </body>
    </html>
    """


def render_home_html() -> str:
    """HTML template for home page"""

    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG System - Qdrant & PostgreSQL Vector Search</title>
        <style>
            :root {
                --primary-color: #2563eb;
                --secondary-color: #1e40af;
                --accent-color: #3b82f6;
                --success-color: #059669;
                --warning-color: #d97706;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
            }
            
            * {
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background: var(--light-bg);
                color: var(--text-color);
            }
            
            .hero {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 4rem 0;
                text-align: center;
                margin-bottom: 3rem;
            }
            
            .hero h1 {
                font-size: 3rem;
                font-weight: 300;
                margin: 0 0 1rem 0;
            }
            
            .hero p {
                font-size: 1.25rem;
                margin: 0 auto;
                max-width: 600px;
                opacity: 0.9;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }
            
            .cards-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 2rem;
                margin-bottom: 3rem;
            }
            
            .feature-card {
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border: 1px solid var(--border-color);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .feature-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            
            .card-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            
            .card-title {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: var(--primary-color);
            }
            
            .quick-actions {
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 3rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            
            .action-buttons {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 1rem;
                margin-top: 1.5rem;
            }
            
            .action-button {
                background: var(--primary-color);
                color: white;
                padding: 1rem 2rem;
                border: none;
                border-radius: 0.75rem;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                font-weight: 500;
                font-size: 1rem;
                transition: background 0.2s, transform 0.2s;
            }
            
            .action-button:hover {
                background: var(--secondary-color);
                transform: translateY(-1px);
            }
            
            .action-button.secondary {
                background: var(--success-color);
            }
            
            .action-button.warning {
                background: var(--warning-color);
            }
            
            .query-form {
                background: white;
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            
            .form-group {
                margin-bottom: 1.5rem;
            }
            
            .form-label {
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 600;
                color: var(--text-color);
            }
            
            .form-input {
                width: 100%;
                padding: 0.75rem;
                border: 2px solid var(--border-color);
                border-radius: 0.5rem;
                font-size: 1rem;
                transition: border-color 0.2s;
            }
            
            .form-input:focus {
                outline: none;
                border-color: var(--primary-color);
            }
            
            .submit-button {
                background: var(--primary-color);
                color: white;
                padding: 0.75rem 2rem;
                border: none;
                border-radius: 0.5rem;
                font-size: 1rem;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s;
            }
            
            .submit-button:hover {
                background: var(--secondary-color);
            }
            
            .footer {
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                border-top: 1px solid var(--border-color);
                margin-top: 3rem;
            }
            
            @media (max-width: 768px) {
                .hero h1 {
                    font-size: 2rem;
                }
                
                .action-buttons {
                    flex-direction: column;
                    align-items: center;
                }
                
                .container {
                    padding: 0 1rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="hero">
            <div class="container">
                <h1>🤖 RAG System</h1>
                <p>Advanced vector search with Qdrant and PostgreSQL+pgvector. Search through educational documents using AI-powered semantic similarity.</p>
            </div>
        </div>
        
        <div class="container">
            <div class="quick-actions">
                <h2>🚀 Try the System</h2>
                <p>Quick search form - ask any question about the documents</p>
                
                <form class="query-form" action="/ai" method="get">
                    <div class="form-group">
                        <label for="query" class="form-label">Ask a question:</label>
                        <input 
                            type="text" 
                            id="query" 
                            name="query" 
                            class="form-input" 
                            placeholder="e.g., ¿Qué es pgvector y cómo se usa?"
                            value=""
                        >
                    </div>
                    <button type="submit" class="submit-button">🤖 Ask AI</button>
                </form>
                
                <div class="action-buttons">
                    <a href="/demo/pipeline" class="action-button">📚 Complete Demo</a>
                    <a href="/ask?query=pgvector" class="action-button secondary">🔍 Search Only</a>
                    <a href="/compare?query=vectores" class="action-button warning">⚖️ Compare DBs</a>
                </div>
            </div>
            
            <div class="cards-grid">
                <div class="feature-card">
                    <div class="card-icon">🔍</div>
                    <h3 class="card-title">Semantic Search</h3>
                    <p>Search documents using meaning, not just keywords. Our system understands context and finds relevant information even when exact words don't match.</p>
                    <p><strong>Features:</strong></p>
                    <ul>
                        <li>Multilingual support (Spanish/English)</li>
                        <li>Context-aware search</li>
                        <li>Metadata filtering</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="card-icon">🤖</div>
                    <h3 class="card-title">AI-Powered Responses</h3>
                    <p>Get intelligent answers generated by AI based on the most relevant document sections. Combines retrieval with generation for comprehensive responses.</p>
                    <p><strong>Capabilities:</strong></p>
                    <ul>
                        <li>Contextual answer generation</li>
                        <li>Source citation</li>
                        <li>Multiple language support</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="card-icon">⚖️</div>
                    <h3 class="card-title">Database Comparison</h3>
                    <p>Compare performance and results between Qdrant (specialized vector DB) and PostgreSQL with pgvector extension side by side.</p>
                    <p><strong>Compare:</strong></p>
                    <ul>
                        <li>Search speed and accuracy</li>
                        <li>Result relevance</li>
                        <li>Different similarity metrics</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="card-icon">📚</div>
                    <h3 class="card-title">Educational Demos</h3>
                    <p>Learn how vector search works with step-by-step explanations and live examples. Perfect for understanding RAG systems.</p>
                    <p><strong>Learn about:</strong></p>
                    <ul>
                        <li>Text embedding process</li>
                        <li>Vector similarity calculations</li>
                        <li>Manual database queries</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="card-icon">🔧</div>
                    <h3 class="card-title">Manual Operations</h3>
                    <p>Learn to create embeddings and perform searches manually. Understand what happens behind the scenes in RAG systems.</p>
                    <p><strong>Manual guides:</strong></p>
                    <ul>
                        <li>Create text embeddings</li>
                        <li>Direct database queries</li>
                        <li>Performance optimization</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="card-icon">📊</div>
                    <h3 class="card-title">System Monitoring</h3>
                    <p>Monitor system performance, GPU usage, and database status. Track search times and accuracy metrics.</p>
                    <p><strong>Monitor:</strong></p>
                    <ul>
                        <li>Search performance metrics</li>
                        <li>Database connection status</li>
                        <li>Model loading status</li>
                    </ul>
                </div>
            </div>
            
            <div class="quick-actions">
                <h2>📖 Documentation & Resources</h2>
                <div class="action-buttons">
                    <a href="/docs" class="action-button">📖 API Documentation</a>
                    <a href="/manual/embed" class="action-button secondary">📊 Manual Embedding</a>
                    <a href="/manual/search" class="action-button secondary">🔍 Manual Search</a>
                    <a href="/filters/examples" class="action-button warning">🔍 Filter Examples</a>
                    <a href="/gpu-status" class="action-button warning">⚡ System Status</a>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>RAG System with Qdrant and PostgreSQL+pgvector | Educational Demo Platform</p>
            <p>Built for advanced database systems course</p>
        </div>
    </body>
    </html>
    """
