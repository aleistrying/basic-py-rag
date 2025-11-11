"""
HTML templates for manual process explanation routes
"""


def render_manual_embedding_html(data: dict = None) -> str:
    """HTML template for manual embedding process explanation"""

    # Default data if none provided
    if data is None:
        data = {
            "query": "bases de datos vectoriales",
            "expanded_query": "bases de datos vectoriales",
            "embedding_dimensions": 768,
            "model_name": "intfloat/multilingual-e5-base",
            "sample_values": "[0.1234, -0.5678, 0.9876, ...]"
        }

    query = data.get("query", "texto de ejemplo")
    expanded_query = data.get("expanded_query", query)
    dimensions = data.get("embedding_dimensions", 768)
    model = data.get("model_name", "intfloat/multilingual-e5-base")
    sample_values = data.get("sample_values", "[valores de ejemplo]")

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manual Embedding Process</title>
        <style>
            :root {
                --primary-color: #8b5cf6;
                --secondary-color: #7c3aed;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
                --code-bg: #1e293b;
                --code-color: #e2e8f0;
            }

            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background: var(--light-bg);
                color: var(--text-color);
            }

            .header {
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                color: white;
                padding: 2rem 0;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }

            .header h1 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
            }

            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 0 1rem;
            }

            .step-card {
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border-left: 4px solid var(--primary-color);
            }

            .step-title {
                color: var(--primary-color);
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }

            .code-block {
                background: var(--code-bg);
                color: var(--code-color);
                padding: 1.5rem;
                border-radius: 0.5rem;
                overflow-x: auto;
                font-family: 'Fira Code', 'Courier New', monospace;
                font-size: 0.875rem;
                line-height: 1.5;
                margin: 1rem 0;
            }

            .explanation {
                background: #f0f9ff;
                border: 1px solid #0ea5e9;
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 1rem 0;
            }

            .navigation {
                text-align: center;
                margin: 2rem 0;
            }

            .nav-button {
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
            }

            .nav-button:hover {
                background: var(--secondary-color);
                transform: translateY(-1px);
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>📊 Manual Embedding Process</h1>
                <p>Learn how to create text embeddings manually</p>
            </div>
        </div>

        <div class="container">
            <div class="step-card">
                <h2 class="step-title">🔧 1. Using Python with sentence-transformers</h2>
                <p>Generate embeddings programmatically:</p>

                <div class="code-block">
<pre>pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the same model we use in the RAG system
model = SentenceTransformer('{model}')

# Your text
text = "{query}"

# Generate embedding
embedding = model.encode(text)
print(f"Dimensions: {{embedding.shape}}")
print(f"First 10: {{embedding[:10]}}")

# Convert to list for JSON
embedding_list = embedding.tolist()
</pre>
                </div>

                <div class="explanation">
                    <strong>💡 Key points:</strong>
                    <ul>
                        <li>{dimensions}-dimensional vector</li>
                        <li>Similar texts have similar vectors</li>
                        <li>Captures semantic meaning</li>
                        <li>Sample values: {sample_values}</li>
                    </ul>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">🐘 2. Store in PostgreSQL</h2>
                <p>Store embeddings in PostgreSQL with pgvector:</p>

                <div class="code-block">
<pre>import psycopg2
import json

# Connect
conn = psycopg2.connect(
    host="localhost",
    database="rag_db",
    user="rag_user",
    password="rag_password"
)
cur = conn.cursor()

# Insert with embedding
embedding_str = json.dumps(embedding_list)

sql = '''
INSERT INTO documents (content, metadata, embedding)
VALUES (%s, %s, %s::vector)
'''

cur.execute(sql, (
    "{query}",
    json.dumps({{"source": "manual"}}),
    embedding_str
))

conn.commit()
</pre>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">📊 3. Store in Qdrant</h2>
                <p>Alternative: store in Qdrant vector database:</p>

                <div class="code-block">
<pre>from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Connect
client = QdrantClient("localhost", port=6333)

# Create point
point = PointStruct(
    id=1,
    vector=embedding_list,
    payload={{
        "content": "{query}",
        "source": "manual"
    }}
)

# Insert
client.upsert(
    collection_name="course_docs_clean",
    points=[point]
)
</pre>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">🔍 4. Manual Search</h2>
                <p>Search for similar documents:</p>

                <div class="code-block">
# PostgreSQL search time
query_embedding = model.encode("{query}")
embedding_str = json.dumps(query_embedding.tolist())

sql = '''
SELECT content, metadata,
       1 - (embedding &#60;=&#62; %s::vector) as similarity
FROM documents
ORDER BY embedding &#60;=&#62; %s::vector
LIMIT 5
'''

cur.execute(sql, (embedding_str, embedding_str))
results = cur.fetchall()
search_time = (time.time() - start_time) * 1000  # {search_time}ms

# Example: {search_time}ms
print(f"PostgreSQL search time: {{search_time:.2f}}ms")
print(f"Found {{len(results)}} results")  # Example: {results_count} results

# Qdrant search
search_result = client.search(
    collection_name="course_docs_clean",
    query_vector=query_embedding.tolist(),
    limit=5
)
</pre>
                </div>
            </div>

            <div class="navigation">
                <a href="/" class="nav-button">🏠 Home</a>
                <a href="/manual/search" class="nav-button">🔍 Manual Search</a>
                <a href="/demo/pipeline" class="nav-button">📚 Complete Demo</a>
                <a href="/demo/embedding" class="nav-button">🧪 Try Demo</a>
            </div>
        </div>
    </body>
    </html>
    """


def render_manual_search_html(data: dict = None) -> str:
    """HTML template for manual search process explanation"""

    # Default data if none provided
    if data is None:
        data = {
            "query": "database vectors",
            "search_time_ms": "12.5",
            "results_count": "5"
        }

    query = data.get("query", "example query")
    search_time = data.get("search_time_ms", "0.0")
    results_count = data.get("results_count", "0")

    return """
    <!DOCTYPE html >
    <html lang = "es" >
    <head >
        <meta charset = "UTF-8" >
        <meta name = "viewport" content = "width=device-width, initial-scale=1.0" >
        <title > Manual Search Process < /title >
        <style >
            :root {
                --primary-color: #059669;
                --secondary-color: #047857;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
                --code-bg: #1e293b;
                --code-color: #e2e8f0;
            }

            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background: var(--light-bg);
                color: var(--text-color);
            }

            .header {
                background: linear-gradient(135deg, #059669 0%, #047857 100%);
                color: white;
                padding: 2rem 0;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px - 1px rgba(0, 0, 0, 0.1);
            }

            .header h1 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 300;
            }

            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 0 1rem;
            }

            .step-card {
                background: var(--card-bg);
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px - 1px rgba(0, 0, 0, 0.1);
                border-left: 4px solid var(--primary-color);
            }

            .step-title {
                color: var(--primary-color);
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }

            .code-block {
                background: var(--code-bg);
                color: var(--code-color);
                padding: 1.5rem;
                border-radius: 0.5rem;
                overflow-x: auto;
                font-family: 'Fira Code', 'Courier New', monospace;
                font-size: 0.875rem;
                line-height: 1.5;
                margin: 1rem 0;
            }

            .comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                margin: 2rem 0;
            }

            @ media(max-width: 768px) {
                .comparison {
                    grid-template-columns: 1fr;
                }
            }

            .db-card {
                background: var(--light-bg);
                border-radius: 0.5rem;
                padding: 1.5rem;
                border: 2px solid var(--border-color);
            }

            .db-card.qdrant {
                border-color: #6366f1;
            }

            .db-card.postgres {
                border-color: #f59e0b;
            }

            .navigation {
                text-align: center;
                margin: 2rem 0;
            }

            .nav-button {
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
            }

            .nav-button: hover {
                background: var(--secondary-color);
                transform: translateY(-1px);
            }
        < /style >
    < / head >
    < body >
        < div class= "header" >
            < div class= "container" >
                < h1 >🔍 Manual Search Process < /h1 >
                < p > Learn how to perform semantic search manually < /p >
            < / div >
        < / div >

        < div class= "container" >
            < div class= "step-card" >
                < h2 class = "step-title" >🎯 1. Generate Query Embedding < /h2 >
                < p > Convert your search query into a vector: < /p >

                < div class= "code-block" >
< pre > from sentence_transformers import SentenceTransformer

# Load model
model= SentenceTransformer('intfloat/multilingual-e5-base')

# Search query
query= "¿Qué es pgvector?"

# Generate embedding
query_embedding= model.encode(query)
query_list= query_embedding.tolist()

print(f"Query: {query}")
print(f"Dimensions: {len(query_list)}")
< /pre >
                < / div >
            < / div >

            < div class= "step-card" >
                < h2 class = "step-title" >🔄 2. Database Comparison < /h2 >
                < p > Search in both databases: < /p >

                < div class= "comparison" >
                    < div class= "db-card postgres" >
                        < h3 style = "color: #f59e0b;" >🐘 PostgreSQL < /h3 >
                        < div class= "code-block" >
< pre >  # PostgreSQL search
import psycopg2
import json

conn= psycopg2.connect(...)
cur= conn.cursor()

embedding_str= json.dumps(query_list)

sql= '''
SELECT content, metadata,
       1 - (embedding &#60;=&#62; %s::vector) as similarity
FROM documents
ORDER BY embedding &#60;=&#62; %s::vector
LIMIT 5
'''

cur.execute(sql, (embedding_str, embedding_str))
results= cur.fetchall()
< /pre >
                        < / div >
                    < / div >

                    < div class= "db-card qdrant" >
                        < h3 style = "color: #6366f1;" >📊 Qdrant < /h3 >
                        < div class= "code-block" >
< pre >  # Qdrant search
from qdrant_client import QdrantClient

client= QdrantClient("localhost", port=6333)

search_result= client.search(
    collection_name="course_docs_clean",
    query_vector=query_list,
    limit=5,
    with_payload=True
)

for hit in search_result:
    print(f"Score: {hit.score}")
    print(f"Content: {hit.payload['content']}")
< /pre >
                        < / div >
                    < / div >
                < / div >
            < / div >

            < div class= "step-card" >
                < h2 class = "step-title" >📊 3. Similarity Metrics < /h2 >
                < p > Understanding different similarity measures: < /p >

                < div class= "comparison" >
                    < div class= "db-card postgres" >
                        < h4 style= "color: #f59e0b;" > PostgreSQL pgvector < /h4 >
                        < ul >
                            # 60;=&#62; (cosine)</li>
                            < li > <strong > Distance: < /strong > &
                            < li > <strong > Range: < /strong > 0-2 (lower=similar) < /li >
                            < li > <strong > Similarity: < /strong > 1 - distance < /li >
                        < / ul >
                    < / div >

                    < div class= "db-card qdrant" >
                        < h4 style= "color: #6366f1;" > Qdrant < /h4 >
                        < ul >
                            < li > <strong > Score: < /strong > Cosine similarity < /li >
                            < li > <strong > Range: < /strong > 0-1 (higher=similar) < /li >
                            < li > <strong > Default: < /strong > Cosine metric < /li >
                        < / ul >
                    < / div >
                < / div >
            < / div >

            < div class= "navigation" >
                < a href = "/" class = "nav-button" >🏠 Home < /a >
                < a href = "/manual/embed" class = "nav-button" >📊 Manual Embedding < /a >
                < a href = "/compare?query=pgvector" class = "nav-button" >⚖️ Live Comparison < /a >
                < a href = "/demo/pipeline" class = "nav-button" >📚 Complete Demo < /a >
            < / div >
        < / div >
    < / body >
    < / html >
    """.format(query=query, search_time=search_time, results_count=results_count)
