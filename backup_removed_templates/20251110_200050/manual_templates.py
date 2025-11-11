"""
HTML templates for manual process explanation routes
"""


def render_manual_embedding_html() -> str:
    """HTML template for manual embedding process explanation"""

    html = """
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
                --accent-color: #a855f7;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
                --code-bg: #1e293b;
                --code-color: #e2e8f0;
                --gradient: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
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

            .header {
                background: var(--gradient);
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
                display: flex;
                align-items: center;
                gap: 0.5rem;
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

            .warning {
                background: #fef3c7;
                border: 1px solid #f59e0b;
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

            .footer {
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                border-top: 1px solid var(--border-color);
                margin-top: 2rem;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>📊 Manual Embedding Process</h1>
                <p>Learn how to create text embeddings manually using different tools</p>
            </div>
        </div>

        <div class="container">
            <div class="step-card">
                <h2 class="step-title">🔧 1. Using Python with sentence-transformers</h2>
                <p>The most common way to generate embeddings programmatically:</p>

                <div class="code-block">
<pre>pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the same model we use in the RAG system
model = SentenceTransformer('intfloat/multilingual-e5-base')

# Your text
text = "¿Qué es pgvector?"

# Generate embedding
embedding = model.encode(text)
print(f"Embedding dimensions: {embedding.shape}")
print(f"First 10 values: {embedding[:10]}")

# Convert to list for JSON storage
embedding_list = embedding.tolist()
</pre>
                </div>

                <div class="explanation">
                    <strong>💡 What happens here:</strong>
                    <ul>
                        <li>The model converts text into a 768-dimensional vector</li>
                        <li>Similar texts will have similar vectors (high cosine similarity)</li>
                        <li>The vector captures semantic meaning, not just keywords</li>
                    </ul>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">🌐 2. Using Ollama API</h2>
                <p>Generate embeddings using the same Ollama service:</p>

                <div class="code-block">
<pre>import requests
import json

# Ollama embeddings API
url = "http://localhost:11434/api/embeddings"

data = {
    "model": "mxbai-embed-large",  # or "nomic-embed-text"
    "prompt": "¿Qué es pgvector?"
}

response = requests.post(url, json=data)
result = response.json()

embedding = result["embedding"]
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
</pre>
                </div>

                <div class="explanation">
                    <strong>💡 Different models produce different dimensions:</strong>
                    <ul>
                        <li><code>mxbai-embed-large</code>: 1024 dimensions</li>
                        <li><code>nomic-embed-text</code>: 768 dimensions</li>
                        <li>Our system uses the E5 model: 768 dimensions</li>
                    </ul>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">🐘 3. Store in PostgreSQL</h2>
                <p>Once you have the embedding, store it in PostgreSQL with pgvector:</p>

                <div class="code-block">
<pre>import psycopg2
import json

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="rag_db",
    user="rag_user",
    password="rag_password"
)
cur = conn.cursor()

# Insert document with embedding
embedding_str = json.dumps(embedding_list)  # Convert list to JSON string

# SQL query with vector insertion
sql = '''
    INSERT INTO documents (content, metadata, embedding)
    VALUES (%s, %s, %s::vector)
'''

cur.execute(sql, (
    "¿Qué es pgvector?",
    json.dumps({"source": "manual", "type": "question"}),
    embedding_str
))

conn.commit()
cur.close()
conn.close()
</pre>
                </div>

                <div class="warning">
                    <strong>⚠️ Important:</strong> Make sure the embedding dimension matches your vector column definition in PostgreSQL!
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">📊 4. Store in Qdrant</h2>
                <p>Alternative: store in Qdrant vector database:</p>

                <div class="code-block">
<pre>from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Create point with embedding
point = PointStruct(
    id=1,  # Unique ID
    vector=embedding_list,
    payload={
        "content": "¿Qué es pgvector?",
        "source": "manual",
        "type": "question"
    }
)

# Insert into collection
client.upsert(
    collection_name="course_docs_clean",
    points=[point]
)
</pre>
                </div>

                <div class="explanation">
                    <strong>💡 Qdrant vs PostgreSQL:</strong>
                    <ul>
                        <li><strong>Qdrant:</strong> Specialized vector database, faster for large-scale vector operations</li>
                        <li><strong>PostgreSQL+pgvector:</strong> Traditional database with vector extension, better for mixed data</li>
                    </ul>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">🔍 5. Manual Search Queries</h2>
                <p>Search for similar documents manually:</p>

                <div class="code-block">
<pre># PostgreSQL similarity search
query_embedding = model.encode("database vectors")
embedding_str = json.dumps(query_embedding.tolist())

cur.execute("""
    SELECT content, metadata,
           1 - (embedding <= > % s: : vector) as similarity
    FROM documents
    ORDER BY embedding <= > % s: : vector
    LIMIT 5


""", (embedding_str, embedding_str))

results = cur.fetchall()

# Qdrant similarity search
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
                <a href="/demo/embedding" class="nav-button">🧪 Try Embedding Demo</a>
            </div>
        </div>

        <div class="footer">
            <p>Manual embedding process for RAG systems</p>
        </div>
    </body>
    </html>
    """

    return html


def render_manual_search_html() -> str:
    """HTML template for manual search process explanation"""

    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manual Search Process</title>
        <style>
            :root {
                --primary-color: #059669;
                --secondary-color: #047857;
                --accent-color: #10b981;
                --light-bg: #f8fafc;
                --border-color: #e2e8f0;
                --text-color: #334155;
                --text-muted: #64748b;
                --card-bg: #ffffff;
                --code-bg: #1e293b;
                --code-color: #e2e8f0;
                --gradient: linear-gradient(135deg, #059669 0%, #047857 100%);
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

            .header {
                background: var(--gradient);
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
                display: flex;
                align-items: center;
                gap: 0.5rem;
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

            .comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 2rem;
                margin: 2rem 0;
            }

            @media (max-width: 768px) {
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

            .nav-button:hover {
                background: var(--secondary-color);
                transform: translateY(-1px);
            }

            .footer {
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                border-top: 1px solid var(--border-color);
                margin-top: 2rem;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>🔍 Manual Search Process</h1>
                <p>Learn how to perform semantic search manually in both databases</p>
            </div>
        </div>

        <div class="container">
            <div class="step-card">
                <h2 class="step-title">🎯 1. Generate Query Embedding</h2>
                <p>First, convert your search query into a vector using the same model:</p>

                <div class="code-block">
<pre>from sentence_transformers import SentenceTransformer
import json

# Load the same model used for document embeddings
model = SentenceTransformer('intfloat/multilingual-e5-base')

# Your search query
query = "¿Qué es pgvector y cómo se usa?"

# Generate query embedding (768 dimensions)
query_embedding = model.encode(query)
query_embedding_list = query_embedding.tolist()

print(f"Query: {query}")
print(f"Embedding dimensions: {len(query_embedding_list)}")
print(f"First 10 values: {query_embedding_list[:10]}")
</pre>
                </div>

                <div class="explanation">
                    <strong>💡 Key Point:</strong> Use the SAME model that was used to create document embeddings, otherwise similarity won't work correctly!
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">🔄 2. Database Search Comparison</h2>
                <p>Now search in both databases and compare the results:</p>

                <div class="comparison">
                    <div class="db-card postgres">
                        <h3 style="color: #f59e0b; margin-top: 0;">🐘 PostgreSQL + pgvector</h3>
                        <div class="code-block">
<pre>import psycopg2
import json
import time

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="rag_db",
    user="rag_user",
    password="rag_password"
)
cur = conn.cursor()

# Convert embedding to JSON string
embedding_str = json.dumps(query_embedding_list)

# Measure search time
start_time = time.time()

# Similarity search using cosine distance
cur.execute("""
    SELECT
        content,
        metadata,
        1 - (embedding <= > % s: : vector) as similarity,
        embedding <= > % s: : vector as distance
    FROM documents
    WHERE metadata -> >'source_file' IS NOT NULL
    ORDER BY embedding <= > % s: : vector
    LIMIT 5


""", (embedding_str, embedding_str, embedding_str))

results = cur.fetchall()
search_time = (time.time() - start_time) * 1000

print(f"PostgreSQL search time: {search_time:.2f}ms")
for i, (content, metadata, similarity, distance) in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Distance: {distance:.4f}")
    print(f"  Content: {content[:100]}...")
    print(f"  Metadata: {metadata}")
    print()
</pre>
                        </div>
                    </div>

                    <div class="db-card qdrant">
                        <h3 style="color: #6366f1; margin-top: 0;">📊 Qdrant Vector DB</h3>
                        <div class="code-block">
<pre>from qdrant_client import QdrantClient
import time

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Measure search time
start_time = time.time()

# Similarity search
search_result = client.search(
    collection_name="course_docs_clean",
    query_vector=query_embedding_list,
    limit=5,
    with_payload=True,
    with_vectors=False
)

search_time = (time.time() - start_time) * 1000

print(f"Qdrant search time: {search_time:.2f}ms")
for i, hit in enumerate(search_result):
    print(f"Result {i+1}:")
    print(f"  Score: {hit.score:.4f}")
    print(f"  Content: {hit.payload['content'][:100]}...")
    print(f"  Source: {hit.payload.get('source_file', 'N/A')}")
    print(f"  Page: {hit.payload.get('page_number', 'N/A')}")
    print()
</pre>
                        </div>
                    </div>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">📊 3. Understanding Similarity Metrics</h2>
                <p>Different databases use different similarity metrics:</p>

                <div class="comparison">
                    <div class="db-card postgres">
                        <h4 style="color: #f59e0b;">PostgreSQL pgvector</h4>
                        <ul>
                            <li><strong>Distance:</strong> Cosine distance (0 = identical, 2 = opposite)</li>
                            <li><strong>Similarity:</strong> 1 - distance (1 = identical, -1 = opposite)</li>
                            <li><strong>Operator:</strong> <code>&lt;=&gt;</code> for cosine distance</li>
                        </ul>

                        <div class="code-block">
<pre># Other distance operators available:
# <-> for L2 (Euclidean) distance
# <#> for inner product (negative dot product)

# Example with L2 distance:
cur.execute("""
    SELECT content, embedding < -> % s: : vector as l2_distance
    FROM documents
    ORDER BY embedding < -> % s: : vector
    LIMIT 3
""", (embedding_str, embedding_str))
</pre>
                        </div>
                    </div>

                    <div class="db-card qdrant">
                        <h4 style="color: #6366f1;">Qdrant</h4>
                        <ul>
                            <li><strong>Score:</strong> Cosine similarity (1 = identical, 0 = orthogonal)</li>
                            <li><strong>Default:</strong> Cosine distance metric</li>
                            <li><strong>Range:</strong> 0.0 to 1.0 (higher = more similar)</li>
                        </ul>

                        <div class="code-block">
<pre># Qdrant also supports other metrics:
from qdrant_client.http.models import Distance

# When creating collection, you can specify:
# Distance.COSINE (default)
# Distance.EUCLID (L2 distance)
# Distance.DOT (dot product)

# Search with specific parameters:
search_result = client.search(
    collection_name="course_docs_clean",
    query_vector=query_embedding_list,
    limit=5,
    score_threshold=0.7  # Only results with score > 0.7
)
</pre>
                        </div>
                    </div>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">⚡ 4. Advanced Search Techniques</h2>
                <p>Enhance your searches with filters and reranking:</p>

                <div class="code-block">
<pre># PostgreSQL with metadata filters
cur.execute("""
    SELECT content, metadata, 1 - (embedding <= > % s: : vector) as similarity
    FROM documents
    WHERE metadata -> >'source_file' LIKE '%%pgvector%%'
    AND (1 - (embedding <= > % s: : vector)) > 0.7
    ORDER BY embedding <= > % s: : vector
    LIMIT 5
""", (embedding_str, embedding_str, embedding_str))

# Qdrant with payload filters
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

search_result = client.search(
    collection_name="course_docs_clean",
    query_vector=query_embedding_list,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source_file",
                match=MatchValue(value="pgvector")
            )
        ]
    ),
    limit=5,
    score_threshold=0.7
)

# Rerank results by length preference
def rerank_by_length(results, preferred_length=500):
    """Rerank results preferring chunks close to preferred length"""
    for result in results:
        content_length = len(result['content'])
        length_penalty = abs(
            content_length - preferred_length) / preferred_length
        result['adjusted_score'] = result['similarity'] * \
            (1 - length_penalty * 0.1)

    return sorted(results, key=lambda x: x['adjusted_score'], reverse=True)
</pre>
                </div>
            </div>

            <div class="step-card">
                <h2 class="step-title">📈 5. Performance Comparison</h2>
                <p>Benchmark both databases for your use case:</p>

                <div class="code-block">
<pre>import time
import statistics

def benchmark_search(query_embedding_list, iterations=10):
    """Benchmark search performance"""

    # PostgreSQL benchmark
    pg_times = []
    for _ in range(iterations):
        start = time.time()
        cur.execute("""
            SELECT content FROM documents
            ORDER BY embedding <= > % s: : vector LIMIT 5
        """, (json.dumps(query_embedding_list),))
        results = cur.fetchall()
        pg_times.append((time.time() - start) * 1000)
    
    # Qdrant benchmark  
    qdrant_times = []
    for _ in range(iterations):
        start = time.time()
        search_result = client.search(
            collection_name="course_docs_clean",
            query_vector=query_embedding_list,
            limit=5
        )
        qdrant_times.append((time.time() - start) * 1000)
    
    print("Performance Results:")
    print(f"PostgreSQL - Avg: {statistics.mean(pg_times):.2f}ms, "
          f"Min: {min(pg_times):.2f}ms, Max: {max(pg_times):.2f}ms")
    print(f"Qdrant - Avg: {statistics.mean(qdrant_times):.2f}ms, "
          f"Min: {min(qdrant_times):.2f}ms, Max: {max(qdrant_times):.2f}ms")

# Run benchmark
benchmark_search(query_embedding_list)
</pre>
                </div>
            </div>
            
            <div class="navigation">
                <a href="/" class="nav-button">🏠 Home</a>
                <a href="/manual/embed" class="nav-button">📊 Manual Embedding</a>
                <a href="/compare?query=pgvector" class="nav-button">⚖️ Live Comparison</a>
                <a href="/demo/pipeline" class="nav-button">📚 Complete Demo</a>
            </div>
        </div>
        
        <div class="footer">
            <p>Manual search process for vector databases</p>
        </div>
    </body>
    </html>
    """
    
    return html