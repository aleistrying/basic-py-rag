"""
Comprehensive RAG Pipeline Demo
Shows step-by-step process from text to vector search with examples
"""
import json
import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import requests
import psycopg2
import os
import logging

logger = logging.getLogger(__name__)

# Simple settings without pydantic dependency
PGVECTOR_DATABASE_URL = os.getenv(
    "PGVECTOR_DATABASE_URL", "postgresql://pguser:pgpass@localhost:5432/vectordb")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


class RAGPipelineDemo:
    def __init__(self):
        self.model = None
        self.example_text = """
        PostgreSQL con pgvector es una extensión que permite almacenar y buscar vectores de alta dimensionalidad
        directamente en PostgreSQL. Esta tecnología es fundamental para aplicaciones de inteligencia artificial
        que requieren búsquedas semánticas eficientes. Los vectores pueden representar embeddings de texto,
        imágenes o cualquier otro tipo de datos que se puedan convertir en representaciones numéricas.
        """

    def step_1_parse_text(self) -> Dict[str, Any]:
        """Step 1: Parse and prepare text"""
        raw_text = self.example_text

        # Basic text cleaning
        cleaned_text = re.sub(r'\s+', ' ', raw_text.strip())

        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "step": "1. Text Parsing",
            "description": "Parse raw text and split into processable chunks",
            "input": raw_text,
            "output": sentences,
            "code_example": """
# Text parsing code
text = raw_document_text
cleaned_text = re.sub(r'\\s+', ' ', text.strip())
sentences = re.split(r'[.!?]+', cleaned_text)
            """,
            "explanation": "We clean whitespace and split text into semantic units (sentences) for better embedding quality."
        }

    def step_2_clean_text(self, sentences: List[str]) -> Dict[str, Any]:
        """Step 2: Advanced text cleaning and chunking"""

        def clean_sentence(text):
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^\w\s\.,;:\-]', '', text)
            # Normalize case
            return text.strip()

        cleaned_sentences = [clean_sentence(s)
                             for s in sentences if len(s) > 10]

        # Create chunks with overlap for better context
        chunks = []
        for i, sentence in enumerate(cleaned_sentences):
            chunk = sentence
            # Add context from previous sentence if available
            if i > 0:
                chunk = cleaned_sentences[i-1] + " " + chunk
            chunks.append(chunk)

        return {
            "step": "2. Text Cleaning & Chunking",
            "description": "Clean text and create overlapping chunks for better context",
            "input": sentences,
            "cleaned_sentences": cleaned_sentences,
            "output": chunks,
            "code_example": """
def clean_sentence(text):
    text = re.sub(r'\\s+', ' ', text)
    text = re.sub(r'[^\\w\\s\\.,;:\\-]', '', text)
    return text.strip()

# Create overlapping chunks
chunks = []
for i, sentence in enumerate(sentences):
    chunk = sentence
    if i > 0:
        chunk = sentences[i-1] + " " + chunk
    chunks.append(chunk)
            """,
            "explanation": "We clean text and create overlapping chunks to preserve context between related sentences."
        }

    def step_3_create_embeddings(self, chunks: List[str]) -> Dict[str, Any]:
        """Step 3: Convert text to vector embeddings"""

        if not self.model:
            self.model = SentenceTransformer('intfloat/multilingual-e5-base')

        # Add E5 prefix for better embedding quality
        prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]

        # Generate embeddings
        embeddings = self.model.encode(prefixed_chunks)

        # Convert to list for JSON serialization
        embeddings_list = [emb.tolist() for emb in embeddings]

        return {
            "step": "3. Embedding Generation",
            "description": "Convert text chunks to high-dimensional vectors using E5 model",
            "input": chunks,
            "prefixed_chunks": prefixed_chunks,
            "output": {
                "embeddings_shape": f"{len(embeddings_list)} vectors x {len(embeddings_list[0])} dimensions",
                # First 10 dimensions
                "sample_embedding": embeddings_list[0][:10],
                "full_embeddings": embeddings_list
            },
            "code_example": """
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-base')
prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
embeddings = model.encode(prefixed_chunks)

# Each embedding is 768 dimensions
print(f"Shape: {embeddings.shape}")  # (n_chunks, 768)
            """,
            "explanation": "We use E5 multilingual model to convert text into 768-dimensional vectors. The 'passage:' prefix helps the model understand this is document content."
        }

    def step_4_query_processing(self, query: str) -> Dict[str, Any]:
        """Step 4: Process user query into embedding"""

        if not self.model:
            self.model = SentenceTransformer('intfloat/multilingual-e5-base')

        # Add E5 prefix for queries
        prefixed_query = f"query: {query}"

        # Generate query embedding
        query_embedding = self.model.encode([prefixed_query])[0]
        query_embedding_list = query_embedding.tolist()

        return {
            "step": "4. Query Processing",
            "description": "Convert user query into same vector space as documents",
            "input": query,
            "prefixed_query": prefixed_query,
            "output": {
                "query_embedding_shape": f"1 vector x {len(query_embedding_list)} dimensions",
                "sample_dimensions": query_embedding_list[:10],
                "full_embedding": query_embedding_list
            },
            "code_example": """
query = "¿Qué es pgvector?"
prefixed_query = f"query: {query}"
query_embedding = model.encode([prefixed_query])[0]

# Query embedding has same dimensions as document embeddings
print(f"Query shape: {query_embedding.shape}")  # (768,)
            """,
            "explanation": "We process the query using the same model and add 'query:' prefix to help the model understand this is a search query."
        }

    def step_5_vector_search_qdrant(self, query_embedding: List[float], limit: int = 3) -> Dict[str, Any]:
        """Step 5: Perform vector search in Qdrant with curl examples"""

        # Prepare Qdrant search payload
        search_payload = {
            "vector": query_embedding,
            "limit": limit,
            "with_payload": True,
            "with_vector": True
        }

        # Generate curl command for direct API access
        curl_command = f"""
curl -X POST '{QDRANT_URL}/collections/course_docs_clean/points/search' \\
-H 'Content-Type: application/json' \\
-d '{json.dumps(search_payload, indent=2)}'
        """.strip()

        try:
            # Perform actual search
            response = requests.post(
                f"{QDRANT_URL}/collections/course_docs_clean/points/search",
                json=search_payload,
                timeout=10
            )

            search_results = response.json() if response.status_code == 200 else {
                "error": f"Status {response.status_code}"}

        except Exception as e:
            search_results = {"error": str(e)}

        return {
            "step": "5. Vector Search (Qdrant)",
            "description": "Search for similar vectors using cosine similarity",
            "input": {
                "query_vector_dimensions": len(query_embedding),
                "search_limit": limit,
                "collection": "course_docs_clean"
            },
            "curl_command": curl_command,
            "search_payload": search_payload,
            "output": search_results,
            "code_example": """
import requests

search_payload = {
    "vector": query_embedding.tolist(),
    "limit": 3,
    "with_payload": True,
    "with_vector": True
}

response = requests.post(
    f"{qdrant_url}/collections/course_docs_clean/points/search",
    json=search_payload
)

results = response.json()
            """,
            "explanation": "Qdrant performs cosine similarity search to find the most semantically similar document chunks to our query."
        }

    def step_6_vector_search_pgvector(self, query_embedding: List[float], limit: int = 3) -> Dict[str, Any]:
        """Step 6: Perform vector search in PostgreSQL with pgvector"""

        # Format embedding for PostgreSQL
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # SQL query with vector similarity
        sql_query = f"""
SELECT 
    id,
    content,
    metadata,
    embedding <=> '{embedding_str}'::vector as distance
FROM docs_clean 
ORDER BY embedding <=> '{embedding_str}'::vector 
LIMIT {limit};
        """

        # Equivalent curl command for PostgreSQL REST API (if available)
        curl_command = f"""
# PostgreSQL vector search (via psql or REST API)
psql "{PGVECTOR_DATABASE_URL}" -c "
SELECT id, content, metadata, 
       embedding <=> '{embedding_str[:50]}...'::vector as distance
FROM docs_clean 
ORDER BY embedding <=> '{embedding_str[:50]}...'::vector 
LIMIT {limit};"
        """.strip()

        try:
            # Execute search
            conn = psycopg2.connect(PGVECTOR_DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()

            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "distance": float(row[3])
                })

            cursor.close()
            conn.close()

        except Exception as e:
            formatted_results = {"error": str(e)}

        return {
            "step": "6. Vector Search (PostgreSQL pgvector)",
            "description": "Search using PostgreSQL pgvector extension with cosine distance",
            "input": {
                "query_vector_dimensions": len(query_embedding),
                "search_limit": limit,
                "table": "docs_clean"
            },
            "sql_query": sql_query,
            "curl_command": curl_command,
            "output": formatted_results,
            "code_example": """
import psycopg2

# Format embedding for PostgreSQL
embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

# Execute vector similarity search
cursor.execute(\"\"\"
    SELECT id, content, metadata,
           embedding <=> %s::vector as distance
    FROM docs_clean 
    ORDER BY embedding <=> %s::vector 
    LIMIT %s
\"\"\", (embedding_str, embedding_str, limit))

results = cursor.fetchall()
            """,
            "explanation": "PostgreSQL pgvector uses the <=> operator for cosine distance. Lower distances indicate higher similarity."
        }

    def step_7_similarity_math(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> Dict[str, Any]:
        """Step 7: Explain the mathematical calculations behind similarity"""

        # Convert to numpy for calculations
        query_vec = np.array(query_embedding)

        similarity_calculations = []

        for i, doc_emb in enumerate(doc_embeddings[:3]):  # Show first 3
            doc_vec = np.array(doc_emb)

            # Cosine similarity calculation
            dot_product = np.dot(query_vec, doc_vec)
            query_norm = np.linalg.norm(query_vec)
            doc_norm = np.linalg.norm(doc_vec)
            cosine_similarity = dot_product / (query_norm * doc_norm)

            # Cosine distance (used by both Qdrant and pgvector)
            cosine_distance = 1 - cosine_similarity

            similarity_calculations.append({
                "document_index": i,
                "dot_product": float(dot_product),
                "query_norm": float(query_norm),
                "doc_norm": float(doc_norm),
                "cosine_similarity": float(cosine_similarity),
                "cosine_distance": float(cosine_distance),
                "explanation": f"Similarity = {dot_product:.3f} / ({query_norm:.3f} * {doc_norm:.3f}) = {cosine_similarity:.3f}"
            })

        return {
            "step": "7. Similarity Mathematics",
            "description": "Mathematical calculations behind vector similarity search",
            "input": {
                "query_vector_norm": float(np.linalg.norm(query_vec)),
                "calculation_method": "Cosine Similarity"
            },
            "calculations": similarity_calculations,
            "formulas": {
                "cosine_similarity": "cos(θ) = (A · B) / (||A|| × ||B||)",
                "cosine_distance": "distance = 1 - cosine_similarity",
                "dot_product": "A · B = Σ(ai × bi)",
                "vector_norm": "||A|| = √(Σ(ai²))"
            },
            "code_example": """
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def cosine_distance(vec1, vec2):
    return 1 - cosine_similarity(vec1, vec2)

# Calculate similarity
similarity = cosine_similarity(query_embedding, doc_embedding)
distance = cosine_distance(query_embedding, doc_embedding)
            """,
            "explanation": "Cosine similarity measures the angle between vectors. Values closer to 1 indicate higher similarity. Both databases convert this to distance (1 - similarity) where lower values indicate better matches."
        }

    def step_8_result_ranking(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Rank and present final results"""

        if "error" in search_results:
            ranked_results = {
                "error": "Could not rank results due to search error"}
        else:
            try:
                # Extract and sort results by score/distance
                if "result" in search_results:  # Qdrant format
                    results = search_results["result"]
                    sorted_results = sorted(
                        results, key=lambda x: x.get("score", 0), reverse=True)
                else:  # pgvector format
                    sorted_results = sorted(
                        search_results, key=lambda x: x.get("distance", float('inf')))

                # Add ranking information
                ranked_results = []
                for i, result in enumerate(sorted_results):
                    rank_info = {
                        "rank": i + 1,
                        "result": result,
                        "relevance": "High" if i < 1 else "Medium" if i < 3 else "Low"
                    }
                    ranked_results.append(rank_info)

            except Exception as e:
                ranked_results = {"error": f"Ranking error: {str(e)}"}

        return {
            "step": "8. Result Ranking & Presentation",
            "description": "Rank search results by relevance and prepare final response",
            "input": search_results,
            "output": ranked_results,
            "ranking_criteria": {
                "primary": "Cosine similarity score (Qdrant) or distance (pgvector)",
                "secondary": "Content quality and completeness",
                "presentation": "Top 3 results with metadata and source attribution"
            },
            "code_example": """
# Rank results by similarity score
def rank_results(results, score_key="score", reverse=True):
    return sorted(results, key=lambda x: x.get(score_key, 0), reverse=reverse)

# Add relevance classification
for i, result in enumerate(ranked_results):
    if i == 0:
        result["relevance"] = "High"
    elif i < 3:
        result["relevance"] = "Medium"
    else:
        result["relevance"] = "Low"
            """,
            "explanation": "Results are ranked by similarity score and classified by relevance. The top results are used to generate the final AI response with proper source attribution."
        }


def create_demo_html(demo_steps: List[Dict[str, Any]], query: str) -> str:
    """Create comprehensive HTML demo showing all pipeline steps"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>🔬 RAG Pipeline Demo: {query}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%); 
            color: #e1e1e1; 
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .step-container {{
            margin-bottom: 30px;
            padding: 25px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border-left: 4px solid #4CAF50;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .step-header {{
            color: #4CAF50;
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        
        .step-number {{
            background: #4CAF50;
            color: #000;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
        }}
        
        .description {{
            color: #b3b3b3;
            margin-bottom: 20px;
            font-style: italic;
        }}
        
        .code-block {{
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        .json-block {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.85em;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .curl-command {{
            background: #2d1b69;
            border: 1px solid #5a47a8;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            color: #e1e1e1;
        }}
        
        .explanation {{
            background: rgba(76, 175, 80, 0.1);
            border-left: 3px solid #4CAF50;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        
        .input-output {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        
        .input-section, .output-section {{
            background: rgba(255,255,255,0.02);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .section-title {{
            color: #4CAF50;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .math-formula {{
            background: rgba(255, 193, 7, 0.1);
            border: 1px solid #FFC107;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            text-align: center;
        }}
        
        .navigation {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .nav-link {{
            display: block;
            color: #4CAF50;
            text-decoration: none;
            padding: 5px 0;
            font-size: 0.9em;
        }}
        
        .nav-link:hover {{
            color: #66BB6A;
        }}
        
        @media (max-width: 768px) {{
            .input-output {{
                grid-template-columns: 1fr;
            }}
            .navigation {{
                position: relative;
                top: auto;
                right: auto;
                margin-bottom: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Comprehensive RAG Pipeline Demo</h1>
        <p><strong>Query:</strong> "{query}"</p>
        <p>Complete step-by-step analysis from text to vector search</p>
    </div>
    
    <div class="navigation">
        <strong>Quick Navigation:</strong>
        <a href="#step1" class="nav-link">1. Parse Text</a>
        <a href="#step2" class="nav-link">2. Clean Text</a>
        <a href="#step3" class="nav-link">3. Embeddings</a>
        <a href="#step4" class="nav-link">4. Query Process</a>
        <a href="#step5" class="nav-link">5. Qdrant Search</a>
        <a href="#step6" class="nav-link">6. pgvector Search</a>
        <a href="#step7" class="nav-link">7. Math</a>
        <a href="#step8" class="nav-link">8. Ranking</a>
    </div>
"""

    for i, step in enumerate(demo_steps, 1):
        step_id = f"step{i}"

        html += f"""
    <div id="{step_id}" class="step-container">
        <div class="step-header">
            <div class="step-number">{i}</div>
            {step['step']}
        </div>
        
        <div class="description">{step['description']}</div>
        
        <div class="input-output">
            <div class="input-section">
                <div class="section-title">📥 Input</div>
                <div class="json-block">{json.dumps(step.get('input', {}), indent=2, ensure_ascii=False)}</div>
            </div>
            
            <div class="output-section">
                <div class="section-title">📤 Output</div>
                <div class="json-block">{json.dumps(step.get('output', {}), indent=2, ensure_ascii=False)}</div>
            </div>
        </div>
        
        <div class="explanation">
            <strong>💡 Explanation:</strong> {step.get('explanation', 'No explanation provided')}
        </div>
        
        <div class="section-title">💻 Code Example</div>
        <div class="code-block">{step.get('code_example', '# No code example provided')}</div>
"""

        # Add special sections for specific steps
        if 'curl_command' in step:
            html += f"""
        <div class="section-title">🌐 Direct API Access</div>
        <div class="curl-command">{step['curl_command']}</div>
"""

        if 'formulas' in step:
            html += f"""
        <div class="section-title">📐 Mathematical Formulas</div>
        <div class="math-formula">
            <strong>Cosine Similarity:</strong> {step['formulas']['cosine_similarity']}<br>
            <strong>Cosine Distance:</strong> {step['formulas']['cosine_distance']}<br>
            <strong>Dot Product:</strong> {step['formulas']['dot_product']}<br>
            <strong>Vector Norm:</strong> {step['formulas']['vector_norm']}
        </div>
"""

        html += "</div>"

    html += """
    <div class="header" style="margin-top: 40px;">
        <h2>🎯 Pipeline Complete!</h2>
        <p>This demo showed the complete RAG pipeline from raw text to ranked search results.</p>
        <p><a href="/" style="color: #4CAF50;">← Back to Main Menu</a></p>
    </div>
</body>
</html>
"""

    return html
