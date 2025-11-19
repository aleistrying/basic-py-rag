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
    "PGVECTOR_DATABASE_URL", "postgresql://pguser:pgpass@pgvector_db:5432/vectordb")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")


def format_json_with_syntax_highlighting(data) -> str:
    """Format JSON with proper syntax highlighting"""
    import json
    import html

    if data is None:
        return "null"

    # Convert to properly formatted JSON string
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    # Escape HTML characters
    json_str = html.escape(json_str)

    # Apply syntax highlighting
    lines = json_str.split('\n')
    highlighted_lines = []

    for line in lines:
        # Color different JSON elements
        # Strings (keys and values)
        line = re.sub(
            r'"([^"]*)":', r'<span style="color: #9cdcfe;">"\1"</span><span style="color: #d4d4d4;">:</span>', line)
        line = re.sub(
            r': "([^"]*)"', r': <span style="color: #ce9178;">"\1"</span>', line)

        # Numbers
        line = re.sub(r': (-?\d+\.?\d*)',
                      r': <span style="color: #b5cea8;">\1</span>', line)

        # Booleans and null
        line = re.sub(r': (true|false|null)',
                      r': <span style="color: #569cd6;">\1</span>', line)

        # Brackets and braces
        line = re.sub(
            r'([{}\[\]])', r'<span style="color: #ffd700;">\1</span>', line)

        highlighted_lines.append(line)

    return '\n'.join(highlighted_lines)


def get_svg_icon(name: str, size: str = "16", color: str = "#8b5cf6") -> str:
    """Generate SVG icons for demo pipeline"""
    icons = {
        "input": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M7 17L17 7"/><path d="M7 7H17V17"/></svg>',
        "output": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M7 7L17 17"/><path d="M17 7V17H7"/></svg>',
        "code": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><polyline points="16,18 22,12 16,6"/><polyline points="8,6 2,12 8,18"/></svg>',
        "idea": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="10"/></svg>',
        "web": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2C14.5 7 14.5 17 12 22C9.5 17 9.5 7 12 2z"/></svg>',
        "formula": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M5 12h14"/><path d="M12 5v14"/><circle cx="12" cy="12" r="10"/></svg>',
        "experiment": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><path d="M9 2v6l-3 3.5a2 2 0 0 0 0 3L9 18v2"/><path d="M15 2v6l3 3.5a2 2 0 0 1 0 3L15 18v2"/><line x1="9" y1="7" x2="15" y2="7"/></svg>'
    }
    return icons.get(name, f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>')


class RAGPipelineDemo:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.example_text = """
        PostgreSQL con pgvector es una extensión que permite almacenar y buscar vectores de alta dimensionalidad
        directamente en PostgreSQL. Esta tecnología es fundamental para aplicaciones de inteligencia artificial
        que requieren búsquedas semánticas eficientes. Los vectores pueden representar embeddings de texto,
        imágenes o cualquier otro tipo de datos que se puedan convertir en representaciones numéricas.
        """

    def _load_model_safely(self):
        """Cargar modelo de forma segura con manejo de errores"""
        if self.model_loaded:
            return True

        try:
            import torch
            # Detect GPU availability and set device accordingly
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🔧 Using device: {device}")

            # Configuración específica para contenedores Docker
            torch.set_default_dtype(torch.float32)

            from sentence_transformers import SentenceTransformer

            # Configuración específica para evitar problemas de meta tensors
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

            # Cargar modelo con configuración optimizada para GPU/CPU
            self.model = SentenceTransformer(
                'intfloat/multilingual-e5-base',
                device=device,
                trust_remote_code=False,
                use_auth_token=False
            )

            # Optimizaciones específicas para el device
            if device == "cuda":
                # GPU optimizations
                self.model = self.model.cuda()
                logger.info("🚀 Model loaded on GPU")
            else:
                # CPU optimizations
                self.model = self.model.cpu()
                # Forzar que todos los parámetros estén en CPU y en formato float32
                for param in self.model.parameters():
                    if param.device.type != 'cpu':
                        param.data = param.data.cpu()
                    if param.dtype != torch.float32:
                        param.data = param.data.float()
                logger.info("💻 Model loaded on CPU")

            # Prueba segura del modelo
            with torch.no_grad():
                test_embedding = self.model.encode(
                    ["test"], show_progress_bar=False, convert_to_tensor=False)

            self.model_loaded = True
            device_info = "GPU (CUDA)" if device == "cuda" else "CPU"
            logger.info(f"✅ Modelo E5 cargado exitosamente en {device_info}")
            return True

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            self.model = None
            self.model_loaded = False
            return False

    def _create_mock_embeddings(self, texts: List[str], dimensions: int = 768) -> List[List[float]]:
        """Crear embeddings simulados para demo cuando el modelo no está disponible"""
        import random
        import hashlib

        embeddings = []
        for text in texts:
            # Crear embedding determinístico basado en el hash del texto
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32 - 1)
            random.seed(seed)

            # Generar vector normalizado
            embedding = [random.gauss(0, 1) for _ in range(dimensions)]
            # Normalizar L2
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x/norm for x in embedding]

            embeddings.append(embedding)

        return embeddings

    def step_1_parse_text(self) -> Dict[str, Any]:
        """Step 1: Parse and prepare text"""
        raw_text = self.example_text

        # Basic text cleaning
        cleaned_text = re.sub(r'\s+', ' ', raw_text.strip())

        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "step": "1. Análisis de Texto",
            "description": "Analizar texto crudo y dividir en fragmentos procesables",
            "input": raw_text,
            "output": {
                "sentences": sentences,
                "total_sentences": len(sentences)
            },
            "code_example": """
# Código de análisis de texto
text = raw_document_text
cleaned_text = re.sub(r'\\s+', ' ', text.strip())
sentences = re.split(r'[.!?]+', cleaned_text)
            """,
            "explanation": "Limpiamos espacios en blanco y dividimos el texto en unidades semánticas (oraciones) para mejor calidad de embeddings."
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
            "step": "2. Limpieza y Fragmentación de Texto",
            "description": "Limpiar texto y crear fragmentos superpuestos para mejor contexto",
            "input": sentences,
            "cleaned_sentences": cleaned_sentences,
            "output": {
                "chunks": chunks,
                "total_chunks": len(chunks)
            },
            "code_example": """
def clean_sentence(text):
    text = re.sub(r'\\s+', ' ', text)
    text = re.sub(r'[^\\w\\s\\.,;:\\-]', '', text)
    return text.strip()

# Crear fragmentos superpuestos
chunks = []
for i, sentence in enumerate(sentences):
    chunk = sentence
    if i > 0:
        chunk = sentences[i-1] + " " + chunk
    chunks.append(chunk)
            """,
            "explanation": "Limpiamos el texto y creamos fragmentos superpuestos para preservar el contexto entre oraciones relacionadas."
        }

    def step_3_create_embeddings(self, chunks: List[str]) -> Dict[str, Any]:
        """Step 3: Convert text to vector embeddings"""

        # Intentar cargar modelo de forma segura
        model_available = self._load_model_safely()

        # Add E5 prefix for better embedding quality
        prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]

        if model_available and self.model:
            try:
                # Generate embeddings con el modelo real
                embeddings = self.model.encode(
                    prefixed_chunks, show_progress_bar=False)
                embeddings_list = [emb.tolist() for emb in embeddings]
                model_status = "✅ Modelo E5 real cargado"

            except Exception as e:
                logger.error(f"Error en encode: {e}")
                # Fallback a embeddings simulados
                embeddings_list = self._create_mock_embeddings(prefixed_chunks)
                model_status = "⚠️ Usando embeddings simulados (error en modelo real)"
        else:
            # Usar embeddings simulados
            embeddings_list = self._create_mock_embeddings(prefixed_chunks)
            model_status = "⚠️ Usando embeddings simulados (modelo no disponible)"

        return {
            "step": "3. Generación de Embeddings",
            "description": "Convertir fragmentos de texto a vectores de alta dimensión usando modelo E5",
            "input": chunks,
            "prefixed_chunks": prefixed_chunks,
            "output": {
                "embeddings_shape": f"{len(embeddings_list)} vectores x {len(embeddings_list[0])} dimensiones",
                # Primeras 10 dimensiones
                "sample_embedding": embeddings_list[0][:10],
                "full_embeddings": embeddings_list,
                "model_status": model_status
            },
            "code_example": """
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-base')
prefixed_chunks = [f"passage: {chunk}" for chunk in chunks]
embeddings = model.encode(prefixed_chunks)

# Cada embedding tiene 768 dimensiones
print(f"Forma: {embeddings.shape}")  # (n_chunks, 768)
            """,
            "explanation": "Usamos el modelo E5 multilingüe para convertir texto en vectores de 768 dimensiones. El prefijo 'passage:' ayuda al modelo a entender que es contenido de documento."
        }

    def step_7_query_processing(self, query: str) -> Dict[str, Any]:
        """Step 7: Process user query into embedding"""

        # Intentar cargar modelo
        model_available = self._load_model_safely()

        # Add E5 prefix for queries
        prefixed_query = f"query: {query}"

        if model_available and self.model:
            try:
                # Generate query embedding con modelo real
                query_embedding = self.model.encode(
                    [prefixed_query], show_progress_bar=False)[0]
                query_embedding_list = query_embedding.tolist()
                model_status = "✅ Modelo E5 real"

            except Exception as e:
                logger.error(f"Error en query encode: {e}")
                # Fallback a embedding simulado
                query_embedding_list = self._create_mock_embeddings([prefixed_query])[
                    0]
                model_status = "⚠️ Embedding simulado"
        else:
            # Usar embedding simulado
            query_embedding_list = self._create_mock_embeddings([prefixed_query])[
                0]
            model_status = "⚠️ Embedding simulado"

        return {
            "step": "7. Procesamiento de Consulta",
            "description": "Convertir consulta del usuario al mismo espacio vectorial de los documentos",
            "input": query,
            "prefixed_query": prefixed_query,
            "output": {
                "query_embedding_shape": f"1 vector x {len(query_embedding_list)} dimensiones",
                "sample_dimensions": query_embedding_list[:10],
                "full_embedding": query_embedding_list,
                "model_status": model_status
            },
            "code_example": """
query = "¿Qué es pgvector?"
prefixed_query = f"query: {query}"
query_embedding = model.encode([prefixed_query])[0]

# El embedding de consulta tiene las mismas dimensiones que los embeddings de documentos
print(f"Forma de consulta: {query_embedding.shape}")  # (768,)
            """,
            "explanation": "Procesamos la consulta usando el mismo modelo y agregamos el prefijo 'query:' para ayudar al modelo a entender que es una consulta de búsqueda."
        }

    def step_8_vector_search_qdrant(self, query_embedding: List[float], limit: int = 3) -> Dict[str, Any]:
        """Step 8: Perform vector search in Qdrant with curl examples"""

        # Prepare Qdrant search payload
        search_payload = {
            "vector": query_embedding,
            "limit": limit,
            "with_payload": True,
            "with_vector": True
        }

        # Generate curl command for direct API access (properly escaped)
        vector_str = json.dumps(query_embedding)
        curl_command = f"curl -X POST '{QDRANT_URL}/collections/course_docs_clean/points/search' \\\n  -H 'Content-Type: application/json' \\\n  -d '{{\"vector\": {vector_str}, \"limit\": {limit}, \"with_payload\": true, \"with_vector\": true}}'"

        try:
            # Use the Qdrant client for proper search
            from qdrant_client import QdrantClient
            client = QdrantClient(host="qdrant", port=6333)

            # First try to search in existing collections
            collections = client.get_collections()
            available_collections = [col.name for col in collections.collections] if hasattr(
                collections, 'collections') else []

            search_collection = "course_docs_clean_cosine_hnsw"  # Use actual collection
            if search_collection not in available_collections and available_collections:
                # Use first available
                search_collection = available_collections[0]

            if available_collections:
                search_result = client.query_points(
                    collection_name=search_collection,
                    query=query_embedding,
                    limit=limit,
                    with_payload=True
                )

                search_results = {
                    "results": [
                        {
                            "id": point.id,
                            "score": point.score,
                            "payload": point.payload,
                            "content_preview": str(point.payload.get('content', ''))[:100] + "..."
                        }
                        for point in search_result.points
                    ],
                    "collection_used": search_collection,
                    "total_found": len(search_result.points)
                }
            else:
                search_results = {
                    "error": "No collections found. Use /pipeline/upload to create demo data first."}

        except Exception as e:
            search_results = {"error": f"Search failed: {str(e)}"}

        return {
            "step": "8. Búsqueda Vectorial (Qdrant)",
            "description": "Buscar vectores similares usando similitud de coseno",
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
            "explanation": "Qdrant realiza búsqueda de similitud de coseno para encontrar los fragmentos de documentos más similares semánticamente a nuestra consulta."
        }

    def step_9_vector_search_pgvector(self, query_embedding: List[float], limit: int = 3) -> Dict[str, Any]:
        """Step 9: Perform vector search in PostgreSQL with pgvector"""

        # Format embedding for PostgreSQL
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # SQL query with vector similarity - use demo table name
        sql_query = f"""
SELECT
    id,
    content,
    metadata,
    embedding <=> '{embedding_str}'::vector as distance
FROM demo_vectors
ORDER BY embedding <=> '{embedding_str}'::vector
LIMIT {limit};
        """

        # Equivalent curl command for PostgreSQL (simplified for display)
        curl_command = f"# PostgreSQL vector search (via psql or REST API)\npsql \"{PGVECTOR_DATABASE_URL}\" -c \"\\\n  SELECT id, content, metadata, \\\n    embedding <=> '[{embedding_str[:50]}...]'::vector as distance \\\n  FROM demo_vectors \\\n  ORDER BY embedding <=> '[{embedding_str[:50]}...]'::vector \\\n  LIMIT {limit};\""

        try:
            # Execute search with table existence check
            conn = psycopg2.connect(PGVECTOR_DATABASE_URL)
            cursor = conn.cursor()

            # Check if table exists first
            cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'demo_vectors');")
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                # Create demo table if it doesn't exist
                cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS demo_vectors (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(768)
                );
                """)

                # Insert demo data
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    embedding_json = json.dumps(emb)
                    cursor.execute(
                        "INSERT INTO demo_vectors (content, metadata, embedding) VALUES (%s, %s, %s::vector)",
                        (chunk, '{"source": "demo", "chunk_id": ' +
                         str(i) + '}', embedding_json)
                    )
                conn.commit()

            # Now execute the search
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
            "step": "9. Búsqueda Vectorial (PostgreSQL pgvector)",
            "description": "Búsqueda usando extensión pgvector de PostgreSQL con distancia de coseno",
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

# Formatear embedding para PostgreSQL
embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

# Ejecutar búsqueda de similitud vectorial
cursor.execute(\"\"\"
    SELECT id, content, metadata,
           embedding <=> %s::vector as distance
    FROM docs_clean
    ORDER BY embedding <=> %s::vector
    LIMIT %s
\"\"\", (embedding_str, embedding_str, limit))

results = cursor.fetchall()
            """,
            "explanation": "PostgreSQL pgvector usa el operador <=> para distancia de coseno. Distancias menores indican mayor similitud."
        }

    def step_10_similarity_math(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> Dict[str, Any]:
        """Step 10: Explain the mathematical calculations behind similarity"""
        import numpy as np

        # Convert to numpy for calculations
        query_vec = np.array(query_embedding)

        similarity_calculations = []

        # Ensure we have doc_embeddings to work with
        if not doc_embeddings:
            # Create sample embeddings for demonstration
            doc_embeddings = [np.random.randn(
                len(query_embedding)).tolist() for _ in range(3)]

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
            "step": "10. Matemáticas de Similitud",
            "description": "Cálculos matemáticos detrás de la búsqueda de similitud vectorial",
            "input": {
                "query_vector_norm": float(np.linalg.norm(query_vec)),
                "calculation_method": "Similitud de Coseno"
            },
            "output": {
                "calculations": similarity_calculations,
                "total_documents_compared": len(similarity_calculations),
                "average_similarity": float(np.mean([calc["cosine_similarity"] for calc in similarity_calculations]))
            },
            "formulas": {
                "cosine_similarity": "cos(θ) = (A · B) / (||A|| × ||B||)",
                "cosine_distance": "distancia = 1 - similitud_coseno",
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

# Calcular similitud
similarity = cosine_similarity(query_embedding, doc_embedding)
distance = cosine_distance(query_embedding, doc_embedding)
            """,
            "explanation": "La similitud de coseno mide el ángulo entre vectores. Valores más cercanos a 1 indican mayor similitud. Ambas bases de datos convierten esto a distancia (1 - similitud) donde valores menores indican mejores coincidencias."
        }

    def step_11_result_ranking(self, search_results) -> Dict[str, Any]:
        """Step 11: Rank and present final results"""

        # Handle different input types properly
        ranked_results = []

        try:
            # Handle string input (error message)
            if isinstance(search_results, str):
                ranked_results = {
                    "error": f"Search returned string: {search_results}"}
            # Check if it's an error response
            elif isinstance(search_results, dict) and "error" in search_results:
                ranked_results = {
                    "error": "Could not rank results due to search error"}
            elif isinstance(search_results, list):
                # Direct list of results - handle each item safely
                for i, item in enumerate(search_results):
                    if isinstance(item, dict):
                        # Calculate score from distance if needed (pgvector returns distance, not score)
                        distance = item.get("distance", 0)
                        score = item.get(
                            "score", 1 - distance if distance > 0 else 0)

                        rank_info = {
                            "rank": i + 1,
                            "result": item,
                            "relevance": "High" if i < 1 else "Medium" if i < 3 else "Low",
                            "score": round(score, 4),
                            "distance": round(distance, 4)
                        }
                        ranked_results.append(rank_info)
            elif isinstance(search_results, dict):
                # Dict format - extract results safely
                if "results" in search_results:
                    results = search_results["results"]
                    if isinstance(results, list):
                        for i, result in enumerate(results):
                            if isinstance(result, dict):
                                rank_info = {
                                    "rank": i + 1,
                                    "result": result,
                                    "relevance": "High" if i < 1 else "Medium" if i < 3 else "Low",
                                    "score": result.get("score", 0)
                                }
                                ranked_results.append(rank_info)
                else:
                    # Single result format
                    rank_info = {
                        "rank": 1,
                        "result": search_results,
                        "relevance": "High",
                        "score": search_results.get("score", 0) if isinstance(search_results, dict) else 0
                    }
                    ranked_results.append(rank_info)
            else:
                ranked_results = {
                    "error": f"Unexpected search results format: {type(search_results)}"}

            # If no results, create demo results
            if not ranked_results or isinstance(ranked_results, dict):
                ranked_results = [
                    {
                        "rank": 1,
                        "result": {
                            "content": "PostgreSQL con pgvector es una extensión que permite almacenar vectores.",
                            "score": 0.85,
                            "path": "./data/raw/ejemplo.pdf",
                            "page": 10
                        },
                        "relevance": "High",
                        "score": 0.85
                    }
                ]

        except Exception as e:
            ranked_results = {
                "error": f"Ranking error: {str(e)} - Input type: {type(search_results)}"}

        # Create detailed input description
        input_description = {
            "input_type": str(type(search_results).__name__),
            "items_count": len(search_results) if hasattr(search_results, '__len__') else "unknown",
            "source": "Vector search results from Qdrant or pgvector",
            "sample_result": search_results[0] if isinstance(search_results, list) and search_results else (search_results.get("results", [None])[0] if isinstance(search_results, dict) and "results" in search_results else None)
        }

        return {
            "step": "11. Clasificación y Presentación de Resultados",
            "description": "Clasificar resultados de búsqueda por relevancia y preparar respuesta final",
            "input": input_description,
            "output": ranked_results,
            "ranking_criteria": {
                "primary": "Puntuación de similitud de coseno (Qdrant) o distancia (pgvector)",
                "secondary": "Calidad del contenido y completitud",
                "presentation": "Top 3 resultados con metadatos y atribución de fuente"
            },
            "code_example": """
# Clasificar resultados por puntuación de similitud
def rank_results(results, score_key="score", reverse=True):
    return sorted(results, key=lambda x: x.get(score_key, 0) if isinstance(x, dict) else 0, reverse=reverse)

# Agregar clasificación de relevancia
for i, result in enumerate(ranked_results):
    if i == 0:
        result["relevance"] = "Alta"
    elif i < 3:
        result["relevance"] = "Media"
    else:
        result["relevance"] = "Baja"
            """,
            "explanation": "Los resultados se clasifican por puntuación de similitud y se categorizan por relevancia. Los mejores resultados se usan para generar la respuesta final de IA con atribución de fuentes apropiada."
        }

    def step_12_ai_generation(self, ranked_results: List[Dict], query: str, model: str = "phi3:mini", distance_metric: str = "cosine", index_algorithm: str = "hnsw") -> Dict[str, Any]:
        """Step 12: Generate AI response from ranked results"""

        # Initialize variables outside try block
        context_parts = []
        sources_used = []
        prompt = ""
        final_response = ""
        ai_status = "unknown"

        try:
            # Build context from top results

            # Use top 3 results
            for i, ranked_item in enumerate(ranked_results[:3]):
                result = ranked_item.get("result", {})
                content = result.get("content", "")
                if content:
                    # Extract metadata for source attribution (handle both Qdrant and pgvector formats)
                    metadata = result.get("metadata", {})
                    payload = result.get("payload", {})

                    # Try multiple metadata sources
                    doc_name = (
                        result.get("path", "") or
                        metadata.get("source_name", "") or
                        payload.get("source_name", "") or
                        payload.get("source_path", "").split(
                            "/")[-1] if payload.get("source_path") else ""
                    )
                    doc_name = doc_name.replace(
                        "./data/raw/", "").replace(".pdf", "")

                    # Get page number from multiple possible sources
                    page = result.get('page') or metadata.get(
                        'page') or payload.get('page')
                    page_info = f", p.{page}" if page else ""

                    # Calculate similarity score (handle both score and distance)
                    distance = result.get(
                        'distance', ranked_item.get('distance', 0))
                    similarity = result.get('score', ranked_item.get(
                        'score', 1 - distance if distance > 0 else 0))

                    context_parts.append(
                        f"- ({doc_name if doc_name else 'Demo Pipeline'}{page_info}) {content[:300]}...")
                    sources_used.append({
                        "rank": i + 1,
                        "document": doc_name if doc_name else "Demo Pipeline",
                        "page": page,
                        "similarity": round(similarity, 4),
                        "distance": round(distance, 4),
                        "preview": content[:150] + "..." if len(content) > 150 else content
                    })

            context = "\n".join(context_parts)

            # Create prompt template for LLM
            prompt_template = """Eres un asistente académico. Responde de forma directa y concisa usando SOLO la información de los fragmentos.

Pregunta: {query}

Fragmentos relevantes:
{context}

Instrucciones:
- Da una respuesta clara y directa
- Usa fechas, números y nombres exactos de los fragmentos
- Si la información no está en los fragmentos, responde: "No está disponible."
- Solo menciona el documento si es relevante para entender la respuesta
- NO incluyas referencias bibliográficas completas ni autores irrelevantes
- Responde en español, de forma natural y concisa

Respuesta:
"""

            prompt = prompt_template.format(query=query, context=context)

            # Simulate AI generation (in demo mode)
            try:
                import ollama
                import os
                ollama_host = os.getenv(
                    "OLLAMA_HOST", "http://localhost:11434")
                client = ollama.Client(host=ollama_host)

                response = client.generate(
                    model=model,
                    prompt=prompt,
                    options={
                        "repeat_penalty": 1.0,
                        "temperature": 0.5,
                        "num_predict": 256,
                        "top_p": 0.9
                    }
                )

                generated_text = response.get('response', '').strip()
                ai_status = "Generated successfully"

            except Exception as e:
                # Fallback response for demo
                generated_text = f"Respuesta simulada para '{query}': Según los documentos encontrados, esta información está relacionada con sistemas de bases de datos avanzadas y tecnologías vectoriales. Se encontraron {len(sources_used)} fuentes relevantes con alta similitud semántica."
                ai_status = f"Demo mode (Ollama not available: {str(e)})"

            # Add source attribution
            source_references = []
            for i, source in enumerate(sources_used, 1):
                source_references.append(
                    f"{i}. {source['document']} (p. {source['page'] if source['page'] else 'N/A'})")

            if source_references:
                sources_text = "\n".join(source_references)
                final_response = f"{generated_text}\n\n**Fuentes consultadas:**\n{sources_text}"
            else:
                final_response = generated_text

        except Exception as e:
            final_response = f"Error generating AI response: {str(e)}"
            ai_status = "Error"
            sources_used = []
            prompt = "Error creating prompt"

        return {
            "step": "12. Generación de Respuesta con IA",
            "description": f"Generar respuesta natural usando modelo de lenguaje (LLM) basada en los documentos encontrados con similitud {distance_metric.upper()} + índice {index_algorithm.upper()}",
            "input": {
                "query": query,
                "model": model,
                "distance_metric": distance_metric.upper(),
                "index_algorithm": index_algorithm.upper(),
                "ranked_results_count": len(ranked_results),
                "context_length": len(context_parts)
            },
            "output": {
                "ai_response": final_response,
                "model_used": model,
                "algorithm_used": f"{distance_metric.upper()}+{index_algorithm.upper()}",
                "status": ai_status,
                "sources_count": len(sources_used),
                "sources_used": sources_used
            },
            "prompt_example": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "code_example": """
import ollama

# Construir contexto de los mejores resultados
context = []
for result in top_ranked_results:
    doc_name = result['path'].replace('./data/raw/', '')
    page_info = f", p.{result.get('page', 'N/A')}"
    context.append(f"- ({doc_name}{page_info}) {result['content'][:300]}...")

context_text = "\\n".join(context)

# Crear prompt estructurado para el LLM
prompt = f\"\"\"Eres un asistente académico. Responde usando SOLO la información proporcionada.

Pregunta: {query}
Fuentes: {context_text}

Responde de forma clara y concisa en español.\"\"\"

# Generar respuesta con Ollama
client = ollama.Client(host="http://ollama:11434")
response = client.generate(
    model="phi3:mini",
    prompt=prompt,
    options={"temperature": 0.5, "num_predict": 256}
)

answer = response['response']
            """,
            "explanation": "Se usa un modelo de lenguaje (LLM) para generar una respuesta natural basada en los documentos más relevantes. El prompt incluye contexto específico y instrucciones para responder solo con la información proporcionada."
        }

    def step_4_storage_simulation(self, chunks: List[str], embeddings: List[List[float]], storage_type: str = "both", distance_metric: str = "cosine", index_algorithm: str = "hnsw") -> Dict[str, Any]:
        """Step 4: Simulate database storage with real input/output for both Qdrant and PostgreSQL"""
        import uuid

        # Distance metric configurations
        distance_configs = {
            "cosine": {
                "qdrant_distance": "Cosine",
                "pgvector_operator": "<=>",
                "description": "Mide el ángulo entre vectores (0-2, menor es mejor)\nIdeal para: Texto, embeddings normalizados\nIgnora magnitud, solo dirección",
                "math_formula": "distance = 1 - (A·B)/(|A|×|B|)"
            },
            "euclidean": {
                "qdrant_distance": "Euclid",
                "pgvector_operator": "<->",
                "description": "Distancia L2 en espacio euclidiano\nIdeal para: Coordenadas, características continuas\nSensible a magnitud y dirección",
                "math_formula": "distance = √(Σ(ai-bi)²)"
            },
            "dot_product": {
                "qdrant_distance": "Dot",
                "pgvector_operator": "<#>",
                "description": "Producto interno negativo (-A·B)\nIdeal para: Vectores ya normalizados\nMás rápido que coseno si vectores normalizados",
                "math_formula": "distance = -Σ(ai×bi)"
            },
            "manhattan": {
                "qdrant_distance": "Manhattan",
                "pgvector_operator": "<+>",
                "description": "Distancia L1 (suma de diferencias absolutas)\nIdeal para: Datos dispersos, características categóricas\nMenos sensible a outliers",
                "math_formula": "distance = Σ|ai-bi|"
            }
        }

        # Index algorithm configurations
        index_configs = {
            "hnsw": {
                "qdrant_available": True,
                "pgvector_available": False,
                "description": "Hierarchical Navigable Small World - Algoritmo ANN muy eficiente",
                "qdrant_config": {"m": 16, "ef_construct": 100},
                "pgvector_config": None,  # HNSW not available in pgvector
                "pgvector_alternative": "ivfflat"
            },
            "ivfflat": {
                "qdrant_available": False,
                "pgvector_available": True,
                "description": "Inverted File Flat - Índice ANN básico de pgvector",
                "pgvector_config": {"lists": 100},
                "qdrant_alternative": "hnsw"
            },
            "scalar_quantization": {
                "qdrant_available": True,
                "pgvector_available": False,
                "description": "Compresión escalar (int8) para reducir memoria",
                "qdrant_config": {"type": "int8", "always_ram": True},
                "pgvector_config": None,  # Scalar quantization not available in pgvector
                "pgvector_alternative": "ivfflat"
            },
            "exact": {
                "qdrant_available": True,
                "pgvector_available": True,
                "description": "Búsqueda exacta sin índice (para datasets pequeños)",
                "qdrant_config": None,
                "pgvector_config": None
            }
        }

        current_distance = distance_configs.get(
            distance_metric, distance_configs["cosine"])
        current_index = index_configs.get(
            index_algorithm, index_configs["hnsw"])

        # Generate simulated storage data for demonstration
        doc_id = str(uuid.uuid4())[:8]

        storage_results = {
            "step": "4. Simulación de Almacenamiento en Bases de Datos",
            "description": f"Demostrar almacenamiento en {storage_type} usando métrica {distance_metric.upper()} con índice {index_algorithm.upper()}",
            "input": {
                "chunks_count": len(chunks),
                "sample_chunk": chunks[0][:100] + "..." if chunks else "No chunks available",
                "embedding_dimensions": len(embeddings[0]) if embeddings else 768,
                "storage_type": storage_type,
                "distance_metric": distance_metric.upper(),
                "index_algorithm": index_algorithm.upper(),
                "distance_description": current_distance["description"],
                "index_description": current_index["description"]
            },
            "output": {
                "distance_metric_used": distance_metric.upper(),
                "index_algorithm_used": index_algorithm.upper(),
                "distance_function": current_distance["math_formula"]
            }
        }

        if storage_type in ["qdrant", "both"]:
            # Simulate Qdrant storage with algorithm-specific details
            points_created = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"{doc_id}_qdrant_{i}"
                points_created.append({
                    "id": point_id,
                    "vector_preview": embedding[:5] + [f"...más {len(embedding)-5} dims"],
                    "payload": {
                        "content": chunk[:150] + "..." if len(chunk) > 150 else chunk,
                        "chunk_id": i,
                        "doc_id": doc_id,
                        "distance_metric": distance_metric,
                        "index_algorithm": index_algorithm
                    }
                })

            # Determine actual Qdrant index based on availability
            qdrant_index = index_algorithm if current_index[
                "qdrant_available"] else current_index["qdrant_alternative"]

            qdrant_demo = {
                "phase": f"FASE 2A: ALMACENAMIENTO EN QDRANT ({distance_metric.upper()} + {qdrant_index.upper()})",
                "title": f"Configuración e ingesta de vectores en Qdrant con {current_distance['qdrant_distance']} distance + {qdrant_index} index",
                "input": {
                    "text": chunks[0][:100] + "..." if chunks else "Texto de ejemplo...",
                    "doc_path": "docs/pipeline_demo.pdf",
                    "page": 1,
                    "method": f"{qdrant_index.upper()} Index + {distance_metric.upper()}"
                },
                "process": {
                    "step1": "Embedding con E5-multilingual",
                    "step2": f"Configurar distancia {current_distance['qdrant_distance']}",
                    "step3": f"Inserción con {qdrant_index.upper()}",
                    "step4": "Indexación automática"
                },
                "output": {
                    "collection_name": f"demo_pipeline_{distance_metric}_{qdrant_index}",
                    "points_created": points_created,
                    "point_id": doc_id,
                    "vector_size": len(embeddings[0]) if embeddings else 768,
                    "distance_metric": current_distance["qdrant_distance"],
                    "index_algorithm": qdrant_index.upper(),
                    "index_params": current_index["qdrant_config"] if current_index["qdrant_config"] else {"type": "exact"},
                    "indexed_count": len(chunks),
                    "status": "success"
                },
                "embedding_preview": embeddings[0][:10] if embeddings else [0.1, -0.2, 0.3, 0.0, -0.1, 0.4, 0.2, -0.3, 0.1, 0.0],
                "storage_code": '''# Qdrant: Insertar punto vectorial
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient(host="localhost", port=6333)

# Insertar vector
point_data = PointStruct(
    id="{doc_id}",
    vector={embedding_preview},
    payload={{
        "content": "{text}",
        "path": "docs/pipeline_demo.pdf",
        "page": 1,
        "chunk_id": "{doc_id}"
    }}
)

result = client.upsert(
    collection_name="demo_pipeline_collection",
    points=[point_data]
)'''.format(
                    doc_id=doc_id,
                    embedding_preview=str(embeddings[0][:5] if embeddings else [
                                          0.1, -0.2, 0.3, 0.0, -0.1]) + "...",
                    text=(
                        chunks[0][:50] + "..." if chunks else "Contenido del documento...")
                ),
                "query_example": {
                    "query": "buscar información",
                    "results": [
                        {
                            "id": doc_id,
                            "score": 0.924,
                            "content": chunks[0] if chunks else "Contenido de ejemplo...",
                            "metadata": {
                                "path": "docs/pipeline_demo.pdf",
                                "page": 1
                            }
                        }
                    ]
                }
            }
            storage_results["qdrant_storage"] = qdrant_demo

        if storage_type in ["postgresql", "both"]:
            # Simulate PostgreSQL storage with algorithm-specific details
            rows_created = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                row_id = len(chunks) + 1000 + i
                rows_created.append({
                    "id": row_id,
                    "content_preview": chunk[:150] + "..." if len(chunk) > 150 else chunk,
                    "vector_preview": embedding[:5] + [f"...más {len(embedding)-5} dims"],
                    "distance_operator": current_distance["pgvector_operator"],
                    "distance_metric": distance_metric,
                    "index_algorithm": index_algorithm
                })

            # Determine actual PostgreSQL index based on availability
            pg_index = index_algorithm if current_index[
                "pgvector_available"] else current_index["pgvector_alternative"]

            postgres_demo = {
                "phase": f"FASE 2B: ALMACENAMIENTO EN POSTGRESQL ({distance_metric.upper()} + {pg_index.upper()})",
                "title": f"Configuración e ingesta de vectores en PostgreSQL con pgvector + {distance_metric.upper()}",
                "input": {
                    "text": chunks[0][:100] + "..." if chunks else "Texto de ejemplo...",
                    "doc_path": "docs/pipeline_demo.pdf",
                    "page": 1,
                    "method": f"{pg_index.upper()} Index + {distance_metric.upper()}"
                },
                "process": {
                    "step1": "Embedding con E5-multilingual",
                    "step2": f"Configurar operador {current_distance['pgvector_operator']}",
                    "step3": "Inserción SQL con vector type",
                    "step4": f"Indexación {pg_index.upper()}"
                },
                "output": {
                    "table_name": f"demo_pipeline_{distance_metric}_{pg_index}",
                    "rows_created": rows_created,
                    "row_id": len(chunks) + 1000,
                    "vector_size": len(embeddings[0]) if embeddings else 768,
                    "distance_operator": f"{current_distance['pgvector_operator']} ({distance_metric} distance)",
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": pg_index.upper(),
                    "index_params": current_index["pgvector_config"] if current_index["pgvector_config"] else {"type": "exact"},
                    "total_in_table": len(chunks),
                    "status": "success"
                },
                "embedding_preview": embeddings[0][:10] if embeddings else [0.1, -0.2, 0.3, 0.0, -0.1, 0.4, 0.2, -0.3, 0.1, 0.0],
                "storage_code": f'''-- PostgreSQL: Insertar con pgvector ({distance_metric})
INSERT INTO demo_pipeline_{distance_metric}_{pg_index} (
    content,
    embedding,
    path,
    page,
    metadata,
    distance_metric,
    index_algorithm
) VALUES (
    '{chunks[0][:50] + "..." if chunks else "Contenido del documento..."}',
    '{("[" + ",".join([f"{x:.4f}" for x in (embeddings[0][:5] if embeddings else [0.1, -0.2, 0.3, 0.0, -0.1])]) + ",...]")}'::vector,
    'docs/pipeline_demo.pdf',
    1,
    '{{"chunk_id": "{doc_id}", "source": "pipeline_demo", "distance_metric": "{distance_metric}", "index_algorithm": "{pg_index}"}}'::jsonb,
    '{distance_metric}',
    '{pg_index}'
);

-- Crear índice {pg_index.upper()} para {distance_metric}
CREATE INDEX IF NOT EXISTS demo_vectors_{distance_metric}_{pg_index}_idx
ON demo_pipeline_{distance_metric}_{pg_index}
USING {pg_index} (embedding {current_distance["pgvector_operator"]}_ops)
WITH ({("lists = " + str(current_index["pgvector_config"]["lists"]))
      if current_index["pgvector_config"] else "-- exact search"});

-- Verificar inserción
SELECT id, content, path, page, distance_metric, index_algorithm
FROM demo_pipeline_{distance_metric}_{pg_index}
WHERE id = currval('demo_pipeline_{distance_metric}_{pg_index}_id_seq');''',
                "query_example": {
                    "query": f"SELECT content, (embedding {current_distance['pgvector_operator']} '[0.1,0.2,0.3,...]'::vector) AS distance FROM demo_pipeline_{distance_metric}_{pg_index} WHERE distance_metric='{distance_metric}' ORDER BY distance LIMIT 3",
                    "explanation": f"Búsqueda usando {distance_metric.upper()} distance con operador {current_distance['pgvector_operator']}:\n{current_distance['description']}\nFórmula: {current_distance['math_formula']}",
                    "results": [
                        {
                            "id": len(chunks) + 1000,
                            "distance": 0.076,
                            "content": chunks[0] if chunks else "Contenido de ejemplo...",
                            "metadata": {
                                "path": "docs/pipeline_demo.pdf",
                                "page": 1
                            }
                        }
                    ]
                }
            }
            storage_results["postgresql_storage"] = postgres_demo

        storage_results["code_example"] = '''
# Comparación de almacenamiento vectorial

## QDRANT (Vector Database Nativa)
- Optimizada específicamente para vectores
- HNSW index de alta performance
- API REST simple y potente
- Escalabilidad horizontal
- Filtrado avanzado de metadatos

## POSTGRESQL + PGVECTOR
- Integra vectores en BD relacional
- IVFFlat index para ANN search
- ACID transactions + vectores
- SQL familiar para desarrolladores
- Joins entre datos relacionales y vectores
'''

        storage_results["explanation"] = f"""
Simulación de almacenamiento en {storage_type.upper()}:

🗄️ PREPARACIÓN DE DATOS:
- {len(chunks)} fragmentos procesados
- {len(embeddings[0]) if embeddings else 768} dimensiones por vector
- Metadatos incluyen ruta, página, ID de chunk

{'🟡 QDRANT STORAGE:' if storage_type in ["qdrant", "both"] else ''}
{'- Colección: demo_pipeline_collection' if storage_type in [
    "qdrant", "both"] else ''}
{'- Índice HNSW para búsqueda ANN rápida' if storage_type in [
    "qdrant", "both"] else ''}
{'- Payload con metadatos estructurados' if storage_type in [
    "qdrant", "both"] else ''}

{'🟦 POSTGRESQL STORAGE:' if storage_type in ["postgresql", "both"] else ''}
{'- Tabla: demo_pipeline_vectors' if storage_type in [
    "postgresql", "both"] else ''}
{'- Índice IVFFlat para búsquedas eficientes' if storage_type in [
    "postgresql", "both"] else ''}
{'- JSONB metadata + vector type nativo' if storage_type in [
    "postgresql", "both"] else ''}

⚡ OPTIMIZACIÓN:
- Vectores normalizados para similitud coseno
- Índices configurados para balance velocidad/precisión
- Metadatos permiten filtrado avanzado
- Batch insertion para mejor performance
"""

        return storage_results

    def run_complete_demo_with_storage(self, query: str = "¿Qué es pgvector?", model: str = "phi3:mini", storage_type: str = "both", distance_metric: str = "cosine", index_algorithm: str = "hnsw") -> List[Dict[str, Any]]:
        """Run complete demo pipeline with all steps"""
        demo_steps = []

        # Paso 1: Análisis de Texto
        step1 = self.step_1_parse_text()
        demo_steps.append(step1)

        # Paso 2: Limpieza y Fragmentación
        step2 = self.step_2_clean_text(step1["output"]["sentences"])
        demo_steps.append(step2)

        # Paso 3: Generación de Embeddings
        step3 = self.step_3_create_embeddings(step2["output"]["chunks"])
        demo_steps.append(step3)

        # Paso 4: Simulación de Almacenamiento
        step4 = self.step_4_storage_simulation(
            step2["output"]["chunks"],
            step3["output"]["full_embeddings"],
            storage_type, distance_metric, index_algorithm
        )
        demo_steps.append(step4)

        # Paso 5: Subir a Qdrant
        if storage_type in ["qdrant", "both"]:
            step5 = self.step_5_upload_to_qdrant(
                step2["output"]["chunks"],
                step3["output"]["full_embeddings"],
                distance_metric, index_algorithm
            )
            demo_steps.append(step5)

        # Paso 6: Subir a PostgreSQL
        if storage_type in ["postgresql", "both"]:
            step6 = self.step_6_upload_to_pgvector(
                step2["output"]["chunks"],
                step3["output"]["full_embeddings"],
                distance_metric, index_algorithm
            )
            demo_steps.append(step6)

        # Paso 7: Procesamiento de Consulta
        step7 = self.step_7_query_processing(query)
        demo_steps.append(step7)

        # Paso 8: Búsqueda Vectorial (Qdrant)
        if storage_type in ["qdrant", "both"]:
            step8 = self.step_8_vector_search_qdrant(
                step7["output"]["full_embedding"]
            )
            demo_steps.append(step8)

        # Paso 9: Búsqueda Vectorial (PostgreSQL)
        if storage_type in ["postgresql", "both"]:
            step9 = self.step_9_vector_search_pgvector(
                step7["output"]["full_embedding"]
            )
            demo_steps.append(step9)

        # Paso 10: Matemáticas de Similitud
        step10 = self.step_10_similarity_math(
            step7["output"]["full_embedding"],
            step3["output"]["full_embeddings"]
        )
        demo_steps.append(step10)

        # Paso 11: Clasificación de Resultados
        # Use the last search results
        last_search = None
        for step in reversed(demo_steps):
            if "Búsqueda" in step.get("step", ""):
                last_search = step.get("output", {})
                break

        if not last_search:
            # Create mock search results
            last_search = {
                "results": [
                    {
                        "content": "PostgreSQL con pgvector es una extensión que permite almacenar vectores.",
                        "score": 0.85,
                        "path": "./data/raw/ejemplo.pdf",
                        "page": 10
                    }
                ]
            }

        step11 = self.step_11_result_ranking(last_search)
        demo_steps.append(step11)

        # Paso 12: Generación de IA
        ranked_results = step11["output"]
        if isinstance(ranked_results, list):
            step12 = self.step_12_ai_generation(
                ranked_results, query, model, distance_metric, index_algorithm)
            demo_steps.append(step12)

        return demo_steps

    def step_5_upload_to_qdrant(self, chunks: List[str], embeddings: List[List[float]], distance_metric: str = "cosine", index_algorithm: str = "hnsw") -> Dict[str, Any]:
        """Step 5: Upload documents to Qdrant with HNSW indexing"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            import uuid

            # Distance metric to Qdrant Distance mapping
            distance_mapping = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot_product": Distance.DOT,
                "manhattan": Distance.MANHATTAN
            }

            qdrant_distance = distance_mapping.get(
                distance_metric, Distance.COSINE)

            # Connect to Qdrant - use Docker service name
            client = QdrantClient(host="qdrant", port=6333)
            collection_name = f"demo_collection_{distance_metric}_{index_algorithm}"

            # Check if collection exists, create if not
            try:
                client.get_collection(collection_name)
                collection_exists = True
            except:
                collection_exists = False

            if not collection_exists:
                # Create collection with HNSW index and specified distance metric
                hnsw_config = {
                    "m": 16,              # Number of bi-directional links
                    "ef_construct": 100   # Size of dynamic candidate list
                }

                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=768,  # Dimension of vectors
                        distance=qdrant_distance,
                        hnsw_config=hnsw_config
                    )
                )

            # Prepare points for upload
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk,
                        "metadata": {
                            "chunk_index": idx,
                            "source": "demo_pipeline",
                            "timestamp": "2025-11-19",
                            "distance_metric": distance_metric,
                            "index_algorithm": index_algorithm
                        }
                    }
                )
                points.append(point)

            # Upload vectors
            client.upsert(collection_name=collection_name, points=points)

            # Get collection info
            collection_info = client.get_collection(collection_name)
            vectors_count = collection_info.points_count

            return {
                "step": f"5. Subir a Qdrant con {index_algorithm.upper()} ({distance_metric.upper()})",
                "description": f"Almacenar vectores en Qdrant usando índice {index_algorithm.upper()} para búsqueda rápida con distancia {distance_metric.upper()}",
                "database": "Qdrant",
                "algorithm": f"{index_algorithm.upper()} (Hierarchical Navigable Small World)",
                "input": {
                    "chunks_count": len(chunks),
                    "sample_chunk": chunks[0][:100] + "..." if chunks else "Ejemplo de fragmento de texto procesado",
                    "embedding_dimensions": len(embeddings[0]) if embeddings else 768,
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": index_algorithm.upper()
                },
                "output": {
                    "collection_name": collection_name,
                    "vectors_uploaded": len(points),
                    "total_in_collection": vectors_count,
                    "vector_size": 768,
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": index_algorithm.upper(),
                    "hnsw_params": {
                        "m": 16,
                        "ef_construct": 100
                    },
                    "status": "success"
                },
                "code_example": """
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Conectar a Qdrant
client = QdrantClient(host="qdrant", port=6333)

# Crear colección con HNSW
client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
        hnsw_config={
            "m": 16,              # Conexiones por capa
            "ef_construct": 100   # Calidad de construcción
        }
    )
)

# Subir vectores
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"content": text, "metadata": {...}}
    )
    for text, embedding in zip(texts, embeddings)
]
client.upsert(collection_name="demo_collection", points=points)
""",
                "curl_example": """
# Crear colección via API REST
curl -X PUT 'http://qdrant:6333/collections/demo_collection' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine",
      "hnsw_config": {"m": 16, "ef_construct": 100}
    }
  }'

# Subir punto
curl -X PUT 'http://qdrant:6333/collections/demo_collection/points' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "points": [
      {
        "id": 1,
        "vector": [0.1, 0.2, ...],
        "payload": {"content": "text", "metadata": {}}
      }
    ]
  }'
""",
                "explanation": f"""
HNSW (Hierarchical Navigable Small World):
- Algoritmo de grafo para búsqueda aproximada de vecinos más cercanos (ANN)
- Construye múltiples capas de grafos conectados
- Búsqueda O(log n) en vez de O(n) de fuerza bruta
- Trade-off: velocidad vs precisión ajustable con parámetros

Métricas de Distancia:
- Cosine: Mide ángulo entre vectores (mejor para texto)
- Euclidean: Distancia L2 (mejor para imágenes)  
- Dot Product: Producto punto (similar a cosine pero sin normalizar)

ANN (Approximate Nearest Neighbor):
- No garantiza resultado exacto pero es 1000x más rápido aproximadamente
- Para millones de vectores, ANN es prácticamente obligatorio
- Precisión típica: 95-99% con velocidad sub-milisegundo
"""
            }
        except Exception as e:
            return {
                "step": f"5. Subir a Qdrant con {index_algorithm.upper()} ({distance_metric.upper()})",
                "description": f"Almacenar vectores en Qdrant usando índice {index_algorithm.upper()} para búsqueda rápida con distancia {distance_metric.upper()}",
                "database": "Qdrant",
                "algorithm": f"{index_algorithm.upper()} (Hierarchical Navigable Small World)",
                "connection_status": "Demo Mode (Qdrant no disponible)",
                "input": {
                    "chunks_count": len(chunks),
                    "sample_chunk": chunks[0][:100] + "..." if chunks else "Ejemplo de fragmento de texto procesado",
                    "embedding_dimensions": len(embeddings[0]) if embeddings else 768,
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": index_algorithm.upper()
                },
                "output": {
                    "collection_name": f"demo_collection_{distance_metric}_{index_algorithm}",
                    "vectors_uploaded": len(chunks),
                    "total_in_collection": len(chunks),
                    "vector_size": 768,
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": index_algorithm.upper(),
                    "hnsw_params": {
                        "m": 16,
                        "ef_construct": 100
                    },
                    "status": "demo_simulation"
                },
                "code_example": """
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Conectar a Qdrant
client = QdrantClient(host="qdrant", port=6333)

# Crear colección con HNSW
client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
        hnsw_config={
            "m": 16,              # Conexiones por capa
            "ef_construct": 100   # Calidad de construcción
        }
    )
)

# Subir vectores
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"content": text, "metadata": {...}}
    )
    for text, embedding in zip(texts, embeddings)
]
client.upsert(collection_name="demo_collection", points=points)
""",
                "curl_example": """
# Crear colección via API REST
curl -X PUT 'http://qdrant:6333/collections/demo_collection' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine",
      "hnsw_config": {"m": 16, "ef_construct": 100}
    }
  }'

# Subir punto
curl -X PUT 'http://qdrant:6333/collections/demo_collection/points' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "points": [
      {
        "id": 1,
        "vector": [0.1, 0.2, ...],
        "payload": {"content": "text", "metadata": {}}
      }
    ]
  }'
""",
                "explanation": f"""
HNSW (Hierarchical Navigable Small World):
- Algoritmo de grafo para búsqueda aproximada de vecinos más cercanos (ANN)
- Construye múltiples capas de grafos conectados
- Búsqueda O(log n) en vez de O(n) de fuerza bruta
- Trade-off: velocidad vs precisión ajustable con parámetros

Métricas de Distancia:
- Cosine: Mide ángulo entre vectores (mejor para texto)
- Euclidean: Distancia L2 (mejor para imágenes)  
- Dot Product: Producto punto (similar a cosine pero sin normalizar)

ANN (Approximate Nearest Neighbor):
- No garantiza resultado exacto pero es 1000x más rápido aproximadamente
- Para millones de vectores, ANN es prácticamente obligatorio
- Precisión típica: 95-99% con velocidad sub-milisegundo

⚠️ NOTA: Qdrant no disponible - mostrando configuración de demostración
""",
                "connection_error": str(e)
            }

    def step_6_upload_to_pgvector(self, chunks: List[str], embeddings: List[List[float]], distance_metric: str = "cosine", index_algorithm: str = "ivfflat") -> Dict[str, Any]:
        """Step 6: Upload documents to PostgreSQL with pgvector"""

        # Input structure
        input_data = {
            "chunks_count": len(chunks),
            "sample_chunk": chunks[0][:100] + "..." if chunks else "No chunks",
            "embedding_dimensions": len(embeddings[0]) if embeddings else 768,
            "distance_metric": distance_metric.upper(),
            "index_algorithm": index_algorithm.upper()
        }

        try:
            import psycopg2
            import json

            # Connect to PostgreSQL - use Docker service name
            # Algorithm to pgvector operator mapping
            operator_mapping = {
                "cosine": "vector_cosine_ops",
                "euclidean": "vector_l2_ops",
                "dot_product": "vector_ip_ops",
                "manhattan": "vector_l1_ops"
            }

            pgvector_operator = operator_mapping.get(
                distance_metric, "vector_cosine_ops")

            conn = psycopg2.connect(
                host="pgvector_db",
                port=5432,
                database="vectordb",
                user="pguser",
                password="pgpass"
            )
            cur = conn.cursor()

            # Ensure table exists with proper schema
            cur.execute("""
                CREATE TABLE IF NOT EXISTS demo_vectors (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(768),
                    metadata JSONB,
                    algorithm VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # Create algorithm-specific index for efficient search
            try:
                index_name = f"demo_vectors_embedding_{distance_metric}_{index_algorithm}_idx"
                if index_algorithm == "ivfflat":
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON demo_vectors 
                        USING ivfflat (embedding {pgvector_operator})
                        WITH (lists = 100);
                    """)
                else:
                    # For exact search, no special index needed
                    pass
            except:
                pass  # Index might already exist

            # Insert vectors with algorithm info
            inserted_count = 0
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                embedding_json = json.dumps(embedding)
                metadata_json = json.dumps({
                    "chunk_index": idx,
                    "source": "demo_pipeline",
                    "timestamp": "2025-11-10",
                    "distance_metric": distance_metric,
                    "index_algorithm": index_algorithm
                })

                cur.execute("""
                    INSERT INTO demo_vectors (content, embedding, metadata, algorithm)
                    VALUES (%s, %s::vector, %s::jsonb, %s)
                """, (chunk, embedding_json, metadata_json, f"{distance_metric}_{index_algorithm}"))
                inserted_count += 1

            conn.commit()

            # Get table stats
            cur.execute("SELECT COUNT(*) FROM demo_vectors WHERE algorithm = %s;",
                        (f"{distance_metric}_{index_algorithm}",))
            total_vectors = cur.fetchone()[0]

            cur.close()
            conn.close()

            return {
                "step": f"6. Subir a PostgreSQL + pgvector ({distance_metric.upper()} + {index_algorithm.upper()})",
                "description": f"Almacenar vectores en PostgreSQL con índice {index_algorithm.upper()} para búsqueda rápida usando distancia {distance_metric.upper()}",
                "database": "PostgreSQL + pgvector",
                "algorithm": f"{index_algorithm.upper()} (Index) + {distance_metric.upper()} (Distance)",
                "input": input_data,
                "output": {
                    "table_name": "demo_vectors",
                    "vectors_uploaded": inserted_count,
                    "total_in_table": total_vectors,
                    "vector_size": 768,
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": index_algorithm.upper(),
                    "distance_operator": f"{pgvector_operator} ({distance_metric} distance)",
                    "index_name": f"demo_vectors_embedding_{distance_metric}_{index_algorithm}_idx",
                    "status": "success"
                },
                "ivfflat_params": {
                    "lists": 100,
                    "explanation": "Divide vectores en 100 clusters. Búsqueda solo en clusters relevantes (ANN)."
                },
                "code_example": """
import psycopg2
import json

# Conectar a PostgreSQL
conn = psycopg2.connect(
    host="localhost", database="vectordb",
    user="pguser", password="pgpass"
)
cur = conn.cursor()

# Crear tabla con vector column
cur.execute('''
    CREATE TABLE demo_vectors (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(768),
        metadata JSONB
    );
''')

# Crear índice IVFFlat para ANN search
cur.execute('''
    CREATE INDEX ON demo_vectors 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
''')

# Insertar vector
embedding_json = json.dumps(embedding_list)
cur.execute(
    "INSERT INTO demo_vectors (content, embedding, metadata) "
    "VALUES (%s, %s::vector, %s::jsonb)",
    (text, embedding_json, metadata_json)
)
conn.commit()
""",
                "sql_queries": """
-- Buscar vectores similares con cosine distance
SELECT 
    content,
    1 - (embedding <=> %s::vector) as similarity,
    embedding <=> %s::vector as distance
FROM demo_vectors
ORDER BY embedding <=> %s::vector
LIMIT 5;

-- Operadores de distancia disponibles:
-- <->  L2 distance (Euclidean)
-- <#>  Inner product (negative dot product)  
-- <=>  Cosine distance

-- Ver estadísticas del índice
SELECT * FROM pg_indexes WHERE tablename = 'demo_vectors';
""",
                "explanation": """
pgvector Extension:
- Agrega tipos de datos vector a PostgreSQL
- Operadores optimizados para similaridad
- Índices especializados (IVFFlat, HNSW)

IVFFlat Index:
- Inverted File with Flat compression
- Divide espacio vectorial en clusters (lists)
- Búsqueda solo en clusters cercanos (ANN)
- Trade-off: lists más grande = más preciso pero más lento

Ventajas pgvector:
- Todo en una base de datos relacional
- ACID transactions + vectores
- Joins entre datos relacionales y vectores  
- Sintaxis SQL familiar

Desventajas vs Qdrant:
- Menos optimizado que DB vectorial dedicada
- No escala tan bien para más de 10M vectores
- Menos features avanzados (filtros, scoring, etc)
"""
            }
        except Exception as e:
            return {
                "step": "6. Subir a PostgreSQL + pgvector",
                "description": f"Almacenar vectores en PostgreSQL con índice {index_algorithm.upper()}",
                "input": input_data,
                "output": {
                    "error": str(e),
                    "status": "failed"
                },
                "explanation": f"Error al subir a PostgreSQL: {str(e)}. Asegúrate de que PostgreSQL con pgvector esté corriendo."
            }


def create_demo_html(demo_steps: List[Dict[str, Any]], query: str, model: str = "phi3:mini",
                     storage_type: str = "both", distance_metric: str = "cosine",
                     index_algorithm: str = "hnsw") -> str:
    """Crear demostración HTML completa mostrando todos los pasos del pipeline"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG Pipeline Demo: {query}</title>
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
            border-left: 4px solid #8b5cf6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .step-header {{
            color: #8b5cf6;
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        
        .step-number {{
            background: #8b5cf6;
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
            padding: 20px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Fira Code', 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.5;
            color: #e1e1e1;
            white-space: pre-wrap;
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
            white-space: pre-wrap;
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
            white-space: pre-wrap;
        }}
        
        .explanation {{
            background: rgba(76, 175, 80, 0.1);
            border-left: 3px solid #8b5cf6;
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
            color: #8b5cf6;
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
            background: rgba(0,0,0,0.9);
            padding: 10px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
            max-width: 250px;
            opacity: 0.3;
            transition: opacity 0.3s ease;
        }}
        
        .navigation:hover {{
            opacity: 1;
        }}
        
        .nav-header {{
            color: #8b5cf6;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 0.9em;
        }}
        
        .nav-section {{
            margin-bottom: 15px;
        }}
        
        .nav-section-title {{
            color: #a78bfa;
            font-weight: bold;
            font-size: 0.8em;
            margin-bottom: 5px;
            text-transform: uppercase;
        }}
        
        .nav-link {{
            display: block;
            color: #d1d5db;
            text-decoration: none;
            padding: 3px 0;
            font-size: 0.8em;
            transition: color 0.2s;
        }}
        
        .nav-link:hover {{
            color: #8b5cf6;
        }}
        
        .phase-divider {{
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            padding: 20px;
            margin: 30px 0;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .step-container {{
            margin-bottom: 40px;
            padding: 30px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border-left: 4px solid #8b5cf6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .input-output {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin: 25px 0;
        }}
        
        .explanation {{
            background: rgba(139, 92, 246, 0.1);
            border-left: 3px solid #8b5cf6;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
            line-height: 1.7;
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
        
        /* Form Controls Styles */
        .demo-controls {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0 40px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .controls-header {{
            color: #8b5cf6;
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .control-group {{
            background: rgba(255,255,255,0.03);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .control-label {{
            color: #c7d2fe;
            font-weight: 600;
            font-size: 0.9em;
            margin-bottom: 8px;
            display: block;
        }}
        
        .control-input, .control-select {{
            width: 100%;
            padding: 10px 12px;
            background: rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: #e1e1e1;
            font-size: 0.9em;
            transition: border-color 0.3s ease;
        }}
        
        .control-input:focus, .control-select:focus {{
            outline: none;
            border-color: #8b5cf6;
            box-shadow: 0 0 10px rgba(139, 92, 246, 0.3);
        }}
        
        .control-description {{
            color: #9ca3af;
            font-size: 0.8em;
            margin-top: 5px;
            line-height: 1.4;
        }}
        
        .controls-actions {{
            display: flex;
            gap: 15px;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .demo-btn {{
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}
        
        .demo-btn:hover {{
            background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
        }}
        
        .demo-btn.secondary {{
            background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
        }}
        
        .demo-btn.secondary:hover {{
            background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
        }}
        
        .current-config {{
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            color: #d1fae5;
        }}
        
        .config-title {{
            color: #10b981;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .config-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 0.9em;
        }}
        
        .config-value {{
            color: #a7f3d0;
            font-weight: 600;
        }}
    </style>
</head>
<body>"""

    # Add header
    html += f"""
    <div class="header">
        <h1>{get_svg_icon("experiment", "24", "#8b5cf6")} Demo Completo del Pipeline RAG</h1>
        <p><strong>Consulta:</strong> "{query}"</p>
        <p>Análisis completo paso a paso de texto a búsqueda vectorial</p>
    </div>
    
    <!-- Interactive Demo Controls -->
    <div class="demo-controls">
        <div class="controls-header">
            {get_svg_icon("gear", "20", "#8b5cf6")} Configuración del Pipeline
        </div>
        
        <div class="current-config">
            <div class="config-title">Configuración Actual</div>
            <div class="config-item">
                <span>Consulta:</span>
                <span class="config-value">"{query}"</span>
            </div>
            <div class="config-item">
                <span>Modelo IA:</span>
                <span class="config-value">{model}</span>
            </div>
            <div class="config-item">
                <span>Almacenamiento:</span>
                <span class="config-value">{storage_type.title()}</span>
            </div>
            <div class="config-item">
                <span>Métrica de Distancia:</span>
                <span class="config-value">{distance_metric.title()}</span>
            </div>
            <div class="config-item">
                <span>Algoritmo de Índice:</span>
                <span class="config-value">{index_algorithm.upper()}</span>
            </div>
        </div>
        
        <form id="demoConfigForm" method="get" action="/demo/pipeline">
            <div class="controls-grid">
                <div class="control-group">
                    <label class="control-label">Consulta de Prueba</label>
                    <input type="text" name="q" value="{query}" class="control-input" 
                           placeholder="Ingresa tu pregunta...">
                    <div class="control-description">
                        La pregunta que se procesará en el pipeline
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Modelo de IA</label>
                    <select name="model" class="control-select">
                        <option value="phi3:mini" {"selected" if model == "phi3:mini" else ""}>Phi-3 Mini (Rápido)</option>
                        <option value="phi3:medium" {"selected" if model == "phi3:medium" else ""}>Phi-3 Medium (Balanceado)</option>
                        <option value="llama3.1:8b" {"selected" if model == "llama3.1:8b" else ""}>Llama 3.1 8B</option>
                        <option value="llama3.1:70b" {"selected" if model == "llama3.1:70b" else ""}>Llama 3.1 70B (Preciso)</option>
                        <option value="gemma2:9b" {"selected" if model == "gemma2:9b" else ""}>Gemma 2 9B</option>
                        <option value="mistral:7b" {"selected" if model == "mistral:7b" else ""}>Mistral 7B</option>
                    </select>
                    <div class="control-description">
                        Modelo de lenguaje para generar respuestas
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Tipo de Almacenamiento</label>
                    <select name="storage_type" class="control-select">
                        <option value="both" {"selected" if storage_type == "both" else ""}>Ambos (Qdrant + pgvector)</option>
                        <option value="qdrant" {"selected" if storage_type == "qdrant" else ""}>Solo Qdrant</option>
                        <option value="postgresql" {"selected" if storage_type == "postgresql" else ""}>Solo pgvector</option>
                    </select>
                    <div class="control-description">
                        Base de datos vectorial para almacenar embeddings
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Métrica de Distancia</label>
                    <select name="distance_metric" class="control-select">
                        <option value="cosine" {"selected" if distance_metric == "cosine" else ""}>Cosine (Recomendado)</option>
                        <option value="euclidean" {"selected" if distance_metric == "euclidean" else ""}>Euclidean</option>
                        <option value="dot_product" {"selected" if distance_metric == "dot_product" else ""}>Dot Product</option>
                        <option value="manhattan" {"selected" if distance_metric == "manhattan" else ""}>Manhattan</option>
                    </select>
                    <div class="control-description">
                        Método para calcular similitud entre vectores
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Algoritmo de Índice</label>
                    <select name="index_algorithm" class="control-select">
                        <option value="hnsw" {"selected" if index_algorithm == "hnsw" else ""}>HNSW (Rápido)</option>
                        <option value="ivfflat" {"selected" if index_algorithm == "ivfflat" else ""}>IVFFlat (pgvector)</option>
                        <option value="scalar_quantization" {"selected" if index_algorithm == "scalar_quantization" else ""}>Scalar Quantization</option>
                        <option value="exact" {"selected" if index_algorithm == "exact" else ""}>Exact Search (Preciso)</option>
                    </select>
                    <div class="control-description">
                        Algoritmo para acelerar búsquedas vectoriales
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Formato de Respuesta</label>
                    <select name="response_format" class="control-select">
                        <option value="html" selected>HTML (Visual)</option>
                        <option value="json">JSON (Datos)</option>
                    </select>
                    <div class="control-description">
                        Formato de salida del pipeline
                    </div>
                </div>
            </div>
            
            <div class="controls-actions">
                <button type="submit" class="demo-btn">
                    {get_svg_icon("refresh", "16", "#ffffff")} Regenerar Demo
                </button>
                <a href="/pipeline" class="demo-btn secondary">
                    {get_svg_icon("gear", "16", "#ffffff")} Gestión Pipeline
                </a>
                <a href="/ai?q={query.replace(' ', '+')}&distance_metric={distance_metric}&index_algorithm={index_algorithm}" class="demo-btn secondary">
                    {get_svg_icon("brain", "16", "#ffffff")} Probar en AI
                </a>
                <a href="/" class="demo-btn secondary">
                    {get_svg_icon("home", "16", "#ffffff")} Inicio
                </a>
            </div>
        </form>
    </div>
    
    <div class="navigation">
        <div class="nav-header">Navegación del Pipeline</div>
        
        <div class="nav-section">
            <div class="nav-section-title">Fase 1: Preparación</div>
            <a href="#step1" class="nav-link">1. Análisis de Texto</a>
            <a href="#step2" class="nav-link">2. Limpieza de Texto</a>
            <a href="#step3" class="nav-link">3. Generación Embeddings</a>
        </div>
        
        <div class="nav-section">
            <div class="nav-section-title">Fase 2: Almacenamiento</div>
            <a href="#step4" class="nav-link">4. Configuración Storage</a>
            <a href="#step5" class="nav-link">5. Upload Qdrant</a>
            <a href="#step6" class="nav-link">6. Upload PostgreSQL</a>
        </div>
        
        <div class="nav-section">
            <div class="nav-section-title">Fase 3: Búsqueda</div>
            <a href="#step7" class="nav-link">7. Procesar Query</a>
            <a href="#step8" class="nav-link">8. Búsqueda Qdrant</a>
            <a href="#step9" class="nav-link">9. Búsqueda pgvector</a>
            <a href="#step10" class="nav-link">10. Similitud Matemática</a>
        </div>
        
        <div class="nav-section">
            <div class="nav-section-title">Fase 4: Respuesta IA</div>
            <a href="#step11" class="nav-link">11. Clasificar Resultados</a>
            <a href="#step12" class="nav-link">12. Generar Respuesta</a>
        </div>
    </div>"""

    for i, step in enumerate(demo_steps, 1):
        step_id = f"step{i}"

        # Add phase dividers
        if i == 1:
            html += f"""
    <div class="phase-divider">
        FASE 1: PREPARACIÓN DE DOCUMENTOS
        <div style="font-size: 0.8em; font-weight: normal; margin-top: 5px;">
            Análisis, limpieza y vectorización del texto
        </div>
    </div>"""
        elif i == 4:
            html += f"""
    <div class="phase-divider">
        FASE 2: ALMACENAMIENTO EN BASES DE DATOS
        <div style="font-size: 0.8em; font-weight: normal; margin-top: 5px;">
            Configuración e ingesta de vectores en Qdrant y PostgreSQL
        </div>
    </div>"""
        elif i == 7:
            html += f"""
    <div class="phase-divider">
        FASE 3: CONSULTAS Y BÚSQUEDAS
        <div style="font-size: 0.8em; font-weight: normal; margin-top: 5px;">
            Procesamiento de consultas y búsqueda de similitud
        </div>
    </div>"""
        elif i == 11:
            html += f"""
    <div class="phase-divider">
        FASE 4: GENERACIÓN DE RESPUESTA CON IA
        <div style="font-size: 0.8em; font-weight: normal; margin-top: 5px;">
            Clasificación de resultados y generación de respuesta con modelo de lenguaje
        </div>
    </div>"""

        html += f"""
    <div id="{step_id}" class="step-container">
        <div class="step-header">
            <div class="step-number">{i}</div>
            {step['step']}
        </div>
        
        <div class="description">{step.get('description', step.get('step', ''))}</div>
        
        <div class="input-output">
            <div class="input-section">
                <div class="section-title">{get_svg_icon("input", "16", "#8b5cf6")} Entrada</div>
                <div class="json-block">{format_json_with_syntax_highlighting(step.get('input', {"message": "No hay datos de entrada específicos para este paso"}))}</div>
            </div>
            
            <div class="output-section">
                <div class="section-title">{get_svg_icon("output", "16", "#FF9800")} Salida</div>
                <div class="json-block">{format_json_with_syntax_highlighting(step.get('output', {"message": "No hay datos de salida disponibles"}))}</div>
            </div>
        </div>
        
        <div class="explanation">
            <strong>{get_svg_icon("idea", "16", "#2196F3")} Explicación:</strong><br><br>
            <span style="white-space: pre-wrap;">{step.get('explanation', 'Sin explicación disponible')}</span>
        </div>
        
        <div class="section-title">{get_svg_icon("code", "16", "#9C27B0")} Ejemplo de Código</div>
        <div class="code-block">{step.get('code_example', '# Sin ejemplo de código disponible')}</div>"""

        # Add special sections for specific steps
        if 'curl_command' in step:
            html += f"""
        <div class="section-title">{get_svg_icon("web", "16", "#00BCD4")} Acceso Directo a API</div>
        <div class="curl-command">{step['curl_command']}</div>"""

        if 'formulas' in step:
            html += f"""
        <div class="section-title">{get_svg_icon("formula", "16", "#795548")} Fórmulas Matemáticas</div>
        <div class="math-formula">
            <strong>Similitud de Coseno:</strong> {step['formulas']['cosine_similarity']}<br>
            <strong>Distancia de Coseno:</strong> {step['formulas']['cosine_distance']}<br>
            <strong>Producto Punto:</strong> {step['formulas']['dot_product']}<br>
            <strong>Norma Vectorial:</strong> {step['formulas']['vector_norm']}
        </div>"""

        html += """
    </div>"""

    html += f"""
    <div class="header" style="margin-top: 40px;">
        <h2>{get_svg_icon("experiment", "20", "#8b5cf6")} ¡Pipeline Completo!</h2>
        <p>Esta demostración mostró el pipeline completo RAG desde texto crudo hasta resultados clasificados.</p>
        <p><a href="/" style="color: #8b5cf6;">← Volver al Menú Principal</a></p>
    </div>
</body>
</html>"""

    return html
