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

        # Equivalent curl command for PostgreSQL (simplified for display)
        curl_command = f"# PostgreSQL vector search (via psql or REST API)\npsql \"{PGVECTOR_DATABASE_URL}\" -c \"\\\n  SELECT id, content, metadata, \\\n    embedding <=> '[{embedding_str[:50]}...]'::vector as distance \\\n  FROM docs_clean \\\n  ORDER BY embedding <=> '[{embedding_str[:50]}...]'::vector \\\n  LIMIT {limit};\""

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
            "step": "10. Matemáticas de Similitud",
            "description": "Cálculos matemáticos detrás de la búsqueda de similitud vectorial",
            "input": {
                "query_vector_norm": float(np.linalg.norm(query_vec)),
                "calculation_method": "Similitud de Coseno"
            },
            "calculations": similarity_calculations,
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
            "step": "12. Clasificación y Presentación de Resultados",
            "description": "Clasificar resultados de búsqueda por relevancia y preparar respuesta final",
            "input": search_results,
            "output": ranked_results,
            "ranking_criteria": {
                "primary": "Puntuación de similitud de coseno (Qdrant) o distancia (pgvector)",
                "secondary": "Calidad del contenido y completitud",
                "presentation": "Top 3 resultados con metadatos y atribución de fuente"
            },
            "code_example": """
# Clasificar resultados por puntuación de similitud
def rank_results(results, score_key="score", reverse=True):
    return sorted(results, key=lambda x: x.get(score_key, 0), reverse=reverse)

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

    def step_9_ai_generation(self, ranked_results: List[Dict], query: str, model: str = "phi3:mini", distance_metric: str = "cosine", index_algorithm: str = "hnsw") -> Dict[str, Any]:
        """Step 9: Generate AI response from ranked results"""

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
                    # Extract metadata for source attribution
                    doc_name = result.get("path", "").replace(
                        "./data/raw/", "").replace(".pdf", "")
                    page_info = f", p.{result.get('page')}" if result.get(
                        'page') else ""
                    context_parts.append(
                        f"- ({doc_name}{page_info}) {content[:300]}...")
                    sources_used.append({
                        "rank": i + 1,
                        "document": doc_name,
                        "page": result.get('page'),
                        "similarity": result.get('score', 0),
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
            "step": "15. Generación de Respuesta con IA",
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
WITH ({("lists = " + str(current_index["pgvector_config"]["lists"])) if current_index["pgvector_config"] else "-- exact search"});

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
{'- Colección: demo_pipeline_collection' if storage_type in ["qdrant", "both"] else ''}
{'- Índice HNSW para búsqueda ANN rápida' if storage_type in ["qdrant", "both"] else ''}
{'- Payload con metadatos estructurados' if storage_type in ["qdrant", "both"] else ''}

{'🟦 POSTGRESQL STORAGE:' if storage_type in ["postgresql", "both"] else ''}
{'- Tabla: demo_pipeline_vectors' if storage_type in ["postgresql", "both"] else ''}
{'- Índice IVFFlat para búsquedas eficientes' if storage_type in ["postgresql", "both"] else ''}
{'- JSONB metadata + vector type nativo' if storage_type in ["postgresql", "both"] else ''}

⚡ OPTIMIZACIÓN:
- Vectores normalizados para similitud coseno
- Índices configurados para balance velocidad/precisión  
- Metadatos permiten filtrado avanzado
- Batch insertion para mejor performance
"""

        return storage_results

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
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=len(embeddings[0]),
                        distance=qdrant_distance,  # Use selected distance metric
                        # HNSW parameters for efficient ANN search
                        hnsw_config={
                            "m": 16,              # Number of connections per layer
                            "ef_construct": 100    # Size of candidate list for construction
                        }
                    )
                )

            # Upload vectors with metadata
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk,
                        "chunk_index": idx,
                        "source": "demo_pipeline",
                        "timestamp": "2025-11-10"
                    }
                )
                points.append(point)

            # Batch upload
            client.upsert(collection_name=collection_name, points=points)

            # Get collection info
            collection_info = client.get_collection(collection_name)

            return {
                "step": f"5. Subir a Qdrant con {index_algorithm.upper()} ({distance_metric.upper()})",
                "description": f"Cargar vectores en Qdrant usando índice {index_algorithm.upper()} para búsqueda ANN eficiente con distancia {distance_metric.upper()}",
                "database": "Qdrant",
                "algorithm": f"{index_algorithm.upper()} + {distance_metric.upper()} distance",
                "input": {
                    "embeddings_count": len(embeddings),
                    "vector_dimensions": len(embeddings[0]) if embeddings else 768,
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": index_algorithm.upper(),
                    "sample_vector": embeddings[0][:5] if embeddings else [0.1, -0.2, 0.3, 0.0, -0.1],
                    "collection_name": collection_name
                },
                "output": {
                    "collection_name": collection_name,
                    "vectors_uploaded": len(points),
                    "vector_size": len(embeddings[0]),
                    "distance_metric": distance_metric.upper(),
                    "index_algorithm": index_algorithm.upper(),
                    "qdrant_distance": qdrant_distance.name if hasattr(qdrant_distance, 'name') else str(qdrant_distance),
                    "indexed_count": collection_info.vectors_count,
                    "status": "success"
                },
                "hnsw_params": {
                    "m": 16,
                    "ef_construct": 100,
                    "explanation": "M controla conexiones por capa (mayor = más preciso pero más memoria). EF_construct controla calidad de índice."
                },
                "code_example": """
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Conectar a Qdrant
client = QdrantClient(host="localhost", port=6333)

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
curl -X PUT 'http://localhost:6333/collections/demo_collection' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine",
      "hnsw_config": {"m": 16, "ef_construct": 100}
    }
  }'

# Subir punto
curl -X PUT 'http://localhost:6333/collections/demo_collection/points' \\
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
                "explanation": """
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
                "step": "5. Subir a Qdrant con HNSW",
                "description": "Cargar vectores en Qdrant usando índice HNSW para búsqueda ANN eficiente",
                "database": "Qdrant",
                "algorithm": "HNSW (Hierarchical Navigable Small World)",
                "output": {
                    "collection_name": "demo_collection",
                    "vectors_uploaded": len(embeddings),
                    "vector_size": len(embeddings[0]) if embeddings else 768,
                    "distance_metric": "Cosine Similarity",
                    "status": "demo_mode",
                    "error": str(e)
                },
                "hnsw_params": {
                    "m": 16,
                    "ef_construct": 100,
                    "explanation": "M controla conexiones por capa (mayor = más preciso pero más memoria). EF_construct controla calidad de índice."
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
                "explanation": "Asegúrate de que Qdrant esté corriendo en qdrant:6333. En modo demo, se muestran las operaciones que se realizarían."
            }

    def step_6_upload_to_pgvector(self, chunks: List[str], embeddings: List[List[float]], distance_metric: str = "cosine", index_algorithm: str = "ivfflat") -> Dict[str, Any]:
        """Step 6: Upload documents to PostgreSQL with pgvector"""
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
                "error": str(e),
                "explanation": "Asegúrate de que PostgreSQL con pgvector esté corriendo en localhost:5432"
            }

    def step_11_ann_vs_exact_search(self, query_embedding: List[float]) -> Dict[str, Any]:
        """Step 11: Compare ANN (Approximate) vs Exact search performance"""
        import time
        import numpy as np

        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(host="qdrant", port=6333)
            collection_name = "demo_collection"

            # ANN Search with HNSW
            start_ann = time.time()
            ann_results = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=5,
                search_params={"hnsw_ef": 128}  # Search quality parameter
            ).points
            ann_time = (time.time() - start_ann) * 1000  # ms

            # Exact Search (disable HNSW)
            start_exact = time.time()
            exact_results = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=5,
                search_params={"exact": True}  # Force exact search
            ).points
            exact_time = (time.time() - start_exact) * 1000  # ms

            # Compare results
            ann_scores = [r.score for r in ann_results]
            exact_scores = [r.score for r in exact_results]

            # Calculate precision: how many ANN results match exact top-5
            ann_ids = {r.id for r in ann_results}
            exact_ids = {r.id for r in exact_results}
            precision = len(ann_ids.intersection(exact_ids)) / len(exact_ids)

            return {
                "step": "11. ANN vs Búsqueda Exacta",
                "description": "Comparar búsqueda aproximada (HNSW) vs búsqueda exacta (fuerza bruta)",
                "output": {
                    "ann_search": {
                        "time_ms": round(ann_time, 2),
                        "top_scores": [round(s, 4) for s in ann_scores],
                        "algorithm": "HNSW (Approximate)"
                    },
                    "exact_search": {
                        "time_ms": round(exact_time, 2),
                        "top_scores": [round(s, 4) for s in exact_scores],
                        "algorithm": "Brute Force (Exact)"
                    },
                    "comparison": {
                        "speedup": round(exact_time / ann_time, 2),
                        "precision": round(precision * 100, 2),
                        "verdict": f"ANN es {round(exact_time/ann_time, 1)}x más rápido con {round(precision*100, 1)}% precisión"
                    }
                },
                "code_example": """
from qdrant_client import QdrantClient

client = QdrantClient("localhost", 6333)

# Búsqueda ANN con HNSW (rápida, aproximada)
ann_results = client.query_points(
    collection_name="demo_collection",
    query=query_embedding,
    limit=5,
    search_params={"hnsw_ef": 128}  # Mayor = más preciso
).points

# Búsqueda Exacta (lenta, precisa al 100%)
exact_results = client.query_points(
    collection_name="demo_collection", 
    query=query_embedding,
    limit=5,
    search_params={"exact": True}
).points
""",
                "explanation": """
🚀 ANN (Approximate Nearest Neighbor):
- Sacrifica pequeña precisión por velocidad dramática
- Típicamente 95-99% precisión con 100-1000x velocidad
- Esencial para bases de datos con >100k vectores
- Parámetro hnsw_ef controla trade-off velocidad/precisión

🎯 Búsqueda Exacta (Brute Force):
- Compara query con TODOS los vectores
- 100% precisión garantizada
- O(n) complejidad - se vuelve muy lenta
- Solo viable para datasets pequeños (<10k vectores)

📊 Cuándo usar cada una:
- ANN: Producción, latencia crítica, millones de vectores
- Exact: Desarrollo, validación, datasets pequeños, máxima precisión requerida

💡 Benchmark típico (1M vectores):
- ANN: ~5ms para top-5, 98% precisión
- Exact: ~5000ms para top-5, 100% precisión
- En producción, SIEMPRE usar ANN
"""
            }
        except Exception as e:
            return {
                "step": "11. ANN vs Búsqueda Exacta",
                "error": str(e),
                "explanation": "Requiere colección demo_collection en Qdrant"
            }

    def step_14_cosine_similarity_calculation(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> Dict[str, Any]:
        """Step 14: Detailed cosine similarity calculation with math"""
        import numpy as np

        # Use first 10 dims for visualization
        query_vec = np.array(query_embedding[:10])
        doc_vec = np.array(
            doc_embeddings[0][:10]) if doc_embeddings else np.random.randn(10)

        # Step by step cosine calculation
        dot_product = np.dot(query_vec, doc_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norm = np.linalg.norm(doc_vec)
        cosine_sim = dot_product / (query_norm * doc_norm)

        # Full vector calculation
        full_query = np.array(query_embedding)
        full_doc = np.array(doc_embeddings[0]) if doc_embeddings else np.random.randn(
            len(query_embedding))
        full_cosine = np.dot(full_query, full_doc) / \
            (np.linalg.norm(full_query) * np.linalg.norm(full_doc))

        # Calculate for multiple documents
        similarities = []
        for doc_emb in doc_embeddings[:5]:
            doc_array = np.array(doc_emb)
            sim = np.dot(full_query, doc_array) / \
                (np.linalg.norm(full_query) * np.linalg.norm(doc_array))
            similarities.append(float(sim))

        return {
            "step": "14. Cálculo de Similitud Coseno",
            "description": "Matemática detallada de cómo se calcula la similitud entre vectores",
            "formula": "cosine_similarity = (A · B) / (||A|| × ||B||)",
            "calculation_steps": {
                "1_dot_product": {
                    "formula": "A · B = Σ(a_i × b_i)",
                    "value": float(dot_product),
                    "explanation": "Suma de productos elemento por elemento"
                },
                "2_query_magnitude": {
                    "formula": "||A|| = √(Σ(a_i²))",
                    "value": float(query_norm),
                    "explanation": "Longitud del vector query (norma L2)"
                },
                "3_doc_magnitude": {
                    "formula": "||B|| = √(Σ(b_i²))",
                    "value": float(doc_norm),
                    "explanation": "Longitud del vector documento"
                },
                "4_cosine_similarity": {
                    "formula": f"{dot_product:.4f} / ({query_norm:.4f} × {doc_norm:.4f})",
                    "value": float(cosine_sim),
                    "explanation": "Similitud coseno normalizada entre -1 y 1"
                }
            },
            "output": {
                "sample_calculation": {
                    "dimensions_shown": 10,
                    "cosine_similarity": float(cosine_sim)
                },
                "full_vector_calculation": {
                    "dimensions": len(query_embedding),
                    "cosine_similarity": float(full_cosine)
                },
                "multiple_documents": {
                    "similarities": similarities,
                    "best_match_index": int(np.argmax(similarities)),
                    "best_match_score": float(max(similarities))
                }
            },
            "code_example": """
import numpy as np

def cosine_similarity(vec_a, vec_b):
    # Paso 1: Producto punto
    dot_product = np.dot(vec_a, vec_b)
    
    # Paso 2: Magnitudes (normas)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Paso 3: Similitud coseno
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity

# Ejemplo con vectores de 768 dimensiones
query_emb = np.array([...])  # Vector de consulta
doc_emb = np.array([...])    # Vector de documento

similarity = cosine_similarity(query_emb, doc_emb)
print(f"Similitud: {similarity:.4f}")

# Para múltiples documentos
docs = [doc1_emb, doc2_emb, doc3_emb]
similarities = [cosine_similarity(query_emb, doc) for doc in docs]
best_match = np.argmax(similarities)
""",
            "explanation": """
📐 Similitud Coseno Explicada:
- Mide el ángulo entre dos vectores
- Rango: -1 (opuestos) a 1 (idénticos)
- 0 = vectores ortogonales (no relacionados)
- Ignora magnitud, solo dirección

🔢 Por qué funciona para texto:
- Vectores de palabras similares apuntan en direcciones similares
- "Rey" - "Hombre" + "Mujer" ≈ "Reina" funciona por geometría
- Embeddings capturan relaciones semánticas como geometría

⚡ Optimizaciones en producción:
- Vectores pre-normalizados: similarity = dot_product
- SIMD instructions para paralelizar cálculos
- GPU acceleration para batch processing
- Índices ANN evitan calcular todas las similaridades

📊 Interpretación de scores:
- > 0.9: Muy similar (casi duplicados)
- 0.7-0.9: Alta similaridad semántica
- 0.5-0.7: Relacionados temáticamente  
- < 0.5: Poco o nada relacionados
"""
        }

    def step_15_ranking_and_reranking(self, search_results: List[Dict], query: str) -> Dict[str, Any]:
        """Step 15: Initial ranking + reranking strategies"""

        # Initial ranking by cosine similarity
        initial_ranked = sorted(
            search_results, key=lambda x: x.get('score', 0), reverse=True)

        # Reranking strategy 1: Boost recent documents
        reranked_recency = initial_ranked.copy()
        for item in reranked_recency:
            if 'timestamp' in item.get('metadata', {}):
                # Add small boost for recent docs
                item['adjusted_score'] = item['score'] * 1.05
            else:
                item['adjusted_score'] = item['score']
        reranked_recency.sort(key=lambda x: x['adjusted_score'], reverse=True)

        # Reranking strategy 2: Length-based scoring
        reranked_length = initial_ranked.copy()
        for item in reranked_length:
            content_length = len(item.get('content', ''))
            # Prefer medium-length chunks (200-500 chars)
            if 200 <= content_length <= 500:
                length_boost = 1.1
            elif content_length < 200:
                length_boost = 0.9
            else:
                length_boost = 1.0
            item['adjusted_score'] = item['score'] * length_boost
        reranked_length.sort(key=lambda x: x['adjusted_score'], reverse=True)

        # Reranking strategy 3: Diversity (MMR - Maximal Marginal Relevance)
        diversified = [initial_ranked[0]] if initial_ranked else []
        remaining = initial_ranked[1:].copy()

        while remaining and len(diversified) < 5:
            # Select next doc that maximizes: relevance - similarity to already selected
            best_idx = 0
            best_score = -float('inf')

            for idx, candidate in enumerate(remaining):
                relevance = candidate['score']
                # Simple diversity: just alternate selection
                diversity_penalty = 0.1 * len(diversified)
                mmr_score = relevance - diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            diversified.append(remaining.pop(best_idx))

        return {
            "step": "13. Ranking y Re-ranking",
            "description": "Estrategias para ordenar y re-ordenar resultados después de búsqueda vectorial",
            "strategies": {
                "initial_ranking": {
                    "method": "Cosine Similarity",
                    "description": "Ordenar por score de similitud coseno",
                    "top_3_scores": [r['score'] for r in initial_ranked[:3]]
                },
                "rerank_recency": {
                    "method": "Recency Boost",
                    "description": "Dar boost a documentos más recientes",
                    "top_3_scores": [r['adjusted_score'] for r in reranked_recency[:3]],
                    "boost_factor": 1.05
                },
                "rerank_length": {
                    "method": "Length-based Scoring",
                    "description": "Preferir chunks de longitud óptima (200-500 chars)",
                    "top_3_scores": [r['adjusted_score'] for r in reranked_length[:3]],
                    "optimal_range": "200-500 caracteres"
                },
                "rerank_diversity": {
                    "method": "MMR (Maximal Marginal Relevance)",
                    "description": "Balancear relevancia con diversidad",
                    "selected_count": len(diversified),
                    "explanation": "Evita resultados muy similares entre sí"
                }
            },
            "output": {
                "initial_top_3": [
                    {"content": r['content'][:100], "score": r['score']}
                    for r in initial_ranked[:3]
                ],
                "reranked_top_3": [
                    {"content": r['content'][:100], "score": r.get(
                        'adjusted_score', r['score'])}
                    for r in reranked_length[:3]
                ]
            },
            "code_example": """
# Ranking inicial por similitud
results = sorted(search_results, key=lambda x: x['score'], reverse=True)

# Re-ranking con boost de recencia
for result in results:
    days_old = (today - result['date']).days
    recency_boost = 1.0 / (1 + days_old/30)  # Decay over time
    result['final_score'] = result['score'] * (1 + 0.1 * recency_boost)

# Re-ranking con preferencia de longitud
for result in results:
    length = len(result['content'])
    if 200 <= length <= 500:
        length_factor = 1.1  # Boost
    elif length < 100:
        length_factor = 0.8  # Penalize
    else:
        length_factor = 1.0
    result['final_score'] = result['score'] * length_factor

# MMR para diversidad
def mmr_rerank(results, lambda_param=0.5):
    selected = [results[0]]
    remaining = results[1:]
    
    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -float('inf')
        
        for idx, candidate in enumerate(remaining):
            relevance = candidate['score']
            
            # Calcular similaridad máxima con ya seleccionados
            max_sim = max([
                cosine_similarity(candidate['embedding'], s['embedding'])
                for s in selected
            ])
            
            # MMR score: balancear relevancia y diversidad
            mmr_score = lambda_param * relevance - (1-lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected.append(remaining.pop(best_idx))
    
    return selected

reranked = mmr_rerank(results, lambda_param=0.7)
""",
            "explanation": """
🎯 Estrategias de Ranking:

1️⃣ RANKING INICIAL (Similitud Coseno):
   - Primero: ordenar por score de búsqueda vectorial
   - Más simple y directo
   - Base para otros métodos

2️⃣ RE-RANKING POR RECENCIA:
   - Documentos recientes son más relevantes
   - Útil para noticias, logs, datos temporales
   - Trade-off: información vieja puede ser valiosa

3️⃣ RE-RANKING POR LONGITUD:
   - Chunks muy cortos: poco contexto
   - Chunks muy largos: mucho ruido
   - Óptimo: 200-500 caracteres para embeddings

4️⃣ MMR (DIVERSIDAD):
   - Evita resultados redundantes
   - Top-5 cubre más temas
   - Balance: relevancia vs novedad

🏆 MEJOR PRÁCTICA:
   1. Búsqueda vectorial inicial (top-50)
   2. Re-ranking con modelo cross-encoder (top-10)
   3. Aplicar reglas de negocio (filtros, boosts)
   4. MMR para diversidad (top-5 final)

⚡ EN PRODUCCIÓN:
   - A/B testing de estrategias
   - Métricas: MRR, NDCG, Precision@K
   - Logging de clicks para reentrenamiento
"""
        }

    def run_complete_demo_with_storage(self, query: str = "¿Qué es pgvector?", model: str = "phi3:mini", storage_type: str = "both", distance_metric: str = "cosine", index_algorithm: str = "hnsw") -> List[Dict[str, Any]]:
        """Run complete educational pipeline with integrated database storage demos

        Complete Educational Pipeline with Storage - Logical Flow:
        FASE 1: PREPARACIÓN DE DOCUMENTOS
        PASO 1: Análisis de Texto
        PASO 2: Limpieza y Fragmentación de Texto  
        PASO 3: Generación de Embeddings

        FASE 2: ALMACENAMIENTO EN BASES DE DATOS
        PASO 4: Simulación de Almacenamiento (Qdrant y/o PostgreSQL)
        PASO 5: Subir a Qdrant con HNSW
        PASO 6: Subir a PostgreSQL + pgvector

        FASE 3: CONSULTAS Y BÚSQUEDAS
        PASO 7: Procesamiento de Consulta
        PASO 8: Búsqueda Vectorial (Qdrant)
        PASO 9: Búsqueda Vectorial (PostgreSQL pgvector)
        PASO 10: Matemáticas de Similitud
        PASO 11: ANN vs Búsqueda Exacta

        FASE 4: PROCESAMIENTO DE RESULTADOS
        PASO 12: Clasificación y Presentación de Resultados
        PASO 13: Ranking y Re-ranking
        PASO 14: Cálculo de Similitud Coseno (detallado)
        PASO 15: Generación de Respuesta con IA
        """

        demo_steps = []

        # ===== FASE 1: PREPARACIÓN DE DOCUMENTOS =====
        # Paso 1: Análisis de Texto
        step1 = self.step_1_parse_text()
        demo_steps.append(step1)

        # Paso 2: Limpieza y Fragmentación de Texto
        step2 = self.step_2_clean_text(step1["output"]["sentences"])
        demo_steps.append(step2)

        # Paso 3: Generación de Embeddings
        step3 = self.step_3_create_embeddings(step2["output"]["chunks"])
        demo_steps.append(step3)

        # ===== FASE 2: ALMACENAMIENTO EN BASES DE DATOS =====
        # Paso 4: Simulación de Almacenamiento con datos reales
        step4 = self.step_4_storage_simulation(
            step2["output"]["chunks"],
            step3["output"]["full_embeddings"],
            storage_type,
            distance_metric,
            index_algorithm
        )
        demo_steps.append(step4)

        # Paso 5: Subir a Qdrant con HNSW (technical details)
        if storage_type in ["qdrant", "both"]:
            step5 = self.step_5_upload_to_qdrant(
                step2["output"]["chunks"],
                step3["output"]["full_embeddings"],
                distance_metric,
                index_algorithm
            )
            demo_steps.append(step5)

        # Paso 6: Subir a PostgreSQL + pgvector (technical details)
        if storage_type in ["postgresql", "both"]:
            step6 = self.step_6_upload_to_pgvector(
                step2["output"]["chunks"],
                step3["output"]["full_embeddings"],
                distance_metric,
                index_algorithm
            )
            demo_steps.append(step6)

        # ===== FASE 3: CONSULTAS Y BÚSQUEDAS =====
        # Paso 7: Procesamiento de Consulta
        step7 = self.step_7_query_processing(query)
        demo_steps.append(step7)

        # Paso 8: Búsqueda Vectorial (Qdrant)
        if storage_type in ["qdrant", "both"]:
            step8 = self.step_8_vector_search_qdrant(
                step7["output"]["full_embedding"])
            demo_steps.append(step8)

        # Paso 9: Búsqueda Vectorial (PostgreSQL pgvector)
        if storage_type in ["postgresql", "both"]:
            step9 = self.step_9_vector_search_pgvector(
                step7["output"]["full_embedding"])
            demo_steps.append(step9)

        # Paso 10: Matemáticas de Similitud
        step10 = self.step_10_similarity_math(
            step7["output"]["full_embedding"],
            step3["output"]["full_embeddings"]
        )
        demo_steps.append(step10)

        # Paso 11: ANN vs Búsqueda Exacta
        if storage_type in ["qdrant", "both"]:
            step11 = self.step_11_ann_vs_exact_search(
                step7["output"]["full_embedding"])
            demo_steps.append(step11)

        # ===== FASE 4: PROCESAMIENTO DE RESULTADOS =====
        # Paso 12: Clasificación y Presentación de Resultados
        # Use Qdrant results if available, otherwise use PostgreSQL results
        search_results = None
        for step in demo_steps:
            if step.get("step", "").startswith("8.") and "output" in step:
                search_results = step["output"]
                break
            elif step.get("step", "").startswith("9.") and "output" in step:
                search_results = step["output"]

        if search_results:
            step12 = self.step_8_result_ranking(search_results)
        else:
            # Fallback with mock results
            step12 = self.step_8_result_ranking(
                {"error": "No search results available"})
        demo_steps.append(step12)

        # Paso 13: Ranking y Re-ranking
        if isinstance(search_results, list) and search_results:
            step13 = self.step_15_ranking_and_reranking(search_results, query)
        else:
            # Fallback with mock results
            mock_results = [
                {
                    "content": "PostgreSQL con pgvector es una extensión que permite almacenar vectores.",
                    "score": 0.85,
                    "path": "./data/raw/ejemplo.pdf",
                    "page": 10
                }
            ]
            step13 = self.step_15_ranking_and_reranking(mock_results, query)
        demo_steps.append(step13)

        # Paso 14: Cálculo de Similitud Coseno (detallado)
        if step3["output"]["full_embeddings"]:
            step14 = self.step_14_cosine_similarity_calculation(
                step7["output"]["full_embedding"],
                step3["output"]["full_embeddings"][:3]
            )
            demo_steps.append(step14)

        # Paso 15: Generación de Respuesta con IA
        if isinstance(step12["output"], list) and step12["output"]:
            step15 = self.step_9_ai_generation(
                step12["output"], query, model, distance_metric, index_algorithm)
        else:
            # Fallback with mock results for demo
            mock_results = [
                {
                    "rank": 1,
                    "result": {
                        "content": "PostgreSQL con pgvector es una extensión que permite almacenar vectores.",
                        "score": 0.85,
                        "path": "./data/raw/ejemplo.pdf",
                        "page": 10
                    }
                }
            ]
            step15 = self.step_9_ai_generation(
                mock_results, query, model, distance_metric, index_algorithm)
        demo_steps.append(step15)

        return demo_steps

    def run_complete_demo(self, query: str = "¿Qué es pgvector?", model: str = "phi3:mini") -> List[Dict[str, Any]]:
        """Legacy method - redirects to new storage-integrated demo"""
        return self.run_complete_demo_with_storage(query, model, "both")


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
            <div class="nav-section-title">Fase 1: Documentos</div>
            <a href="#step1" class="nav-link">1. Análisis de Texto</a>
            <a href="#step2" class="nav-link">2. Limpieza de Texto</a>
            <a href="#step3" class="nav-link">3. Embeddings</a>
        </div>
        
        <div class="nav-section">
            <div class="nav-section-title">Fase 2: Almacenamiento</div>
            <a href="#step4" class="nav-link">4. Concepto Almacenamiento</a>
            <a href="#step5" class="nav-link">5. Upload Qdrant</a>
            <a href="#step6" class="nav-link">6. Upload PostgreSQL</a>
        </div>
        
        <div class="nav-section">
            <div class="nav-section-title">Fase 3: Consultas</div>
            <a href="#step7" class="nav-link">7. Procesamiento Query</a>
            <a href="#step8" class="nav-link">8. Búsqueda Qdrant</a>
            <a href="#step9" class="nav-link">9. Búsqueda pgvector</a>
            <a href="#step10" class="nav-link">10. Matemáticas</a>
            <a href="#step11" class="nav-link">11. ANN vs Exact</a>
        </div>
        
        <div class="nav-section">
            <div class="nav-section-title">Fase 4: Resultados</div>
            <a href="#step12" class="nav-link">12. Clasificación</a>
            <a href="#step13" class="nav-link">13. Re-ranking</a>
            <a href="#step14" class="nav-link">14. Coseno Detallado</a>
            <a href="#step15" class="nav-link">15. Generación IA</a>
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
        elif i == 12:
            html += f"""
    <div class="phase-divider">
        FASE 4: PROCESAMIENTO DE RESULTADOS
        <div style="font-size: 0.8em; font-weight: normal; margin-top: 5px;">
            Ranking, re-ranking y generación de respuestas con IA
        </div>
    </div>"""

        print(i)
        print(step.get("description", step.get("step", f"Step {i}")))
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
