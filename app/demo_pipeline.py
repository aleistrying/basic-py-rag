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
            # Configuración específica para contenedores Docker
            torch.set_default_dtype(torch.float32)
            if hasattr(torch, 'set_default_device'):
                torch.set_default_device('cpu')

            from sentence_transformers import SentenceTransformer

            # Configuración específica para evitar problemas de meta tensors
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

            # Cargar modelo con configuración Docker-safe
            self.model = SentenceTransformer(
                'intfloat/multilingual-e5-base',
                device='cpu',
                trust_remote_code=False,
                use_auth_token=False
            )

            # Forzar que todos los parámetros estén en CPU y en formato float32
            self.model = self.model.cpu()
            for param in self.model.parameters():
                if param.device.type != 'cpu':
                    param.data = param.data.cpu()
                if param.dtype != torch.float32:
                    param.data = param.data.float()

            # Prueba segura del modelo
            with torch.no_grad():
                test_embedding = self.model.encode(
                    ["test"], show_progress_bar=False, convert_to_tensor=False)

            self.model_loaded = True
            logger.info(
                "✅ Modelo E5 cargado exitosamente en contenedor Docker")
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

    def step_4_query_processing(self, query: str) -> Dict[str, Any]:
        """Step 4: Process user query into embedding"""

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
            "step": "4. Procesamiento de Consulta",
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
            "step": "5. Búsqueda Vectorial (Qdrant)",
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
            "step": "6. Búsqueda Vectorial (PostgreSQL pgvector)",
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
            "step": "7. Matemáticas de Similitud",
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
            "step": "8. Clasificación y Presentación de Resultados",
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


def create_demo_html(demo_steps: List[Dict[str, Any]], query: str) -> str:
    """Crear demostración HTML completa mostrando todos los pasos del pipeline"""

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
        <h1>🔬 Demo Completo del Pipeline RAG</h1>
        <p><strong>Consulta:</strong> "{query}"</p>
        <p>Análisis completo paso a paso de texto a búsqueda vectorial</p>
    </div>
    
    <div class="navigation">
        <strong>Navegación Rápida:</strong>
        <a href="#step1" class="nav-link">1. Analizar Texto</a>
        <a href="#step2" class="nav-link">2. Limpiar Texto</a>
        <a href="#step3" class="nav-link">3. Embeddings</a>
        <a href="#step4" class="nav-link">4. Procesar Consulta</a>
        <a href="#step5" class="nav-link">5. Búsqueda Qdrant</a>
        <a href="#step6" class="nav-link">6. Búsqueda pgvector</a>
        <a href="#step7" class="nav-link">7. Matemáticas</a>
        <a href="#step8" class="nav-link">8. Clasificación</a>
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
                <div class="section-title">📥 Entrada</div>
                <div class="json-block">{json.dumps(step.get('input', {}), indent=2, ensure_ascii=False)}</div>
            </div>
            
            <div class="output-section">
                <div class="section-title">📤 Salida</div>
                <div class="json-block">{json.dumps(step.get('output', {}), indent=2, ensure_ascii=False)}</div>
            </div>
        </div>
        
        <div class="explanation">
            <strong>💡 Explicación:</strong> {step.get('explanation', 'Sin explicación disponible')}
        </div>
        
        <div class="section-title">💻 Ejemplo de Código</div>
        <div class="code-block">{step.get('code_example', '# Sin ejemplo de código disponible')}</div>
"""

        # Add special sections for specific steps
        if 'curl_command' in step:
            html += f"""
        <div class="section-title">🌐 Acceso Directo a API</div>
        <div class="curl-command">{step['curl_command']}</div>
"""

        if 'formulas' in step:
            html += f"""
        <div class="section-title">📐 Fórmulas Matemáticas</div>
        <div class="math-formula">
            <strong>Similitud de Coseno:</strong> {step['formulas']['cosine_similarity']}<br>
            <strong>Distancia de Coseno:</strong> {step['formulas']['cosine_distance']}<br>
            <strong>Producto Punto:</strong> {step['formulas']['dot_product']}<br>
            <strong>Norma Vectorial:</strong> {step['formulas']['vector_norm']}
        </div>
"""

        html += "</div>"

    html += """
    <div class="header" style="margin-top: 40px;">
        <h2>🎯 ¡Pipeline Completo!</h2>
        <p>Esta demostración mostró el pipeline completo RAG desde texto crudo hasta resultados clasificados.</p>
        <p><a href="/" style="color: #4CAF50;">← Volver al Menú Principal</a></p>
    </div>
</body>
</html>
"""

    return html
