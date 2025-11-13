# Teoría Completa: Bases de Datos Vectoriales y RAG

**Documento de referencia para el proyecto 1CA217 - Sistemas de Base de Datos Avanzadas**

---

## Tabla de Contenidos

1. [Fundamentos de Bases de Datos Vectoriales](#1-fundamentos-de-bases-de-datos-vectoriales)
2. [Embeddings y Representación Vectorial](#2-embeddings-y-representación-vectorial)
3. [Búsqueda de Vecinos Más Cercanos (ANN)](#3-búsqueda-de-vecinos-más-cercanos-ann)
4. [Inteligencia Artificial Generativa y LLMs](#4-inteligencia-artificial-generativa-y-llms)
5. [Retrieval-Augmented Generation (RAG)](#5-retrieval-augmented-generation-rag)
6. [Comparativa de Bases de Datos Vectoriales](#6-comparativa-de-bases-de-datos-vectoriales)
7. [Arquitectura del Proyecto](#7-arquitectura-del-proyecto)
8. [Pipeline de Procesamiento](#8-pipeline-de-procesamiento)
9. [Técnicas Avanzadas de RAG](#9-técnicas-avanzadas-de-rag)
10. [Optimización y Métricas](#10-optimización-y-métricas)
11. [Despliegue en la Nube](#11-despliegue-en-la-nube)
12. [Aplicaciones Empresariales](#12-aplicaciones-empresariales)
13. [Recomendaciones y Mejores Prácticas](#13-recomendaciones-y-mejores-prácticas)

---

## 1. Fundamentos de Bases de Datos Vectoriales

### 1.1 ¿Qué es una Base de Datos Vectorial?

Una **base de datos vectorial** es un sistema de almacenamiento especializado diseñado para guardar, indexar y consultar datos representados como **vectores numéricos de alta dimensionalidad** (típicamente entre 384 y 1536 dimensiones).

**Características principales:**

- **Optimización para similitud semántica**: A diferencia de las bases de datos tradicionales que buscan coincidencias exactas, las bases vectoriales encuentran elementos _semánticamente similares_.
- **Vectores como ciudadanos de primera clase**: Los vectores no son un complemento, sino el tipo de dato principal.
- **Índices especializados**: Utilizan estructuras de datos específicas (HNSW, IVF, etc.) para búsquedas rápidas en espacios de alta dimensión.
- **Escalabilidad horizontal**: Diseñadas para manejar millones o miles de millones de vectores.

**Analogía práctica:**

Imagina un mapa tridimensional donde cada punto representa un documento. Documentos sobre el mismo tema están más cerca entre sí. Una búsqueda vectorial es como encontrar todos los puntos cercanos a una ubicación dada, en lugar de buscar palabras clave exactas.

### 1.2 Diferencias con Bases de Datos Tradicionales

| Aspecto                    | BD Tradicional (SQL)                     | BD Vectorial                                   |
| -------------------------- | ---------------------------------------- | ---------------------------------------------- |
| **Tipo de búsqueda**       | Coincidencia exacta (`WHERE name = 'X'`) | Similitud semántica (vectores cercanos)        |
| **Estructura de datos**    | Tablas, filas, columnas                  | Vectores + metadatos                           |
| **Índices**                | B-Tree, Hash                             | HNSW, IVF, Annoy                               |
| **Consulta típica**        | `SELECT * WHERE category = 'tech'`       | "Encuentra documentos similares a este vector" |
| **Caso de uso**            | Transacciones, CRUD                      | Búsqueda semántica, recomendaciones, IA        |
| **Métrica de comparación** | Igualdad (=, !=)                         | Distancia (coseno, euclidiana)                 |

### 1.3 Casos de Uso Principales

1. **Búsqueda Semántica**: "Encuentra documentos sobre 'aprendizaje automático'" → recupera textos sobre ML, IA, redes neuronales, aunque no mencionen exactamente "aprendizaje automático".

2. **Sistemas de Recomendación**: "Usuarios que vieron X también vieron Y" basado en similitud de embeddings de comportamiento.

3. **RAG (Retrieval-Augmented Generation)**: Proporcionar contexto relevante a modelos de lenguaje para respuestas fundamentadas.

4. **Detección de Duplicados**: Encontrar imágenes, textos o productos similares a nivel semántico.

5. **Clasificación y Clustering**: Agrupar elementos similares automáticamente.

---

## 2. Embeddings y Representación Vectorial

### 2.1 ¿Qué son los Embeddings?

Un **embedding** es una representación numérica (vector) de un objeto (texto, imagen, audio, etc.) que captura su **significado semántico** en un espacio vectorial continuo.

**Propiedades clave:**

- **Dimensionalidad fija**: Todos los embeddings del mismo modelo tienen la misma cantidad de dimensiones (e.g., 768 para E5-base).
- **Semántica capturada**: Elementos similares tienen vectores cercanos en el espacio.
- **Operaciones matemáticas**: Puedes sumar, restar, promediar vectores para obtener nuevos significados.

**Ejemplo visual:**

```
"perro"     → [0.2, 0.8, 0.1, ..., 0.5]  (768 números)
"cachorro"  → [0.3, 0.7, 0.2, ..., 0.4]  (muy cercano a "perro")
"auto"      → [0.9, 0.1, 0.8, ..., 0.2]  (lejano de "perro")
```

### 2.2 ¿Cómo se Generan los Embeddings?

Los embeddings se crean mediante **modelos de lenguaje** entrenados en grandes corpus de texto (o imágenes, audio, etc.). Estos modelos aprenden a codificar información semántica en vectores densos.

**Modelos populares para texto:**

- **E5 (Embeddings from Bidirectional Encoder Representations)**: Multilingüe, 768 dimensiones
  - `intfloat/multilingual-e5-base` → usado en este proyecto
- **Sentence-BERT**: Optimizado para similitud de oraciones
- **OpenAI Ada-002**: 1536 dimensiones, muy potente pero requiere API
- **BERT, RoBERTa**: Modelos base para muchas variantes

**Proceso de generación:**

```python
# Ejemplo con E5
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-base')

# Texto → Vector (768 números)
texto = "Las bases de datos vectoriales son poderosas"
embedding = model.encode(texto)  # → array de 768 floats
```

### 2.3 Tipos de Embeddings

#### 2.3.1 Embeddings Densos vs. Sparse

- **Densos (Dense)**: Vectores donde la mayoría de valores son no-cero. Capturan semántica profunda. Usado en modelos transformer.

  - Ejemplo: `[0.23, -0.45, 0.78, 0.12, ...]` (768 valores)

- **Sparse (Dispersos)**: Mayoría de valores son cero. Usado en TF-IDF, BM25.
  - Ejemplo: `[0, 0, 1.2, 0, 0, 0, 0, 3.4, ...]` (mayoría ceros)

#### 2.3.2 Embeddings de Query vs. Documento

Modelos como E5 usan **prefijos diferentes** para mejorar la recuperación:

```python
# Query (lo que el usuario busca)
query_emb = encode("query: ¿Qué son bases vectoriales?")

# Documento (contenido indexado)
doc_emb = encode("passage: Las bases vectoriales almacenan...")
```

Esto mejora la alineación entre consultas y documentos en el espacio vectorial.

### 2.4 Propiedades Matemáticas de los Embeddings

#### Similitud Semántica

Vectores cercanos = conceptos similares:

```
distancia("rey" - "hombre" + "mujer") ≈ "reina"
```

#### Clustering Natural

Conceptos relacionados forman clusters en el espacio:

```
["perro", "gato", "conejo"] → cluster de animales
["auto", "camión", "bicicleta"] → cluster de vehículos
```

#### Composicionalidad

Puedes combinar embeddings para crear nuevos conceptos:

```
embed("París") + embed("Francia") - embed("Berlín") ≈ embed("Alemania")
```

---

## 3. Búsqueda de Vecinos Más Cercanos (ANN)

### 3.1 Problema del Nearest Neighbor (NN)

**Definición**: Dado un vector de consulta **q**, encuentra los **k vectores** más cercanos en una base de datos de **n vectores**, usando alguna métrica de distancia.

**Desafío**: En alta dimensionalidad (768 dims), comparar con todos los vectores (fuerza bruta) es **extremadamente lento**:

- 1 millón de vectores × 768 dimensiones = ~770 millones de comparaciones por búsqueda
- Tiempo: O(n × d) donde n = número de vectores, d = dimensiones

### 3.2 Approximate Nearest Neighbor (ANN)

**Solución**: **Sacrificar precisión perfecta por velocidad** usando algoritmos aproximados.

**Trade-off fundamental:**

```
Velocidad ⬆️  ←→  Precisión ⬇️
```

Los algoritmos ANN encuentran vecinos "casi óptimos" en tiempo **sub-linear** O(log n) o mejor, mediante:

1. **Indexación inteligente**: Pre-procesar vectores en estructuras que permitan búsquedas rápidas
2. **Poda del espacio de búsqueda**: Ignorar regiones del espacio que probablemente no contengan vecinos cercanos
3. **Aproximación controlada**: Parámetros para ajustar precisión vs. velocidad

### 3.3 Métricas de Distancia

#### 3.3.1 Similitud del Coseno (Cosine Similarity)

**Métrica más usada** en búsqueda semántica. Mide el **ángulo** entre vectores, ignorando magnitud.

**Fórmula:**

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

**Rango**: -1 (opuestos) a +1 (idénticos)

**Distancia del coseno** (para mantener "menor es más cercano"):

```
distance = 1 - cos(θ)
```

**Ventajas:**

- Invariante a la magnitud (normalización implícita)
- Excelente para comparar direcciones semánticas
- Estándar en NLP

**Cuándo usar:**

- Textos de diferentes longitudes
- Cuando la dirección importa más que la magnitud
- **Recomendación general para texto**

**Ejemplo en código:**

```python
import numpy as np

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Vectores
v1 = [1, 2, 3]
v2 = [2, 4, 6]  # Misma dirección, doble magnitud

print(cosine_distance(v1, v2))  # → 0.0 (idénticos en dirección)
```

#### 3.3.2 Distancia Euclidiana (L2)

Mide la **distancia geométrica directa** entre dos puntos.

**Fórmula:**

```
distance = √(Σ(ai - bi)²)
```

**Rango**: 0 (idénticos) a ∞

**Ventajas:**

- Intuitiva (distancia "real" en el espacio)
- Sensible a la magnitud

**Desventajas:**

- Sensible a la escala de features
- Puede ser afectada por vectores de diferentes magnitudes

**Cuándo usar:**

- Datos numéricos con magnitudes significativas
- Cuando la escala importa (e.g., coordenadas GPS)

#### 3.3.3 Producto Punto (Dot Product)

Mide **similitud direccional y magnitud**.

**Fórmula:**

```
dot(A, B) = Σ(ai × bi)
```

**Rango**: -∞ a +∞

**Nota**: Mayor valor = mayor similitud (inverso a distancia)

**Cuándo usar:**

- Vectores normalizados (equivalente a coseno)
- Cuando magnitud es significativa

#### 3.3.4 Distancia de Manhattan (L1)

Suma de diferencias absolutas.

**Fórmula:**

```
distance = Σ|ai - bi|
```

**Ventajas:**

- Robusta a outliers
- Rápida de calcular

**Cuándo usar:**

- Datos dispersos (sparse)
- Cuando outliers son problemáticos

### 3.4 Algoritmos de Indexación ANN

#### 3.4.1 HNSW (Hierarchical Navigable Small World)

**El algoritmo más usado** en producción para búsqueda vectorial.

**Concepto:**

Construye un **grafo multicapa** donde cada nodo es un vector. Layers superiores son más dispersos (menos nodos), layers inferiores son densos.

**Cómo funciona:**

1. **Construcción**:

   - Layer 0 (base): Contiene TODOS los vectores
   - Layers superiores: Subconjuntos aleatorios decrecientes
   - Cada nodo conecta a sus vecinos más cercanos

2. **Búsqueda** (top-down):
   ```
   1. Iniciar en el layer superior (pocas conexiones)
   2. Navegar hacia el vecino más cercano
   3. Descender al layer siguiente
   4. Repetir hasta layer 0
   5. Retornar k vecinos más cercanos
   ```

**Parámetros clave:**

- **M** (connections): Número de conexiones por nodo (default: 16)
  - Mayor M = mejor precisión, más memoria
- **efConstruction**: Esfuerzo durante construcción del índice (default: 200)
  - Mayor valor = mejor índice, construcción más lenta
- **efSearch**: Esfuerzo durante búsqueda (default: 100)
  - Mayor valor = mejor precisión, búsqueda más lenta

**Ventajas:**

- **Muy rápido**: Búsquedas en tiempo logarítmico
- **Alta precisión**: ~95-99% recall con parámetros adecuados
- **Escalable**: Maneja millones de vectores

**Desventajas:**

- Uso alto de memoria (~4x tamaño de vectores)
- No permite eliminación eficiente de vectores

**Cuándo usar:**

- **Caso general recomendado** para casi todos los escenarios
- Cuando velocidad y precisión son críticas

**En este proyecto:**

```python
# Qdrant usa HNSW por defecto
# Configuración en ingest_config.py
index_algorithm = "hnsw"
```

#### 3.4.2 IVF (Inverted File Index)

**Concepto:**

Divide el espacio en **clusters** mediante k-means. Cada cluster tiene una lista de vectores que contiene.

**Cómo funciona:**

1. **Construcción**:

   - Aplicar k-means para crear n_clusters centroides
   - Asignar cada vector al cluster más cercano
   - Crear índice invertido: cluster → lista de vectores

2. **Búsqueda**:
   ```
   1. Encontrar los n_probes clusters más cercanos al query
   2. Buscar solo dentro de esos clusters (ignorar el resto)
   3. Retornar k vecinos más cercanos de los clusters visitados
   ```

**Parámetros clave:**

- **n_lists** (n_clusters): Número de clusters (default: √n)
- **n_probes**: Cuántos clusters buscar (default: 1-10)
  - Mayor valor = mejor precisión, más lento

**Ventajas:**

- Menor uso de memoria que HNSW
- Construcción más rápida
- Permite actualizaciones

**Desventajas:**

- Menos preciso que HNSW
- Sensible a la distribución de datos

**Cuándo usar:**

- Datasets muy grandes (>10M vectores)
- Cuando la memoria es limitada
- Cuando necesitas actualizaciones frecuentes

**En este proyecto:**

```python
# PostgreSQL con pgvector puede usar IVF
index_algorithm = "ivfflat"
```

#### 3.4.3 Scalar Quantization

**Concepto:**

**Reducir la precisión** de los vectores (e.g., float32 → uint8) para ahorrar memoria y acelerar búsquedas.

**Cómo funciona:**

1. **Construcción**:

   - Calcular min/max por dimensión
   - Mapear valores float a enteros (e.g., 0-255)

   ```
   quantized_value = (value - min) / (max - min) × 255
   ```

2. **Búsqueda**:
   - Cuantizar el query vector
   - Buscar usando vectores cuantizados (operaciones más rápidas)
   - Opcionalmente: re-rank con vectores originales

**Ventajas:**

- **4x reducción de memoria** (float32 → uint8)
- Búsquedas más rápidas (operaciones enteras)
- Compatible con HNSW, IVF

**Desventajas:**

- Pérdida de precisión (~1-3% recall)

**Cuándo usar:**

- Datasets masivos donde memoria es crítica
- Cuando 95% precision es suficiente

**En este proyecto:**

```python
index_algorithm = "scalar_quantization"
```

#### 3.4.4 Exact Search (Fuerza Bruta)

**Concepto:**

Comparar el query vector con **todos** los vectores en la base de datos.

**Ventajas:**

- **100% precisión** (no aproximado)
- Sin overhead de índice

**Desventajas:**

- **O(n)**: Tiempo lineal con el tamaño del dataset
- Extremadamente lento para >100k vectores

**Cuándo usar:**

- Datasets pequeños (<10k vectores)
- Cuando precisión perfecta es crítica
- Baseline para evaluar otros algoritmos

**En este proyecto:**

```python
index_algorithm = "exact"  # Solo para testing/comparación
```

### 3.5 Comparativa de Algoritmos

| Algoritmo        | Velocidad         | Precisión | Memoria     | Actualización | Mejor Para          |
| ---------------- | ----------------- | --------- | ----------- | ------------- | ------------------- |
| **HNSW**         | ⚡⚡⚡ Muy rápido | ✅ 95-99% | 📦 Alta     | ❌ Difícil    | **General purpose** |
| **IVF**          | ⚡⚡ Rápido       | ⚠️ 85-95% | 📦 Media    | ✅ Fácil      | Datasets grandes    |
| **Scalar Quant** | ⚡⚡⚡ Muy rápido | ⚠️ 93-97% | 📦 Muy baja | ✅ Fácil      | Memoria limitada    |
| **Exact**        | 🐌 Lento          | ✅ 100%   | 📦 Baja     | ✅ Fácil      | Datasets pequeños   |

---

## 4. Inteligencia Artificial Generativa y LLMs

### 4.1 ¿Qué es la Inteligencia Artificial Generativa?

**Inteligencia Artificial Generativa (Generative AI)** se refiere a sistemas de IA que pueden **crear contenido nuevo** (texto, imágenes, audio, código) que es similar pero no idéntico a los datos de entrenamiento.

**Características:**

- **Generación creativa**: Produce contenido original, no solo recupera o clasifica
- **Aprendizaje de patrones**: Captura distribuciones de probabilidad de datos de entrenamiento
- **Multimodal**: Puede generar texto, imágenes, audio, video, código

**Ejemplos:**

- **Texto**: GPT-4, Claude, Llama, Phi
- **Imágenes**: DALL-E, Midjourney, Stable Diffusion
- **Audio**: Whisper (transcripción), MusicGen
- **Código**: GitHub Copilot, CodeLlama

### 4.2 ¿Qué es un Large Language Model (LLM)?

Un **LLM (Large Language Model)** es un modelo de IA entrenado en enormes cantidades de texto para entender y generar lenguaje humano.

**Características:**

- **Escala masiva**: Parámetros en miles de millones (GPT-3: 175B, Llama 3: 70B, Phi-3: 3.8B)
- **Preentrenamiento**: Entrenado en texto de internet, libros, código
- **Fine-tuning**: Ajustado para tareas específicas (chat, programación, razonamiento)
- **Emergent abilities**: Capacidades que surgen con escala (razonamiento, matemáticas)

**Arquitectura base**: Transformer (atención multi-cabeza)

### 4.3 Transformers y GPT

#### 4.3.1 ¿Qué es un Transformer?

**Transformer** es la arquitectura de redes neuronales que revolucionó el NLP (2017, "Attention is All You Need").

**Componentes clave:**

1. **Self-Attention**: Permite al modelo "prestar atención" a diferentes partes del texto simultáneamente

   ```
   Ejemplo: "El gato que persiguió al ratón era rápido"
   → "era rápido" se asocia con "gato", no "ratón"
   ```

2. **Multi-Head Attention**: Múltiples mecanismos de atención en paralelo

   - Cada "head" aprende diferentes tipos de relaciones

3. **Positional Encoding**: Inyecta información sobre el orden de palabras

   - Transformers procesan todo el texto en paralelo, no secuencialmente

4. **Feed-Forward Networks**: Capas densas que procesan representaciones

**Ventajas sobre RNNs/LSTMs:**

- **Paralelización**: Procesa todo el texto a la vez (no secuencial)
- **Captura de dependencias largas**: Atención directa entre tokens distantes
- **Escalabilidad**: Funciona mejor con más datos y parámetros

#### 4.3.2 GPT (Generative Pre-trained Transformer)

**GPT** es una familia de modelos basados en transformers **decoder-only** entrenados para predecir la siguiente palabra.

**Evolución:**

- **GPT-1** (2018): 117M parámetros
- **GPT-2** (2019): 1.5B parámetros
- **GPT-3** (2020): 175B parámetros
- **GPT-4** (2023): ~1T parámetros (estimado)

**Entrenamiento:**

1. **Pre-entrenamiento**:

   ```
   Tarea: Predecir la siguiente palabra
   Entrada: "El gato está en el"
   Objetivo: "sofá"
   ```

2. **Fine-tuning** (opcional):
   - RLHF (Reinforcement Learning from Human Feedback)
   - Instruction tuning (seguir instrucciones)

**Capacidades:**

- Generación de texto coherente y fluido
- Respuesta a preguntas
- Traducción
- Resumen
- Razonamiento básico
- Programación

### 4.4 Relación entre LLMs y Bases de Datos Vectoriales

**Problema fundamental de los LLMs:**

Los LLMs tienen **conocimiento estático** congelado al momento de entrenamiento. No pueden:

- Acceder a datos actualizados
- Conocer información privada/específica de empresa
- Citar fuentes verificables
- Evitar "alucinaciones" (inventar información)

**Solución: Bases de Datos Vectoriales**

Las bases vectoriales **complementan** a los LLMs proporcionando:

1. **Memoria externa**: Almacén de conocimiento actualizable
2. **Recuperación semántica**: Encuentra información relevante para el contexto
3. **Fundamentación**: Respuestas basadas en documentos reales
4. **Actualización sin reentrenamiento**: Añadir nuevos documentos sin modificar el modelo

**Flujo RAG (Retrieval-Augmented Generation):**

```
Usuario: "¿Cuál es la política de vacaciones?"

1. Query → Embedding (vector)
2. Búsqueda vectorial → Documentos relevantes
3. LLM recibe: Query + Contexto de documentos
4. LLM genera: Respuesta fundamentada en documentos

Respuesta: "Según el documento de RRHH actualizado en 2024,
           los empleados tienen 20 días de vacaciones..."
```

**Ventajas de combinar LLMs + Bases Vectoriales:**

- **Actualización en tiempo real**: Nuevos docs → nuevos embeddings → respuestas actualizadas
- **Dominio específico**: Conocimiento privado de empresa, industria
- **Verificabilidad**: Citas a documentos fuente
- **Reducción de alucinaciones**: LLM debe usar contexto proporcionado
- **Escalabilidad**: Millones de docs indexados vs. límite de contexto LLM

---

## 5. Retrieval-Augmented Generation (RAG)

### 5.1 ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** es un patrón arquitectónico que combina:

1. **Retrieval (Recuperación)**: Buscar información relevante en una base de conocimiento
2. **Augmentation (Aumento)**: Inyectar esa información como contexto
3. **Generation (Generación)**: LLM genera respuesta usando el contexto

**Analogía:**

RAG es como un estudiante en un examen de libro abierto:

- **Retrieval**: Buscar en el libro la sección relevante
- **Augmentation**: Leer esa sección
- **Generation**: Escribir la respuesta usando la información del libro

### 5.2 Arquitectura RAG Básica

```
┌─────────────────────────────────────────────────────────┐
│                    FASE DE INDEXACIÓN                    │
│  (Ejecutar una vez o cuando hay nuevos documentos)      │
└─────────────────────────────────────────────────────────┘

PDFs/Docs → [Extracción] → Chunks → [Embedding Model] → Vectores
                ↓
         [Base de Datos Vectorial]
         (Qdrant / PostgreSQL+pgvector)


┌─────────────────────────────────────────────────────────┐
│                  FASE DE CONSULTA (RUNTIME)              │
│              (Cada vez que el usuario pregunta)          │
└─────────────────────────────────────────────────────────┘

Query Usuario → [Embedding Model] → Query Vector
                      ↓
              [Búsqueda Vectorial]
              (ANN: HNSW, IVF, etc.)
                      ↓
           Top-K Documentos Relevantes
                      ↓
              [Construcción de Contexto]
                      ↓
         Prompt = Query + Contexto
                      ↓
                  [LLM]
                      ↓
               Respuesta Final
```

### 5.3 Componentes de un Sistema RAG

#### 5.3.1 Document Loader (Cargador de Documentos)

**Función**: Extraer texto de diferentes formatos.

**Soportados en este proyecto:**

- **PDFs**: PyMuPDF (enhanced), pdfplumber (fallback)
- **TXT**: Lectura directa
- **MD**: Markdown
- **DOCX**: python-docx

**Desafíos:**

- PDFs complejos (tablas, columnas, imágenes)
- OCR para PDFs escaneados
- Preservar estructura y metadatos

**En este proyecto:**

```python
# app/pdf_processing.py
class UnifiedPDFProcessor:
    def extract_enhanced(self, pdf_path):
        # PyMuPDF para extracción avanzada
        # Detecta calidad, estructura, metadatos
```

#### 5.3.2 Text Splitter (Divisor de Texto)

**Función**: Dividir documentos largos en **chunks** (fragmentos) manejables.

**¿Por qué chunking?**

- LLMs tienen **límite de contexto** (4k-128k tokens)
- Embeddings funcionan mejor con textos cortos y coherentes
- Balance: contexto suficiente vs. precisión

**Estrategias:**

1. **Fixed-size**: Chunks de tamaño fijo con overlap

   ```python
   CHUNK_TOKENS = 200  # ~150 palabras
   CHUNK_OVERLAP = 50  # ~40 palabras
   ```

2. **Semántico**: Dividir por párrafos, oraciones, secciones

3. **Jerárquico**: Parent chunks con child chunks anidados

**En este proyecto:**

```python
# scripts/ingest_config.py
CHUNK_TOKENS = 200  # Optimizado para consultas cortas
CHUNK_OVERLAP = 50  # Preservar contexto entre chunks
MIN_CHARS = 50      # Evitar chunks vacíos
```

#### 5.3.3 Embedding Model (Modelo de Embeddings)

**Función**: Convertir texto a vectores numéricos.

**Modelo usado**: `intfloat/multilingual-e5-base`

**Características:**

- **768 dimensiones**
- **Multilingüe**: Español, inglés, 50+ idiomas
- **Prefijos E5**:
  ```python
  query_emb = encode("query: " + user_query)
  doc_emb = encode("passage: " + chunk_text)
  ```

**Alternativas:**

- `Alibaba-NLP/gte-multilingual-base`
- `OpenAI text-embedding-ada-002` (API, 1536 dims)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

#### 5.3.4 Vector Store (Base de Datos Vectorial)

**Función**: Almacenar y buscar vectores eficientemente.

**En este proyecto**: Qdrant + PostgreSQL (comparación)

**Decisiones de diseño:**

- **Distancia**: Cosine (mejor para texto)
- **Índice**: HNSW (mejor balance precisión/velocidad)
- **Metadatos**: source_path, page, chunk_id, quality_score

#### 5.3.5 Retriever (Recuperador)

**Función**: Búsqueda vectorial + filtrado + reranking.

**Flujo:**

```python
1. Embedding del query
2. Búsqueda vectorial (top-k=50)
3. Filtrado por metadatos (opcional)
4. Reranking (MMR, RRF, etc.)
5. Top-k final (5-10 resultados)
```

**En este proyecto:**

```python
# app/rag.py
def search_knowledge_base(
    query, backend="qdrant", k=5, filters=None
):
    # 1. Embedding
    emb = embed_e5([query], is_query=True)[0]

    # 2. Búsqueda vectorial
    hits = BACKENDS[backend](emb, k=k, where=filters)

    # 3. Formato de resultados
    return {"results": hits, "query": query, ...}
```

#### 5.3.6 Prompt Engineering

**Función**: Construir el prompt que recibirá el LLM.

**Estructura típica:**

```python
PROMPT_TEMPLATE = """
Eres un asistente académico. Responde usando SOLO la información de los fragmentos.

Pregunta: {query}

Fragmentos relevantes:
{context}

Instrucciones:
- Respuesta clara y concisa
- Usa fechas, números exactos de los fragmentos
- Si no está en los fragmentos: "No disponible."
- Menciona fuente solo si es relevante

Respuesta:
"""
```

**Mejores prácticas:**

- **System prompt claro**: Define rol y restricciones
- **Contexto estructurado**: Formatea fragmentos legiblemente
- **Few-shot examples** (opcional): Ejemplos de respuestas buenas
- **Limitar contexto**: 1500-3000 tokens para evitar confusión

#### 5.3.7 LLM (Large Language Model)

**Función**: Generar respuesta final usando contexto.

**Opciones en este proyecto:**

- **Ollama local**: phi3:mini, gemma2:2b, llama3:8b
- **APIs externas**: OpenAI, Anthropic, Cohere

**Parámetros clave:**

```python
ollama.generate(
    model="phi3:mini",
    prompt=prompt,
    options={
        "temperature": 0.1,  # Bajo = más determinístico
        "num_predict": 400,  # Max tokens a generar
        "top_p": 0.9,        # Nucleus sampling
        "top_k": 40          # Top-k sampling
    }
)
```

### 5.4 Ventajas y Desventajas de RAG

**Ventajas:**

✅ **Actualización sin reentrenamiento**: Nuevos docs → nuevos embeddings
✅ **Dominio específico**: Conocimiento privado, especializado
✅ **Verificabilidad**: Cita fuentes documentales
✅ **Reducción de alucinaciones**: LLM fundamentado en docs reales
✅ **Control**: Puedes filtrar qué información usa el LLM
✅ **Costo-efectivo**: No requiere fine-tuning del LLM

**Desventajas:**

❌ **Calidad de recuperación**: Si retrieval falla, LLM no puede compensar
❌ **Límite de contexto**: Solo puedes pasar ~k fragmentos al LLM
❌ **Latencia**: Retrieval + LLM añade tiempo
❌ **Chunk quality**: Depende de buena estrategia de chunking
❌ **Contradictions**: Fragmentos pueden contener info contradictoria

---

## 6. Comparativa de Bases de Datos Vectoriales

### 6.1 Qdrant

**Tipo**: Base de datos vectorial nativa, open-source

**Características:**

- **Rendimiento**: Optimizado desde cero para vectores
- **Rust**: Lenguaje de alto rendimiento y seguro
- **Filtering**: Filtros avanzados en metadatos (JSONB)
- **Distancias**: Cosine, Euclidean, Dot Product
- **Índices**: HNSW (default), cuantización escalar
- **APIs**: REST, gRPC, Python client

**Ventajas:**

✅ **Muy rápido**: HNSW optimizado
✅ **Filtrado rico**: Queries complejas en metadatos
✅ **Snapshots**: Backups y versionado
✅ **Clustering**: Escalamiento horizontal
✅ **Docker-friendly**: Fácil despliegue

**Desventajas:**

❌ **No-SQL**: No hay joins, transacciones complejas
❌ **Curva de aprendizaje**: Conceptos nuevos si vienes de SQL
❌ **Ecosistema menor**: Menos herramientas que PostgreSQL

**Cuándo usar:**

- Búsqueda vectorial es la operación principal
- Necesitas máximo rendimiento
- Workload de lectura pesada
- Prototipado rápido de RAG

**En este proyecto:**

```python
# app/qdrant_backend.py
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

results = client.search(
    collection_name="course_docs_clean",
    query_vector=embedding,
    limit=k,
    query_filter={"document_type": "pdf"}
)
```

**Configuración en Docker:**

```yaml
# docker-compose.yml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
  volumes:
    - ./data/qdrant:/qdrant/storage
```

### 6.2 PostgreSQL + pgvector

**Tipo**: Extensión vectorial para PostgreSQL

**Características:**

- **Extensión**: `pgvector` añade tipo `vector` y operadores
- **SQL completo**: Joins, transacciones, vistas, triggers
- **Índices**: IVF-Flat, HNSW (desde pgvector 0.5)
- **Distancias**: L2 (`<->`), cosine (`<=>`), dot (`<#>`)
- **Integración**: Compatible con ORMs (SQLAlchemy, Prisma, etc.)

**Ventajas:**

✅ **SQL familiar**: Usa conocimiento existente
✅ **Joins**: Combina búsqueda vectorial con queries relacionales
✅ **ACID**: Transacciones, consistencia
✅ **Ecosistema maduro**: Herramientas, backups, replicación
✅ **Unified DB**: Datos relacionales + vectores en un solo lugar

**Desventajas:**

❌ **Performance**: Más lento que Qdrant (~2-3x en benchmarks)
❌ **Escalamiento**: Vertical principalmente (sharding complejo)
❌ **Overhead**: PostgreSQL no optimizado solo para vectores

**Cuándo usar:**

- Ya usas PostgreSQL
- Necesitas joins con datos relacionales
- ACID transactions son críticas
- Queriescomplejas SQL + vectorial

**En este proyecto:**

```python
# app/pgvector_backend.py
import psycopg2

conn = psycopg2.connect(
    dbname="vectordb",
    user="pguser",
    password="pgpass",
    host="localhost"
)

# Búsqueda con cosine similarity
sql = """
SELECT content, source_path, page,
       (1 - (embedding <=> %s::vector)) AS similarity
FROM docs_clean
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

cur.execute(sql, [embedding_json, embedding_json, k])
```

**Configuración en Docker:**

```yaml
# docker-compose.yml
postgres:
  image: ankane/pgvector:latest
  environment:
    POSTGRES_USER: pguser
    POSTGRES_PASSWORD: pgpass
    POSTGRES_DB: vectordb
  ports:
    - "5432:5432"
```

**Instalación de extensión:**

```sql
CREATE EXTENSION vector;

CREATE TABLE docs_clean (
    id SERIAL PRIMARY KEY,
    content TEXT,
    source_path TEXT,
    page INTEGER,
    chunk_id TEXT,
    metadata JSONB,
    embedding vector(768)  -- Dimensión del modelo
);

-- Índice HNSW para búsqueda rápida
CREATE INDEX ON docs_clean
USING hnsw (embedding vector_cosine_ops);
```

### 6.3 Otros Sistemas Populares

#### 6.3.1 Pinecone

**Tipo**: SaaS (managed cloud service)

**Ventajas:**

✅ **Fully managed**: Sin ops, auto-scaling
✅ **Performance**: Optimizado para escala masiva
✅ **Metadata filtering**: Avanzado
✅ **Serverless**: Pay-per-use

**Desventajas:**

❌ **Costo**: Puede ser caro a escala
❌ **Vendor lock-in**: Servicio propietario
❌ **Latency**: Red añade latencia vs. local

**Cuándo usar:**

- Startup que prioriza velocidad de desarrollo
- No quieres gestionar infraestructura
- Budget para SaaS

#### 6.3.2 Weaviate

**Tipo**: Vector database nativa, open-source

**Ventajas:**

✅ **GraphQL API**: Queries flexibles
✅ **Módulos**: Integraciones con LLMs, rerankers
✅ **Multimodal**: Texto, imágenes
✅ **Hybrid search**: BM25 + vectorial built-in

**Desventajas:**

❌ **Complejidad**: Muchas opciones, curva de aprendizaje
❌ **Resource usage**: Usa más recursos que Qdrant

**Cuándo usar:**

- Multimodal (texto + imágenes)
- Necesitas hybrid search out-of-the-box

#### 6.3.3 Milvus

**Tipo**: Vector database nativa, open-source (CNCF)

**Ventajas:**

✅ **Escalamiento masivo**: Diseñado para billones de vectores
✅ **GPU acceleration**: Construcción de índices en GPU
✅ **Cloud-native**: Kubernetes, microservicios
✅ **Múltiples índices**: IVF, HNSW, DiskANN

**Desventajas:**

❌ **Complejidad operacional**: Requiere expertise
❌ **Overhead**: Para datasets pequeños es overkill

**Cuándo usar:**

- Escala masiva (>100M vectores)
- Tienes equipo DevOps dedicado
- Infraestructura Kubernetes existente

### 6.4 Comparativa: Qdrant vs. PostgreSQL+pgvector

**En este proyecto comparamos:**

| Aspecto                   | Qdrant                     | PostgreSQL+pgvector             |
| ------------------------- | -------------------------- | ------------------------------- |
| **Velocidad (búsqueda)**  | ⚡⚡⚡ ~5-10ms             | ⚡⚡ ~15-30ms                   |
| **Velocidad (inserción)** | ⚡⚡ Batch rápido          | ⚡ Transaccional más lento      |
| **Filtrado metadatos**    | ✅ JSONB avanzado          | ✅ JSONB + SQL                  |
| **Joins con datos**       | ❌ No soportado            | ✅ SQL completo                 |
| **Escalamiento**          | ✅ Horizontal (clustering) | ⚠️ Vertical (sharding complejo) |
| **Curva aprendizaje**     | 🟡 Nueva API               | 🟢 SQL familiar                 |
| **Operaciones**           | 🟡 Servicio separado       | 🟢 Unificado con DB existente   |
| **Costo infraestructura** | 💰 Medio                   | 💰 Bajo (si ya usas PG)         |
| **Madurez ecosistema**    | 🟡 Creciendo               | 🟢 Muy maduro                   |

**Recomendación:**

- **Qdrant**: Si búsqueda vectorial es tu operación principal y necesitas máxima velocidad
- **PostgreSQL+pgvector**: Si ya usas PostgreSQL, necesitas joins, o quieres simplificar ops

---

## 7. Arquitectura del Proyecto

### 7.1 Visión General

Este proyecto implementa un **sistema RAG completo** con comparación de backends vectoriales (Qdrant vs. PostgreSQL+pgvector).

**Stack tecnológico:**

```
┌─────────────────────────────────────────────────────────┐
│                    CAPA DE APLICACIÓN                    │
│  FastAPI (Python 3.10+) + Jinja2 Templates + HTML/CSS   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     CAPA DE NEGOCIO                      │
│  • RAG Engine (rag.py)                                   │
│  • Advanced RAG (advanced_rag.py)                        │
│  • Orchestrated RAG (orchestrated_rag.py)                │
│  • Reranking (rerank.py)                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────┬──────────────────────────────────┐
│   BACKENDS VECTORIALES                                   │
│  Qdrant              │  PostgreSQL + pgvector            │
│  (qdrant_backend.py) │  (pgvector_backend.py)            │
└──────────────────────┴──────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                         │
│  • Embeddings: E5-multilingual (768 dims)                │
│  • LLM: Ollama (phi3, gemma2, llama3)                    │
│  • Documentos: PDFs, TXT, MD (data/raw/)                 │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Estructura de Directorios

```
base-de-datos-avanzadas/
├── app/                          # Aplicación principal
│   ├── main.py                   # FastAPI routes y endpoints
│   ├── rag.py                    # RAG básico
│   ├── advanced_rag.py           # Técnicas avanzadas
│   ├── orchestrated_rag.py       # Pipeline inteligente
│   ├── rerank.py                 # MMR, context building
│   ├── qdrant_backend.py         # Cliente Qdrant
│   ├── pgvector_backend.py       # Cliente PostgreSQL
│   ├── settings.py               # Configuración
│   └── templates/                # Templates HTML
│       ├── home.html             # Página principal
│       ├── ai_response.html      # Respuestas AI
│       └── ...
│
├── scripts/                      # Pipelines de procesamiento
│   ├── main_pipeline.py          # Pipeline unificado principal
│   ├── pdf_processing.py         # Extracción de PDFs
│   ├── chunker.py                # Segmentación de texto
│   ├── embedding_database.py     # Generación embeddings + upsert
│   ├── query_embed.py            # Query embeddings
│   └── ingest_config.py          # Configuración del pipeline
│
├── data/
│   ├── raw/                      # PDFs originales (input)
│   ├── clean/                    # JSONLs procesados (intermedio)
│   ├── qdrant/                   # Datos de Qdrant (volumen)
│   └── postgres/                 # Init scripts SQL
│
├── docker-compose.yml            # Servicios (Qdrant, PostgreSQL, Ollama)
├── Dockerfile                    # Imagen de la aplicación
├── requirements.txt              # Dependencias Python
└── README.md                     # Documentación de uso
```

### 7.3 Componentes Principales

#### 7.3.1 FastAPI Application (app/main.py)

**Endpoints principales:**

```python
# BÁSICOS
GET  /                           # Página principal
GET  /ask?q=...&backend=qdrant  # Búsqueda vectorial
GET  /ai?q=...&model=phi3       # RAG con LLM
GET  /compare?q=...             # Comparación Qdrant vs PG

# AVANZADOS
GET  /advanced/multi-query       # Multi-Query Rephrasing
GET  /advanced/decompose         # Query Decomposition
GET  /advanced/hyde              # Hypothetical Document Embeddings
GET  /advanced/hybrid            # Hybrid Search (BM25 + Dense)
GET  /advanced/iterative         # Iterative Retrieval
GET  /orchestrated               # Pipeline automático inteligente

# PIPELINE MANAGEMENT
GET  /pipeline                   # Dashboard de pipeline
POST /pipeline/upload            # Subir documentos
POST /pipeline/run               # Ejecutar procesamiento
POST /pipeline/clear             # Limpiar bases de datos
GET  /pipeline/stats             # Estadísticas

# DEMOS EDUCATIVAS
GET  /manual/embed?text=...      # Ver embedding de texto
GET  /manual/search?q=...        # Búsqueda paso a paso
GET  /demo/similarity            # Cálculo de similitud
```

#### 7.3.2 RAG Engine (app/rag.py)

**Responsabilidades:**

1. **Búsqueda semántica**:

   ```python
   def search_knowledge_base(query, backend, k, filters):
       # 1. Query expansion (sinónimos, reformulación)
       # 2. Embedding del query
       # 3. Búsqueda vectorial
       # 4. Enriquecimiento con metadatos
       # 5. Retornar resultados formateados
   ```

2. **Generación de respuestas**:
   ```python
   def generate_llm_answer(query, backend, k, model):
       # 1. Recuperar top-k documentos
       # 2. MMR reranking (diversidad)
       # 3. Construir contexto
       # 4. Prompt engineering
       # 5. LLM generation
       # 6. Retornar respuesta + citas
   ```

**Características:**

- **Query expansion**: Expansión automática con sinónimos
- **Metadata extraction**: Extrae capítulos, secciones, temas
- **MMR reranking**: Balance relevancia/diversidad
- **Citation formatting**: Referencias limpias a documentos

#### 7.3.3 Advanced RAG (app/advanced_rag.py)

**Técnicas implementadas:**

1. **Multi-Query Rephrasing**: Genera variaciones del query
2. **Query Decomposition**: Divide queries complejas
3. **HyDE**: Genera documento hipotético
4. **Hybrid Search**: BM25 + búsqueda densa con RRF
5. **Iterative Retrieval**: Múltiples rondas de búsqueda

**Cada técnica retorna:**

```python
{
    "method": "Multi-Query Search",
    "query_variations": [...],
    "results": [...],
    "timing": {...},
    "ai_answer": "..."
}
```

#### 7.3.4 Orchestrated RAG (app/orchestrated_rag.py)

**Sistema inteligente** que automáticamente:

1. **Detecta señales** del query (corto, abstracto, compuesto)
2. **Aplica técnicas** apropiadas condicionalmente
3. **Verifica answerability** (¿podemos responder?)
4. **Early exit** cuando información es suficiente
5. **Budget control** (máximo de llamadas a retrieval)

**Algoritmo:**

```
Phase 0: Preflight
  - Normalización
  - Detección de señales (is_short, is_abstract, has_conjunctions)

Phase 1: Baseline (SIEMPRE)
  - Hybrid search (BM25 + Dense) con RRF
  - Answerability check
  - Early exit si confidence > 0.7

Phase 2: Conditional Enrichment
  - Si score_gap < 0.05 → Multi-Query
  - Si is_abstract → HyDE
  - Si has_conjunctions → Decomposition
  - Re-check answerability después de cada técnica

Phase 3: Iterative Refinement
  - Genera followup queries para llenar gaps
  - Máximo 2-3 rondas
  - Stop cuando answerable

Final: Reranking & Generation
  - MMR reranking para diversidad
  - LLM generation con contexto completo
```

#### 7.3.5 Backends (qdrant_backend.py, pgvector_backend.py)

**Interfaz unificada:**

```python
def search_backend(
    query_emb,              # Vector de consulta
    k=5,                    # Top-k resultados
    where=None,             # Filtros metadata
    collection_suffix=None  # Para algoritmos específicos
) -> List[Dict]:
    # Retorna:
    # [
    #   {
    #     "score": 0.85,
    #     "content": "texto...",
    #     "path": "doc.pdf",
    #     "page": 5,
    #     "chunk_id": "chunk_2",
    #     "metadata": {...}
    #   },
    #   ...
    # ]
```

**Filtros soportados:**

- `document_type`: "pdf", "txt", "md"
- `section`: "objetivos", "cronograma", "evaluacion"
- `topic`: "nosql", "vectorial", "sql"
- `page`: número de página (PDFs)
- `contains`: texto debe contener string

### 7.4 Flujo de Datos

#### 7.4.1 Indexación (Offline)

```
1. Usuario coloca PDFs en data/raw/

2. Ejecuta: python scripts/main_pipeline.py

3. Pipeline:
   PDFs → [PDF Processor] → Texto limpio → [Chunker] → Chunks
         ↓
   Chunks → [Embedding Model (E5)] → Vectores (768 dims)
         ↓
   Vectores + Metadata → [Qdrant] → Collection "course_docs_clean"
                      → [PostgreSQL] → Table "docs_clean"

4. Resultado:
   - data/clean/*.jsonl (cache intermedio)
   - Qdrant collection con N vectores indexados (HNSW)
   - PostgreSQL table con N filas + índice HNSW
```

#### 7.4.2 Consulta (Runtime)

```
1. Usuario pregunta: "¿Qué son las bases de datos vectoriales?"

2. FastAPI recibe request:
   GET /ai?q=¿Qué son las bases de datos vectoriales?&backend=qdrant

3. RAG Pipeline:
   Query → [Query Expansion] → "bases datos vectoriales vector embedding"
        ↓
   Query expandido → [E5 Encoder] → Query Vector (768 dims)
        ↓
   Query Vector → [Qdrant Search HNSW] → Top-50 candidatos
        ↓
   Candidatos → [MMR Reranking] → Top-5 diversos
        ↓
   Top-5 → [Context Builder] → Contexto formateado
        ↓
   Prompt = System + Query + Contexto → [LLM (Ollama)] → Respuesta
        ↓
   Respuesta + Citas → [FastAPI Response] → Usuario

4. Latencia típica:
   - Query embedding: ~50ms
   - Búsqueda vectorial: ~10ms (Qdrant HNSW)
   - MMR reranking: ~20ms
   - LLM generation: ~2-5s (depende del modelo)
   - Total: ~2-5 segundos
```

### 7.5 Configuración Multi-Algoritmo

El proyecto soporta **16 combinaciones** de backends y configuraciones:

```python
Backends (2): Qdrant, PostgreSQL
Métricas (4): Cosine, Euclidean, Dot Product, Manhattan
Algoritmos (2): HNSW, IVF-Flat

Total: 2 × 4 × 2 = 16 configuraciones

Ejemplo:
- qdrant_cosine_hnsw
- pgvector_euclidean_ivfflat
- qdrant_dot_hnsw
- ...
```

**Uso:**

```bash
# Procesar con todas las combinaciones
python scripts/main_pipeline.py --all-combinations

# Buscar usando configuración específica
curl "http://localhost:8080/ask?q=vectores&backend=qdrant&distance_metric=cosine&index_algorithm=hnsw"
```

---

## 8. Pipeline de Procesamiento

### 8.1 Pipeline Básico (main_pipeline.py)

**Propósito**: Procesar documentos desde PDFs crudos hasta vectores indexados en ambas bases de datos.

**Comando:**

```bash
python scripts/main_pipeline.py [OPTIONS]
```

**Opciones principales:**

```bash
--clear              # Limpiar bases de datos antes de procesar
--force              # Forzar reprocesamiento (ignorar cache)
--basic              # Usar extracción básica (sin enhanced)
--memory-safe        # Modo seguro de memoria (default: ON)
--batch-size N       # Tamaño de batch para embeddings
--large-docs         # Optimizar para documentos grandes
--distance-metric M  # cosine (default), euclidean, dot_product, manhattan
--index-algorithm A  # hnsw (default), ivfflat, scalar_quantization, exact
--parallel           # Procesamiento paralelo de PDFs
--workers N          # Número de workers paralelos (default: auto)
--stats              # Mostrar estadísticas
--config             # Mostrar configuración
```

**Ejemplos:**

```bash
# Uso básico (recomendado)
python scripts/main_pipeline.py --clear

# Procesamiento paralelo (8 cores)
python scripts/main_pipeline.py --parallel --workers 8

# Documentos grandes (libros médicos)
python scripts/main_pipeline.py --large-docs --batch-size 256

# Solo configuración euclidiana + IVF
python scripts/main_pipeline.py --distance-metric euclidean --index-algorithm ivfflat --single-combination
```

### 8.2 Fases del Pipeline

#### Fase 1: Extracción de PDFs

**Módulo**: `scripts/pdf_processing.py`

**Proceso:**

1. **Detección de calidad**:

   ```python
   # PyMuPDF analiza la estructura del PDF
   quality_score = analyze_pdf_quality(pdf_path)
   # > 0.8: Alta calidad (texto digital)
   # 0.5-0.8: Media (texto + imágenes)
   # < 0.5: Baja (escaneo, OCR necesario)
   ```

2. **Extracción inteligente**:

   ```python
   if quality_score > 0.8:
       text = extract_with_pymupdf(pdf_path)  # Rápido
   else:
       text = extract_with_pdfplumber(pdf_path)  # Robusto
   ```

3. **Metadatos**:

   ```python
   metadata = {
       "quality_score": 0.92,
       "extractor": "pymupdf",
       "total_pages": 45,
       "extraction_time_ms": 1200,
       "has_images": True,
       "has_tables": False
   }
   ```

4. **Output**: `data/clean/documento.jsonl`
   ```jsonl
   {"page": 1, "content": "...", "metadata": {...}}
   {"page": 2, "content": "...", "metadata": {...}}
   ```

#### Fase 2: Chunking (Segmentación)

**Módulo**: `scripts/chunker.py`

**Proceso:**

1. **Configuración**:

   ```python
   CHUNK_TOKENS = 200  # ~150 palabras
   CHUNK_OVERLAP = 50  # ~40 palabras
   MIN_CHARS = 50      # Mínimo caracteres por chunk
   ```

2. **Tokenización**:

   ```python
   # Usa tokenizer de E5 para precisión
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

   tokens = tokenizer.encode(text)
   ```

3. **Sliding window**:

   ```
   Chunk 1: tokens[0:200]
   Chunk 2: tokens[150:350]  (overlap de 50 desde posición 150)
   Chunk 3: tokens[300:500]
   ...
   ```

4. **Enriquecimiento**:

   ```python
   chunk = {
       "content": decoded_text,
       "chunk_id": f"chunk_{idx}",
       "page": page_num,
       "source_path": pdf_path,
       "metadata": {
           "chunk_length": len(decoded_text),
           "token_count": len(tokens),
           "section": detect_section(decoded_text),  # "objetivos", etc.
           "quality_score": parent_quality
       }
   }
   ```

5. **Output**: `data/clean/documento.chunks.jsonl`
   ```jsonl
   {"content": "...", "chunk_id": "chunk_0", "page": 1, ...}
   {"content": "...", "chunk_id": "chunk_1", "page": 1, ...}
   ```

#### Fase 3: Embedding Generation

**Módulo**: `scripts/embedding_database.py`

**Proceso:**

1. **Carga del modelo**:

   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('intfloat/multilingual-e5-base')
   model.to('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Preparación con prefijos**:

   ```python
   # E5 requiere prefijos específicos
   texts = [f"passage: {chunk['content']}" for chunk in chunks]
   ```

3. **Batch encoding**:

   ```python
   # Procesar en batches para eficiencia
   batch_size = 512  # Configurado para hardware potente

   for i in range(0, len(texts), batch_size):
       batch = texts[i:i+batch_size]
       embeddings = model.encode(
           batch,
           batch_size=batch_size,
           show_progress_bar=True,
           convert_to_numpy=True
       )
       # embeddings.shape = (batch_size, 768)
   ```

4. **Resultado**:
   ```python
   {
       "content": "Las bases de datos vectoriales...",
       "embedding": [0.023, -0.154, 0.782, ..., 0.234],  # 768 floats
       "source_path": "data/raw/tema1.pdf",
       "page": 5,
       "chunk_id": "chunk_12",
       "metadata": {...}
   }
   ```

#### Fase 4: Database Upsert

**Módulo**: `scripts/embedding_database.py`

**Proceso para Qdrant:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. Crear colección
client.create_collection(
    collection_name="course_docs_clean_cosine_hnsw",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    ),
    hnsw_config={
        "m": 16,              # Conexiones por nodo
        "ef_construct": 200   # Calidad de construcción
    }
)

# 2. Insertar vectores en batch
points = [
    PointStruct(
        id=idx,
        vector=chunk["embedding"],
        payload={
            "content": chunk["content"],
            "source_path": chunk["source_path"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "metadata": chunk["metadata"]
        }
    )
    for idx, chunk in enumerate(chunks)
]

client.upsert(
    collection_name="course_docs_clean_cosine_hnsw",
    points=points,
    wait=True
)
```

**Proceso para PostgreSQL:**

```python
import psycopg2

# 1. Crear tabla
conn = psycopg2.connect(dsn)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS docs_clean_cosine_hnsw (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        source_path TEXT,
        page INTEGER,
        chunk_id TEXT,
        metadata JSONB,
        embedding vector(768)
    );
""")

# 2. Crear índice HNSW
cur.execute("""
    CREATE INDEX IF NOT EXISTS docs_clean_cosine_hnsw_idx
    ON docs_clean_cosine_hnsw
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
""")

# 3. Insertar vectores
for chunk in chunks:
    cur.execute("""
        INSERT INTO docs_clean_cosine_hnsw
        (content, source_path, page, chunk_id, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        chunk["content"],
        chunk["source_path"],
        chunk["page"],
        chunk["chunk_id"],
        json.dumps(chunk["metadata"]),
        json.dumps(chunk["embedding"])  # Convertir a JSON
    ))

conn.commit()
```

### 8.3 Pipeline con Procesamiento Paralelo

**Ventaja**: Acelera el procesamiento de múltiples PDFs usando todos los cores disponibles.

**Activación:**

```bash
python scripts/main_pipeline.py --parallel --workers 8
```

**Arquitectura:**

```python
from concurrent.futures import ProcessPoolExecutor

def process_single_pdf(pdf_path):
    # 1. Extracción
    # 2. Chunking
    # 3. Guardar JSONL
    return success

def process_pdf_parallel(pdf_files, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit todos los PDFs
        futures = {
            executor.submit(process_single_pdf, pdf): pdf
            for pdf in pdf_files
        }

        # Recolectar resultados conforme terminan
        for future in as_completed(futures):
            pdf = futures[future]
            try:
                result = future.result()
                print(f"✅ {pdf.name} procesado")
            except Exception as e:
                print(f"❌ {pdf.name} falló: {e}")
```

**Performance:**

```
Dataset: 10 PDFs, ~500 páginas total

Secuencial:      ~8 minutos
Paralelo (8w):   ~2 minutos (4x speedup)
Paralelo (16w):  ~1.5 minutos (5.3x speedup)
```

**Trade-offs:**

✅ **Ventajas**:

- Velocidad significativa en multi-PDF
- Uso completo de CPU multi-core

❌ **Desventajas**:

- Mayor uso de memoria (múltiples procesos)
- Overhead de coordinación (< 5%)
- No beneficia si hay un solo PDF

### 8.4 Monitoreo y Estadísticas

**Comando:**

```bash
python scripts/main_pipeline.py --stats
```

**Output:**

```
📊 Pipeline Statistics
================================================

📁 Documents:
   Raw PDFs:        12 files
   Processed:       12 files (100%)
   Failed:          0 files

📄 Chunks:
   Total chunks:    2,847
   Avg per doc:     237
   Size range:      150-200 tokens

🔢 Embeddings:
   Model:           intfloat/multilingual-e5-base
   Dimensions:      768
   Total vectors:   2,847

💾 Qdrant:
   Collections:     16 (all combinations)
   Total vectors:   45,552 (16 × 2,847)
   Index type:      HNSW (M=16, ef=200)
   Memory usage:    ~1.2 GB

💾 PostgreSQL:
   Tables:          16 (all combinations)
   Total rows:      45,552
   Index type:      HNSW
   Disk usage:      ~890 MB

⏱️  Processing Time:
   PDF extraction:  2.3 minutes
   Chunking:        0.8 minutes
   Embeddings:      4.2 minutes
   Upsert (both):   1.5 minutes
   Total:           8.8 minutes
```

---

## 9. Técnicas Avanzadas de RAG

### 9.1 Multi-Query Rephrasing

**Problema**: Queries del usuario pueden ser ambiguos o usar vocabulario diferente al de los documentos.

**Solución**: Generar múltiples variaciones del query y fusionar resultados.

**Algoritmo:**

```
1. LLM genera N variaciones del query
   Original: "¿Qué son bases de datos vectoriales?"
   Var 1: "Explica las bases de datos vectoriales"
   Var 2: "Definición de vector databases"
   Var 3: "Cómo funcionan los almacenes vectoriales"

2. Búsqueda vectorial para cada variación
   Results1 = search(var1) → Top-50
   Results2 = search(var2) → Top-50
   Results3 = search(var3) → Top-50

3. Reciprocal Rank Fusion (RRF)
   Para cada documento d:
     score(d) = Σ(1 / (k + rank_i(d)))

   Ejemplo:
   Doc A: rank1=1, rank2=3, rank3=2
   score(A) = 1/(60+1) + 1/(60+3) + 1/(60+2) = 0.0481

4. Re-rank y retornar Top-K
```

**Ventajas:**

✅ Mayor recall (más documentos relevantes capturados)
✅ Robusto a diferentes formas de expresar la pregunta
✅ Combina perspectivas múltiples

**Desventajas:**

❌ Más llamadas a LLM y búsqueda vectorial (latencia)
❌ Mayor costo computacional

**Cuándo usar:**

- Queries cortos o ambiguos
- Score gap bajo entre resultados (poca confianza)
- Cuando precisión es crítica

**Endpoint:**

```bash
curl "http://localhost:8080/advanced/multi-query?q=bases vectoriales&num_variations=3"
```

### 9.2 Query Decomposition

**Problema**: Queries complejos contienen múltiples sub-preguntas que deberían responderse por separado.

**Solución**: Descomponer en sub-queries, buscar cada uno, sintetizar respuesta final.

**Algoritmo:**

```
1. LLM descompone query complejo
   Original: "¿Cuáles son las ventajas y desventajas de Qdrant vs PostgreSQL?"

   Sub-Q1: "¿Cuáles son las ventajas de Qdrant?"
   Sub-Q2: "¿Cuáles son las desventajas de Qdrant?"
   Sub-Q3: "¿Cuáles son las ventajas de PostgreSQL?"
   Sub-Q4: "¿Cuáles son las desventajas de PostgreSQL?"

2. Búsqueda independiente para cada sub-query
   Results1 = search(sub_q1)
   Results2 = search(sub_q2)
   ...

3. Síntesis (opcional)
   LLM combina todas las respuestas parciales:
   "Basado en la información recuperada:

    Ventajas de Qdrant:
    - [de Results1]

    Desventajas de Qdrant:
    - [de Results2]
    ..."
```

**Ventajas:**

✅ Maneja queries multi-parte
✅ Respuestas más estructuradas
✅ Mejor cobertura de temas complejos

**Desventajas:**

❌ Overhead de descomposición (LLM call)
❌ Múltiples búsquedas (latencia)

**Cuándo usar:**

- Queries con "y", "vs", enumeraciones
- Preguntas comparativas
- Solicitudes multi-aspecto

**Endpoint:**

```bash
curl "http://localhost:8080/advanced/decompose?q=ventajas y desventajas de Qdrant&synthesize=true"
```

### 9.3 HyDE (Hypothetical Document Embeddings)

**Problema**: Query y documentos pueden estar en "espacios semánticos" diferentes (domain gap).

**Solución**: LLM genera un documento hipotético que respondería la pregunta, buscar con su embedding.

**Algoritmo:**

```
1. LLM genera documento hipotético
   Query: "¿Qué es pgvector?"

   Hypothetical Doc:
   "pgvector es una extensión de PostgreSQL que añade soporte
    para vectores y búsqueda de similitud. Permite almacenar
    embeddings directamente en PostgreSQL y realizar búsquedas
    ANN usando índices IVF o HNSW..."

2. Embedding del documento hipotético
   hyp_emb = encode("passage: " + hypothetical_doc)
   # Nota: usa prefijo "passage", no "query"

3. Búsqueda vectorial con hyp_emb
   results = search(hyp_emb, k=50)

4. Opcionalmente: Generar respuesta final
   LLM responde usando los documentos recuperados
```

**Intuición:**

Los documentos están escritos en cierto estilo/vocabulario. Un documento hipotético generado por el LLM está más cerca de ese estilo que un query directo del usuario.

**Ventajas:**

✅ Excelente para queries abstractos
✅ Cierra el domain gap
✅ Mejora recall en dominios técnicos

**Desventajas:**

❌ Depende de calidad del LLM
❌ LLM puede "alucinar" en doc hipotético
❌ Mayor latencia (LLM + búsqueda)

**Cuándo usar:**

- Queries muy cortos o abstractos
- Domain gap conocido (e.g., lenguaje casual → técnico)
- Cuando búsqueda directa falla

**Endpoint:**

```bash
curl "http://localhost:8080/advanced/hyde?q=pgvector"
```

### 9.4 Hybrid Search (BM25 + Dense)

**Problema**: Búsqueda vectorial pura ignora matches léxicos exactos (keywords).

**Solución**: Combinar búsqueda keyword (BM25) con búsqueda densa (vectorial) usando RRF.

**Algoritmo:**

```
1. Búsqueda Semántica (Dense)
   emb = encode("query: " + query)
   semantic_results = vector_search(emb, k=50)

2. Búsqueda Keyword (BM25)
   # BM25: TF-IDF mejorado, estándar en full-text search
   keyword_results = bm25_search(query, corpus, k=50)

   BM25(d, q) = Σ(IDF(qi) × (f(qi,d) × (k1+1)) / (f(qi,d) + k1 × (1-b+b×|d|/avgdl)))

   Donde:
   - f(qi, d): frecuencia del término qi en documento d
   - |d|: longitud del documento
   - avgdl: longitud promedio de documentos
   - k1, b: parámetros de calibración

3. Reciprocal Rank Fusion
   Fusionar ambos resultados ponderados:

   score_final(d) = λ × score_semantic(d) + (1-λ) × score_bm25(d)

   Usando RRF:
   score(d) = λ × (1/(60+rank_semantic(d))) + (1-λ) × (1/(60+rank_bm25(d)))
```

**Ventajas:**

✅ Combina lo mejor de ambos mundos
✅ Robusto a queries con términos técnicos específicos
✅ Mejora precision y recall

**Desventajas:**

❌ Requiere implementación BM25 (o ElasticSearch integración)
❌ Más complejo de configurar

**Parámetros:**

- **semantic_weight (λ)**: Balance entre semántico y keyword
  - 0.7 (default): 70% semántico, 30% keyword
  - 0.5: Balance equitativo
  - 0.9: Casi puro semántico

**Cuándo usar:**

- Queries con nombres propios, IDs, términos técnicos exactos
- Cuando sabes que usuarios buscan keywords específicos
- **Recomendación general** como baseline

**Endpoint:**

```bash
curl "http://localhost:8080/advanced/hybrid?q=PostgreSQL pgvector&semantic_weight=0.7"
```

### 9.5 Iterative Retrieval (Multi-Round)

**Problema**: Respuesta completa requiere información de múltiples fuentes que la búsqueda inicial no captura.

**Solución**: Iterativamente generar nuevas búsquedas basadas en gaps de información identificados.

**Algoritmo:**

```
Round 1:
  1. Búsqueda inicial con query original
  2. LLM evalúa: ¿Es suficiente para responder?
  3. Si NO: Identificar qué información falta

Round 2:
  1. Generar followup query específico para los gaps
     Ejemplo: Si falta "ventajas de Qdrant"
     Followup: "ventajas y beneficios de usar Qdrant"
  2. Búsqueda con followup query
  3. Acumular resultados
  4. Re-evaluar answerability

Round N:
  Repetir hasta:
  - Información suficiente (answerable=True)
  - Max rounds alcanzado (default: 3)
  - No se pueden generar más followups
```

**Ejemplo real:**

```
Query: "¿Cómo configurar HNSW en Qdrant y PostgreSQL?"

Round 1:
  Search: "configurar HNSW Qdrant PostgreSQL"
  Results: Docs sobre HNSW general
  Answerability: PARCIAL (falta detalles de configuración)
  Missing: "parámetros específicos de HNSW"

Round 2:
  Followup: "parámetros M ef_construction HNSW"
  Results: Docs con valores de parámetros
  Answerability: SÍ
  → STOP, generar respuesta
```

**Ventajas:**

✅ Maneja queries multi-hop complejos
✅ Adaptativo (para cuando encuentra gaps)
✅ Mejora cobertura de información

**Desventajas:**

❌ Múltiples llamadas LLM + búsqueda (latencia alta)
❌ Riesgo de "rabbit holes" (buscar info irrelevante)

**Cuándo usar:**

- Queries muy complejos que requieren múltiples documentos
- Cuando answerability check muestra gaps claros
- Tienes budget de tiempo/costo para múltiples rondas

**Endpoint:**

```bash
curl "http://localhost:8080/advanced/iterative?q=configurar HNSW&max_rounds=3"
```

### 9.6 Orchestrated RAG (Sistema Inteligente)

**Concepto**: Pipeline que **automáticamente decide** qué técnicas aplicar basado en características del query.

**Ventajas principales:**

✅ **Plug-and-play**: Usuario no necesita decidir qué técnica usar
✅ **Eficiente**: Early exit cuando info es suficiente
✅ **Budget control**: Limita llamadas a retrieval/LLM
✅ **Adaptativo**: Se ajusta a complejidad del query

**Ejemplo de ejecución:**

```
Query: "introducción vectores"

Phase 0: Preflight
  Signals: is_short=True, is_abstract=True, query_length=2

Phase 1: Baseline
  Hybrid search → 50 results
  Answerability: PARCIAL (confidence=0.65)
  → Continue

Phase 2: Enrichment
  Signal: is_short=True → Apply Multi-Query
  Generate 2 variations
  Search each → 100 more results
  Answerability: SÍ (confidence=0.82)
  → Early exit (no need for HyDE, Decomposition)

Final:
  MMR reranking → Top-10
  LLM generates answer
  Return results with execution trace
```

**Endpoint:**

```bash
curl "http://localhost:8080/orchestrated?q=bases de datos vectoriales"
```

### 9.7 Comparativa de Técnicas

| Técnica           | Latencia  | Complejidad    | Recall      | Precision   | Mejor Para          |
| ----------------- | --------- | -------------- | ----------- | ----------- | ------------------- |
| **Básico**        | 🟢 ~50ms  | 🟢 Baja        | ⚠️ Media    | ⚠️ Media    | Queries simples     |
| **Multi-Query**   | 🟡 ~200ms | 🟡 Media       | ✅ Alta     | ✅ Alta     | Queries ambiguos    |
| **Decomposition** | 🟡 ~300ms | 🟡 Media       | ✅ Alta     | ✅ Alta     | Queries complejos   |
| **HyDE**          | 🟡 ~250ms | 🟡 Media       | ✅ Muy alta | ⚠️ Media    | Queries abstractos  |
| **Hybrid**        | 🟢 ~80ms  | 🟢 Baja        | ✅ Alta     | ✅ Alta     | **General purpose** |
| **Iterative**     | 🔴 ~1s    | 🔴 Alta        | ✅ Muy alta | ✅ Alta     | Queries multi-hop   |
| **Orchestrated**  | 🟡 ~300ms | 🟢 Baja (auto) | ✅ Muy alta | ✅ Muy alta | **Producción**      |

**Recomendaciones:**

1. **Prototipado**: RAG básico
2. **Producción general**: Orchestrated RAG o Hybrid Search
3. **Casos específicos**: Elegir técnica apropiada basado en tabla

---

## 10. Optimización y Métricas

### 10.1 Métricas de Evaluación

#### 10.1.1 Retrieval Metrics

**Recall@K**: ¿Qué porcentaje de documentos relevantes están en top-K?

```
Recall@5 = (Relevantes en Top-5) / (Total de Relevantes)

Ejemplo:
- Relevantes totales: 10
- Relevantes en Top-5: 7
- Recall@5 = 7/10 = 0.7 (70%)
```

**Precision@K**: ¿Qué porcentaje de top-K son relevantes?

```
Precision@5 = (Relevantes en Top-5) / 5

Ejemplo:
- Top-5 retornados: 5
- Relevantes: 4
- Precision@5 = 4/5 = 0.8 (80%)
```

**MRR (Mean Reciprocal Rank)**: ¿Qué tan arriba está el primer resultado relevante?

```
MRR = 1 / (Rank del primer relevante)

Ejemplo:
- Primer relevante en posición 3
- MRR = 1/3 = 0.333
```

**NDCG (Normalized Discounted Cumulative Gain)**: Métrica que considera relevancia graduada

```
NDCG@K = DCG@K / IDCG@K

DCG = Σ(relevance_i / log2(i+1))

Ejemplo con relevance scores [3, 2, 3, 0, 1]:
DCG@5 = 3/log2(2) + 2/log2(3) + 3/log2(4) + 0/log2(5) + 1/log2(6)
      = 3.0 + 1.26 + 1.5 + 0 + 0.39 = 6.15
```

#### 10.1.2 Generation Metrics

**Faithfulness**: ¿La respuesta se basa en el contexto?

```
Método automático: Verificar si claims en respuesta tienen support en contexto
Método manual: Evaluadores humanos califican 0-5
```

**Answer Relevance**: ¿La respuesta responde la pregunta?

```
Embedding similarity entre pregunta y respuesta
score = cosine(embed(question), embed(answer))
```

**Context Relevance**: ¿El contexto recuperado es relevante?

```
Método: LLM judge
"De los siguientes fragmentos, ¿cuántos ayudan a responder la pregunta?"
score = (fragmentos útiles) / (fragmentos totales)
```

**BLEU/ROUGE**: Similitud con respuestas de referencia (gold standard)

```
BLEU: Precision de n-gramas
ROUGE: Recall de n-gramas

Requiere respuestas gold manualmente creadas
```

#### 10.1.3 Latency Metrics

```python
# Desglose típico para este proyecto:

Query Embedding:        50 ms   (5%)
Vector Search:          10 ms   (1%)   [Qdrant HNSW]
MMR Reranking:          20 ms   (2%)
Context Building:       5 ms    (0.5%)
LLM Generation:         2000 ms (92%)  [phi3:mini]
-------------------------------------------------
Total:                  2085 ms (100%)

# LLM domina la latencia → optimizar primero
```

### 10.2 Optimización de Búsqueda Vectorial

#### 10.2.1 Tuning de HNSW

**Parámetro M** (conexiones por nodo):

```python
M = 16  # Default, buen balance
M = 8   # Menor memoria, más rápido insert, menor recall
M = 32  # Mayor recall, más memoria, insert lento

Recomendación:
- Datasets pequeños (<100k): M=16
- Datasets grandes (>1M): M=16-24
- Máxima precisión: M=32-48
```

**Parámetro ef_construction** (construcción del índice):

```python
ef_construction = 200  # Default
ef_construction = 100  # Construcción rápida, menor recall
ef_construction = 400  # Mejor recall, construcción lenta

Recomendación:
- Prototipado: 100-200
- Producción: 200-400
```

**Parámetro ef_search** (búsqueda):

```python
ef_search = 100  # Default
ef_search = 50   # Rápido, menor recall
ef_search = 200  # Mejor recall, más lento

# Este es el que ajustas en runtime para trade-off velocidad/precisión

Recomendación:
- Latency crítica: 50-100
- Balance: 100-200
- Máxima precisión: 200-500
```

**Benchmark ejemplo:**

```
Dataset: 1M vectores (768 dims)
K = 10

Config          | QPS   | Recall@10 | P99 Latency
----------------|-------|-----------|-------------
M=16, ef=100    | 2500  | 0.94      | 12 ms
M=16, ef=200    | 1800  | 0.97      | 15 ms
M=32, ef=200    | 1600  | 0.98      | 18 ms
M=32, ef=400    | 1200  | 0.99      | 25 ms
```

#### 10.2.2 Batch Processing

**Embedding generation:**

```python
# ❌ Malo: Una a la vez
for text in texts:
    emb = model.encode(text)  # 50ms × 1000 = 50 segundos

# ✅ Bueno: Batch
batch_size = 512
embeddings = model.encode(texts, batch_size=batch_size)  # ~5 segundos

# Speedup: 10x
```

**Database upsert:**

```python
# ❌ Malo: Inserción individual
for point in points:
    client.upsert(collection, [point])  # N network calls

# ✅ Bueno: Batch upsert
batch_size = 1000
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    client.upsert(collection, batch)

# Speedup: 50-100x
```

#### 10.2.3 Caching

**Query cache:**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_search(query_hash, k, backend):
    # Cache basado en hash del query
    # Evita re-embedding y búsqueda para queries repetidos
    results = perform_search(query, k, backend)
    return results

# Uso:
query_hash = hashlib.md5(query.encode()).hexdigest()
results = cached_search(query_hash, k, backend)

# Hit rate típico: 20-40% en producción
```

**Embedding cache:**

```python
# Guardar embeddings de chunks en disco
# Evita re-calcular si contenido no cambió

import pickle

def load_or_compute_embeddings(chunks):
    cache_file = "embeddings_cache.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    embeddings = model.encode(chunks)

    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings
```

### 10.3 Optimización de LLM

#### 10.3.1 Selección de Modelo

**Trade-off: Tamaño vs. Calidad**

```
Modelo           | Parámetros | Latencia | Calidad | VRAM   | Uso
-----------------|------------|----------|---------|--------|------------------
phi3:mini        | 3.8B       | ~2s      | Buena   | 4 GB   | Prototipado, CPU
gemma2:2b        | 2B         | ~1.5s    | Media   | 3 GB   | Edge devices
llama3:8b        | 8B         | ~4s      | Excelente| 8 GB  | Producción GPU
llama3:70b       | 70B        | ~20s     | SOTA    | 48 GB  | Máxima calidad
gpt-4 (API)      | ~1.7T      | ~3s      | SOTA    | N/A    | Cloud, $$$
```

**Recomendación para este proyecto:**

- **CPU only**: phi3:mini (mejor balance)
- **GPU (8GB)**: llama3:8b
- **GPU (16GB+)**: llama3:70b o mixtral:8x7b
- **Producción cloud**: GPT-4 API o Claude Opus

#### 10.3.2 Parámetros de Generación

**Temperature:**

```python
temperature = 0.1  # Determinístico, factual (recomendado para RAG)
temperature = 0.7  # Creativo, variado
temperature = 1.0  # Muy variado, puede ser incoherente

Para RAG: 0.1-0.3 (queremos respuestas factuales)
```

**Top-P (Nucleus Sampling):**

```python
top_p = 0.9  # Default, bueno para RAG
top_p = 1.0  # Sin filtrado
top_p = 0.5  # Más conservador

Para RAG: 0.8-0.95
```

**Max Tokens:**

```python
num_predict = 400   # Respuestas concisas (recomendado)
num_predict = 800   # Respuestas detalladas
num_predict = 2000  # Artículos largos

Para RAG: 300-500 (suficiente para respuesta + citas)
```

#### 10.3.3 Prompt Optimization

**Técnicas:**

1. **Few-shot examples**:

   ```python
   prompt = """
   Responde basándote solo en el contexto.

   Ejemplo:
   Pregunta: ¿Qué es pgvector?
   Contexto: pgvector es una extensión...
   Respuesta: pgvector es una extensión de PostgreSQL que permite...

   Ahora tu turno:
   Pregunta: {query}
   Contexto: {context}
   Respuesta:
   """
   ```

2. **Chain-of-Thought**:

   ```python
   prompt = """
   Analiza paso a paso:
   1. ¿Qué pregunta el usuario?
   2. ¿Qué información relevante hay en el contexto?
   3. ¿Cómo combinar esa información?
   4. Respuesta final clara y concisa
   """
   ```

3. **Constrained output**:

   ```python
   prompt = """
   Responde en este formato EXACTO:

   RESPUESTA: [respuesta en 2-3 oraciones]
   FUENTES: [lista de documentos citados]
   CONFIANZA: [Alta/Media/Baja]
   """
   ```

### 10.4 Monitoreo en Producción

**Métricas clave a trackear:**

```python
# 1. Latencia (percentiles)
latency_p50 = 500 ms
latency_p95 = 1200 ms
latency_p99 = 2500 ms

# 2. Throughput
queries_per_second = 50 QPS
concurrent_requests = 10

# 3. Cache hit rate
cache_hit_rate = 0.35  # 35% de queries son cache hits

# 4. Error rate
error_rate = 0.02  # 2% de requests fallan

# 5. User satisfaction (si hay feedback)
thumbs_up_rate = 0.78  # 78% respuestas positivas

# 6. Answerability rate
answered_queries = 0.92  # 92% de queries son respondibles

# 7. Costos (si API externa)
cost_per_query = $0.005
monthly_cost = $15000
```

**Herramientas:**

- **Prometheus + Grafana**: Métricas de tiempo real
- **ELK Stack**: Logs centralizados
- **Sentry**: Error tracking
- **LangSmith / Weights & Biases**: LLM observability

---

## 11. Despliegue en la Nube

### 11.1 Arquitectura Cloud-Native

**Stack recomendado:**

```
┌─────────────────────────────────────────────────────────┐
│                    LOAD BALANCER                         │
│  (AWS ALB / GCP Load Balancer / Cloudflare)             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  APPLICATION TIER                        │
│  • FastAPI containers (ECS / Cloud Run / K8s)           │
│  • Auto-scaling (2-10 instances)                        │
│  • Health checks, rolling updates                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────┬──────────────────────────────────┐
│   VECTOR DATABASES                                       │
│  Qdrant Cloud        │  AWS RDS PostgreSQL + pgvector   │
│  (managed)           │  (managed)                        │
└──────────────────────┴──────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  EMBEDDING & LLM                         │
│  • Embedding: Hugging Face Inference API                │
│  • LLM: OpenAI API / Anthropic API / Self-hosted        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    STORAGE & MONITORING                  │
│  • S3/GCS: Raw PDFs, backups                            │
│  • CloudWatch/Stackdriver: Logs, metrics                │
│  • Sentry: Error tracking                               │
└─────────────────────────────────────────────────────────┘
```

### 11.2 Opciones por Proveedor

#### 11.2.1 AWS Deployment

**Opción 1: Serverless (Simple)**

```yaml
# AWS CDK / Terraform

# API Gateway + Lambda
API Gateway
→ Lambda Function (FastAPI via Mangum)
→ Qdrant Cloud API
→ RDS PostgreSQL (pgvector)
→ Bedrock (LLM) o OpenAI API
# Ventajas:
# - Zero ops, auto-scaling
# - Pay-per-request
# - Fácil setup

# Desventajas:
# - Cold starts (~3s primer request)
# - Timeouts (15 min max Lambda)
```

**Opción 2: ECS Fargate (Balanceado)**

```yaml
# ECS Cluster
- Service: rag-api
  - Task Definition:
      - Container: fastapi-app
        - Image: your-ecr-repo/rag-api:latest
        - CPU: 2 vCPU
        - Memory: 4 GB
        - Ports: 8080
  - Scaling:
      - Min: 2 instances
      - Max: 10 instances
      - Target CPU: 70%

# ALB (Application Load Balancer)
- Health check: /health
- SSL/TLS certificate
- WAF enabled

# RDS PostgreSQL
- Instance: db.r6g.xlarge (8GB RAM)
- Storage: 500 GB SSD
- Multi-AZ: Yes
- Backups: Daily

# Qdrant: Self-hosted en EC2 o Qdrant Cloud

# Costos estimados:
# - ECS Fargate: ~$150/month (2 instances)
# - RDS: ~$200/month
# - ALB: ~$20/month
# - Qdrant EC2: ~$100/month
# - Total: ~$470/month
```

**Opción 3: EKS (Kubernetes - Producción)**

```yaml
# Amazon EKS
- Node Group: 3 nodes (m5.xlarge)
- Deployments:
    - rag-api (HPA: 2-10 replicas)
    - qdrant-statefulset (3 replicas)
- Ingress: NGINX Ingress Controller
- Monitoring: Prometheus + Grafana
- Logging: FluentD → CloudWatch
# Ventajas:
# - Máximo control
# - Multi-cloud portable
# - Avanzado: service mesh, canary deployments

# Desventajas:
# - Complejidad operacional
# - Costo más alto
# - Requiere expertise Kubernetes
```

#### 11.2.2 GCP Deployment

**Opción Recomendada: Cloud Run + Cloud SQL**

```yaml
# Cloud Run (serverless containers)
gcloud run deploy rag-api \
  --image gcr.io/PROJECT/rag-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars "POSTGRES_HOST=CLOUD_SQL_IP,QDRANT_HOST=qdrant.example.com"

# Cloud SQL (PostgreSQL + pgvector)
- Instance: db-n1-standard-4 (4 vCPU, 15GB RAM)
- Version: PostgreSQL 15
- Extensions: pgvector
- HA: Regional
- Backups: Automated daily

# Qdrant: Qdrant Cloud o GKE self-hosted

# Cloud Storage: Para PDFs y backups

# Costos estimados:
# - Cloud Run: ~$100/month (pay-per-use)
# - Cloud SQL: ~$250/month
# - Qdrant Cloud: ~$50/month (starter)
# - Total: ~$400/month
```

#### 11.2.3 Azure Deployment

**Opción: Container Apps + Azure Database**

```yaml
# Azure Container Apps
- App: rag-api
  - Image: youracr.azurecr.io/rag-api:latest
  - Replicas: 1-10 (autoscaling)
  - Resources: 2 CPU, 4 GB RAM

# Azure Database for PostgreSQL
- Tier: General Purpose
- Compute: 4 vCores
- Storage: 512 GB
- Extensions: pgvector (requiere Flexible Server)

# Qdrant: Self-hosted en AKS o VM

# Application Gateway: Load balancing + WAF

# Costos estimados:
# - Container Apps: ~$120/month
# - PostgreSQL: ~$280/month
# - Total: ~$400/month
```

### 11.3 Managed Services vs. Self-Hosted

#### Qdrant

**Qdrant Cloud (Managed):**

```
Ventajas:
✅ Zero ops (backups, updates, monitoring incluidos)
✅ Auto-scaling
✅ 99.9% SLA
✅ Multi-region replication

Desventajas:
❌ Costo: ~$50-500/month dependiendo de scale
❌ Menos control sobre configuración

Pricing:
- Free tier: 1 cluster, 1GB vectores
- Starter: $50/month (10M vectores)
- Pro: $200+/month (100M+ vectores)
```

**Self-Hosted:**

```
Ventajas:
✅ Control total
✅ Menor costo en escala grande
✅ Data residency control

Desventajas:
❌ Requiere ops expertise
❌ Backups, monitoring, scaling manual
❌ Responsabilidad de uptime

Costos:
- EC2 m5.xlarge: ~$140/month
- + EBS storage: ~$50/month
- = ~$190/month
```

#### PostgreSQL + pgvector

**Managed (RDS, Cloud SQL, Azure Database):**

```
Ventajas:
✅ Backups automáticos, point-in-time recovery
✅ Multi-AZ alta disponibilidad
✅ Patches automáticos
✅ Read replicas fáciles

Desventajas:
❌ Costo premium (~2x vs. self-hosted)
❌ Menos control sobre configuración PostgreSQL

Pricing (AWS RDS ejemplo):
- db.r6g.xlarge (4 vCPU, 32GB): ~$350/month
- + Storage 500GB: ~$115/month
- = ~$465/month
```

**Self-Hosted:**

```
Ventajas:
✅ Menor costo
✅ Control total (parámetros, extensiones)

Desventajas:
❌ Requiere DBA expertise
❌ Backups manuales
❌ HA setup complejo

Costos:
- EC2 r6g.xlarge: ~$180/month
- + EBS gp3 500GB: ~$40/month
- = ~$220/month
```

### 11.4 CI/CD Pipeline

**GitHub Actions ejemplo:**

```yaml
# .github/workflows/deploy.yml
name: Deploy RAG API

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Build and push Docker image
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker build -t rag-api:$GITHUB_SHA .
          docker tag rag-api:$GITHUB_SHA $ECR_REGISTRY/rag-api:latest
          docker push $ECR_REGISTRY/rag-api:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster rag-cluster \
            --service rag-api \
            --force-new-deployment
```

### 11.5 Costos Comparativos

**Escenario: Startup (10k queries/day)**

| Componente     | AWS              | GCP              | Azure               | Self-Hosted    |
| -------------- | ---------------- | ---------------- | ------------------- | -------------- |
| **Compute**    | ECS $150         | Cloud Run $100   | Container Apps $120 | EC2 $80        |
| **Vector DB**  | Qdrant Cloud $50 | Qdrant Cloud $50 | Qdrant Cloud $50    | EC2 $140       |
| **PostgreSQL** | RDS $200         | Cloud SQL $250   | Azure DB $280       | EC2 $180       |
| **LLM**        | Bedrock $200     | Vertex AI $250   | Azure OpenAI $200   | Ollama $0      |
| **Otros**      | ALB $20, S3 $10  | LB $20, GCS $10  | Gateway $20         | -              |
| **TOTAL**      | **$630/month**   | **$680/month**   | **$670/month**      | **$400/month** |

**Escenario: Empresa (1M queries/day)**

| Componente     | Managed (AWS)          | Self-Hosted (AWS)        |
| -------------- | ---------------------- | ------------------------ |
| **Compute**    | ECS $600 (8 instances) | EKS $800 (nodes)         |
| **Vector DB**  | Qdrant Cloud $500      | EC2 cluster $600         |
| **PostgreSQL** | RDS $800 (larger)      | RDS $800 (same)          |
| **LLM**        | OpenAI API $5000       | Self-hosted GPUs $2000   |
| **CDN**        | CloudFront $100        | CloudFront $100          |
| **Monitoring** | CloudWatch $50         | Prometheus $0 (ops cost) |
| **TOTAL**      | **$7050/month**        | **$4300/month**          |

**Recomendación:**

- **< 100k queries/day**: Managed services (simplicidad > costo)
- **100k-1M queries/day**: Híbrido (compute managed, databases self-hosted)
- **> 1M queries/day**: Mayormente self-hosted con expertise dedicado

---

## 12. Aplicaciones Empresariales

### 12.1 Casos de Uso por Industria

#### 12.1.1 Servicio al Cliente

**Problema**: Agentes necesitan buscar en manuales/FAQs extensos para resolver tickets.

**Solución RAG:**

```
Base de conocimiento: Manuales de productos, FAQs, tickets históricos
Query: "Cliente no puede instalar software en Windows 11"

RAG retorna:
- Sección manual: Requisitos Windows 11
- FAQ: Problemas comunes instalación
- Ticket similar: Solución verificada

Agente: Responde en 1 minuto vs. 10 minutos búsqueda manual
```

**Métricas de impacto:**

- ⬇️ Tiempo promedio de resolución: -60%
- ⬆️ Satisfacción del cliente: +25%
- ⬇️ Escalaciones a nivel 2: -40%

**Implementación:**

```python
# Integración con Zendesk/Salesforce
@app.post("/support/search")
def support_search(ticket_id: str, query: str):
    # Filtrar por tipo de producto del ticket
    results = search_knowledge_base(
        query,
        filters={"product": ticket.product, "status": "verified"}
    )

    # Generar respuesta sugerida
    suggested_response = generate_llm_answer(query, results)

    return {
        "suggested_response": suggested_response,
        "knowledge_articles": results,
        "confidence": 0.85
    }
```

#### 12.1.2 Legal / Compliance

**Problema**: Abogados necesitan buscar precedentes, cláusulas específicas en miles de contratos.

**Solución RAG:**

```
Base de conocimiento: Contratos, casos legales, regulaciones
Query: "Cláusulas de indemnización en contratos de proveedores de software"

RAG retorna:
- Cláusulas relevantes de 15 contratos existentes
- Precedentes legales relacionados
- Regulaciones aplicables

Abogado: Prepara documento en 2 horas vs. 2 días
```

**Características especiales:**

- **Citation critical**: Cada claim debe tener referencia exacta (página, sección)
- **High precision required**: Falsos positivos pueden ser legalmente riesgosos
- **Confidentiality**: Deployment on-premise o VPC privado

**Implementación:**

```python
# RAG con citas explícitas
def legal_search(query: str, jurisdiction: str):
    results = search_knowledge_base(
        query,
        filters={"jurisdiction": jurisdiction, "status": "active"},
        k=20  # Mayor recall para legal
    )

    # Generar respuesta con citas numeradas
    response = generate_llm_answer(
        query,
        results,
        prompt_template=LEGAL_PROMPT  # Enfatiza citas exactas
    )

    return {
        "answer": response,
        "citations": extract_citations(response, results),
        "confidence": calculate_confidence(results),
        "disclaimer": "Revisar con abogado antes de usar"
    }
```

#### 12.1.3 Healthcare / Investigación Médica

**Problema**: Médicos necesitan acceso rápido a literatura médica, guidelines clínicos.

**Solución RAG:**

```
Base de conocimiento: Papers médicos, guidelines, historiales clínicos
Query: "Tratamiento hipertensión en pacientes con diabetes tipo 2"

RAG retorna:
- Guidelines actualizados (ADA, AHA)
- Estudios clínicos recientes
- Protocolos del hospital

Médico: Toma decisión informada en minutos
```

**Consideraciones:**

- **Compliance**: HIPAA, GDPR para datos de pacientes
- **Evidence-based**: Priorizar papers peer-reviewed, guidelines oficiales
- **Multimodal**: Textos + imágenes médicas

**Implementación:**

```python
# RAG médico con ranking por evidencia
def medical_search(query: str, patient_context: dict):
    # Búsqueda con filtros por nivel de evidencia
    results = search_knowledge_base(
        query,
        filters={
            "source_type": ["guideline", "rct", "meta_analysis"],
            "evidence_level": ["A", "B"],  # Alta calidad
            "publication_date": {"gte": "2020-01-01"}
        },
        k=10
    )

    # Personalizar con contexto del paciente
    answer = generate_llm_answer(
        query,
        results,
        system_prompt=MEDICAL_PROMPT,
        additional_context=patient_context
    )

    return {
        "clinical_recommendation": answer,
        "evidence_sources": results,
        "evidence_quality": assess_evidence_quality(results),
        "disclaimer": "Not a substitute for professional medical advice"
    }
```

#### 12.1.4 E-commerce / Retail

**Problema**: Clientes tienen preguntas sobre productos, políticas, comparaciones.

**Solución RAG:**

```
Base de conocimiento: Catálogos, especificaciones, reviews, políticas
Query: "¿Qué laptop es mejor para edición de video bajo $1000?"

RAG retorna:
- Especificaciones de laptops relevantes
- Reviews y ratings
- Comparativas de rendimiento

Cliente: Encuentra producto ideal sin agente humano
```

**Implementación:**

```python
# RAG e-commerce con datos estructurados
def product_search(query: str, user_profile: dict):
    # Búsqueda híbrida (texto + atributos)
    results = search_knowledge_base(
        query,
        filters={
            "category": "laptops",
            "price": {"lte": 1000},
            "in_stock": True
        },
        k=10
    )

    # Enriquecer con datos estructurados (DB relacional)
    products = enrich_with_product_data(results)

    # Generar recomendación personalizada
    recommendation = generate_llm_answer(
        query,
        products,
        additional_context={
            "user_preferences": user_profile,
            "previous_purchases": get_purchase_history(user_profile["id"])
        }
    )

    return {
        "recommendation": recommendation,
        "products": products,
        "personalization_score": 0.82
    }
```

### 12.2 ROI y Justificación de Inversión

**Costos iniciales:**

```
Desarrollo:
- Developer time (3 meses): $30,000
- Infrastructure setup: $5,000
- Total: $35,000

Costos operacionales (mensuales):
- Cloud infrastructure: $500-2000
- LLM API costs: $200-5000 (depende de volumen)
- Monitoring/maintenance: $500
- Total mensual: $1,200-7,500
```

**Beneficios (ejemplo servicio al cliente):**

```
Métricas antes de RAG:
- 100 tickets/día
- 15 min promedio de resolución
- 10 agentes × $20/hora
- Costo: $50,000/mes

Métricas después de RAG:
- 100 tickets/día
- 6 min promedio de resolución (-60%)
- 6 agentes necesarios (4 menos)
- Costo: $30,000/mes agentes + $2,000 RAG = $32,000/mes

Ahorro: $18,000/mes = $216,000/año

ROI: ($216,000 - $35,000 - $2,000×12) / $35,000 = 511% primer año
```

### 12.3 Consideraciones de Implementación

#### 12.3.1 Data Privacy y Seguridad

**Preocupaciones:**

1. **Datos sensibles en embeddings**: Embeddings pueden "memorizar" datos sensibles
2. **LLM leakage**: LLM puede exponer información de un usuario a otro
3. **Compliance**: GDPR, CCPA, HIPAA, etc.

**Mitigaciones:**

```python
# 1. PII Detection y Anonymization
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

def anonymize_before_embedding(text):
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text, language='es', entities=["PERSON", "EMAIL", "PHONE"])

    anonymizer = AnonymizerEngine()
    anonymized = anonymizer.anonymize(text, results)
    return anonymized.text

# 2. Access Control en búsquedas
def search_with_access_control(query, user_id):
    # Filtrar solo documentos que el usuario tiene permiso de ver
    user_permissions = get_user_permissions(user_id)

    results = search_knowledge_base(
        query,
        filters={"document_id": {"in": user_permissions.allowed_docs}}
    )
    return results

# 3. Audit logging
def log_query(user_id, query, results):
    audit_log.write({
        "timestamp": datetime.now(),
        "user_id": user_id,
        "query": query,
        "num_results": len(results),
        "accessed_documents": [r["path"] for r in results]
    })
```

#### 12.3.2 Multilingual Support

**Challenge**: Soportar múltiples idiomas sin degradar calidad.

**Solución:**

```python
# 1. Usar modelo multilingüe
model = SentenceTransformer('intfloat/multilingual-e5-base')
# Soporta 50+ idiomas con embeddings compartidos

# 2. Language detection
from langdetect import detect

def multilingual_search(query):
    language = detect(query)

    # Opción A: Traducir query a inglés (si corpus en inglés)
    if language != 'en':
        query_en = translate(query, target='en')
        results = search_knowledge_base(query_en)

    # Opción B: Búsqueda directa (modelo multilingüe)
    else:
        results = search_knowledge_base(query)

    # Traducir respuesta al idioma original
    answer = generate_llm_answer(query, results)
    if language != 'es':
        answer = translate(answer, target=language)

    return answer
```

#### 12.3.3 Continuous Learning

**Problema**: Documentos y conocimiento evolucionan constantemente.

**Solución: Pipeline de actualización incremental**

```python
# 1. Webhook cuando documento nuevo/actualizado
@app.post("/webhook/document-updated")
def document_updated(doc_id: str, doc_url: str):
    # Download documento
    content = download_document(doc_url)

    # Procesar
    chunks = chunk_document(content)
    embeddings = embed_chunks(chunks)

    # Actualizar bases de datos
    # Opción A: Eliminar viejo + insertar nuevo
    delete_document_chunks(doc_id)
    upsert_chunks(embeddings)

    # Opción B: Upsert con mismo ID (overwrite)
    upsert_chunks(embeddings, upsert=True)

    return {"status": "updated"}

# 2. Scheduled re-indexing
@app.post("/admin/reindex")
def scheduled_reindex():
    # Daily/weekly: Re-procesar documentos modificados
    modified_docs = get_modified_documents(since=last_index_time)

    for doc in modified_docs:
        process_document(doc)

    return {"reindexed": len(modified_docs)}
```

---

## 13. Recomendaciones y Mejores Prácticas

### 13.1 Para Estudiantes y Aprendizaje

**Progresión recomendada:**

1. **Semana 1: Fundamentos**

   - Entender embeddings (ejecutar `/manual/embed`)
   - Calcular similitudes manualmente
   - Probar búsqueda básica (`/ask`)

2. **Semana 2: RAG Básico**

   - Implementar RAG simple (query → búsqueda → LLM)
   - Experimentar con diferentes modelos embedding
   - Medir recall y precision

3. **Semana 3: Técnicas Avanzadas**

   - Probar Multi-Query, HyDE, Hybrid Search
   - Comparar resultados de cada técnica
   - Entender trade-offs

4. **Semana 4: Optimización**

   - Tuning de HNSW (M, ef_construction)
   - Benchmarking de diferentes configuraciones
   - Análisis de latencia

5. **Proyecto Final**
   - Sistema RAG completo en dominio específico
   - Comparativa Qdrant vs PostgreSQL
   - Presentación de resultados

**Recursos de práctica:**

```bash
# 1. Dataset pequeño para experimentos
curl http://localhost:8080/demo/pipeline?storage_type=both

# 2. Comparar backends
curl http://localhost:8080/compare?q=tu_query

# 3. Visualizar embeddings (PCA/t-SNE)
curl http://localhost:8080/demo/similarity

# 4. Probar técnicas avanzadas
curl http://localhost:8080/advanced/multi-query?q=tu_query
curl http://localhost:8080/advanced/hyde?q=tu_query
```

### 13.2 Para Desarrolladores

**Mejores prácticas de código:**

```python
# 1. ✅ Usar type hints
from typing import List, Dict, Optional

def search_documents(
    query: str,
    k: int = 5,
    filters: Optional[Dict] = None
) -> List[Dict]:
    pass

# 2. ✅ Manejo de errores robusto
try:
    results = search_backend(query)
except VectorDBConnectionError as e:
    logger.error(f"DB connection failed: {e}")
    # Fallback: cache, alternate backend
    results = search_cache(query)
except Exception as e:
    logger.exception("Unexpected error in search")
    raise HTTPException(status_code=500, detail="Search failed")

# 3. ✅ Logging estructurado
import structlog

logger = structlog.get_logger()
logger.info(
    "search_completed",
    query=query,
    backend=backend,
    results_count=len(results),
    latency_ms=latency,
    user_id=user_id
)

# 4. ✅ Testing
import pytest

def test_search_returns_relevant_results():
    query = "bases de datos vectoriales"
    results = search_knowledge_base(query, k=5)

    assert len(results) == 5
    assert all("vectorial" in r["content"].lower() or "vector" in r["content"].lower()
               for r in results[:3])  # Top 3 deben ser relevantes

# 5. ✅ Configuration management
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str
    postgres_dsn: str
    embedding_model: str

    class Config:
        env_file = ".env"

settings = Settings()
```

**Patrones de diseño útiles:**

```python
# 1. Strategy Pattern para backends
class VectorBackend(ABC):
    @abstractmethod
    def search(self, query_vector, k):
        pass

class QdrantBackend(VectorBackend):
    def search(self, query_vector, k):
        return self.client.search(...)

class PGVectorBackend(VectorBackend):
    def search(self, query_vector, k):
        return self.conn.execute(...)

# 2. Factory Pattern
def get_backend(backend_type: str) -> VectorBackend:
    if backend_type == "qdrant":
        return QdrantBackend()
    elif backend_type == "pgvector":
        return PGVectorBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_type}")

# 3. Repository Pattern para abstraer acceso a datos
class DocumentRepository:
    def __init__(self, backend: VectorBackend):
        self.backend = backend

    def search(self, query, k, filters):
        # Business logic
        emb = self.embed(query)
        results = self.backend.search(emb, k)
        return self.format_results(results)
```

### 13.3 Para Arquitectos de Sistemas

**Decisiones arquitectónicas clave:**

| Decisión       | Opción A             | Opción B                 | Recomendación                              |
| -------------- | -------------------- | ------------------------ | ------------------------------------------ |
| **Vector DB**  | Qdrant               | PostgreSQL               | Qdrant para velocidad, PG para integración |
| **Embedding**  | API externa (OpenAI) | Self-hosted (E5)         | Self-hosted para control y costo           |
| **LLM**        | API (GPT-4)          | Self-hosted (Llama)      | API para calidad, self-hosted para costo   |
| **Deployment** | Serverless           | Kubernetes               | Serverless si < 100k queries/day           |
| **Caching**    | Redis                | In-memory                | Redis para multi-instance                  |
| **Monitoring** | Managed (Datadog)    | Self-hosted (Prometheus) | Managed para empezar, migrar a self-hosted |

**Checklist de producción:**

- [ ] **Seguridad**

  - [ ] HTTPS/TLS en todos los endpoints
  - [ ] API key authentication
  - [ ] Rate limiting (e.g., 100 req/min por IP)
  - [ ] Input validation y sanitization
  - [ ] PII detection/anonymization

- [ ] **Reliability**

  - [ ] Health checks configurados
  - [ ] Circuit breakers para dependencias externas
  - [ ] Retry logic con exponential backoff
  - [ ] Graceful degradation (fallbacks)
  - [ ] Multi-AZ deployment

- [ ] **Observability**

  - [ ] Structured logging
  - [ ] Metrics (latency, error rate, throughput)
  - [ ] Distributed tracing (Jaeger/Zipkin)
  - [ ] Alerting (PagerDuty/OpsGenie)

- [ ] **Performance**

  - [ ] Caching strategy implementado
  - [ ] Connection pooling configurado
  - [ ] Query optimization (índices, batch)
  - [ ] Load testing completado (target: 1000 req/min)

- [ ] **Data Management**
  - [ ] Backups automatizados (diarios + PITR)
  - [ ] Disaster recovery plan documentado
  - [ ] Data retention policy definida
  - [ ] GDPR compliance (derecho al olvido)

### 13.4 Recomendación Final: ¿Qué Base de Datos Elegir?

**Para este proyecto académico:**

✅ **Usar ambas** (Qdrant + PostgreSQL) para:

- Comparar rendimiento objetivamente
- Entender trade-offs prácticos
- Aprender dos paradigmas diferentes

**Para producción:**

**Elegir Qdrant si:**

- ✅ Búsqueda vectorial es la operación principal (>80% de queries)
- ✅ Velocidad es crítica (P95 latency < 20ms)
- ✅ Escalamiento horizontal es necesario
- ✅ Empezando proyecto greenfield (sin DB legacy)
- ✅ Dataset grande (>10M vectores)

**Elegir PostgreSQL+pgvector si:**

- ✅ Ya usas PostgreSQL para datos relacionales
- ✅ Necesitas joins entre datos vectoriales y relacionales
- ✅ ACID transactions son requeridas
- ✅ Quieres simplificar stack (una DB menos)
- ✅ Equipo tiene expertise en PostgreSQL
- ✅ Dataset mediano (<5M vectores)

**Híbrido (ambos):**

- ✅ Dataset masivo (>100M vectores): Qdrant para vectores, PG para metadata
- ✅ Multi-tenancy: Qdrant por tenant, PG para billing/auth
- ✅ Migración: PG inicial, migrar a Qdrant al escalar

**Regla general:**

```
if performance_critical and vector_workload_dominant:
    use Qdrant
elif existing_postgresql_ecosystem and need_sql_integration:
    use PostgreSQL + pgvector
else:
    start_with = PostgreSQL  # Más familiar, menos ops
    migrate_to = Qdrant      # Cuando escales
```

---

## Conclusión

Este documento cubre la teoría completa de bases de datos vectoriales y RAG aplicada al proyecto 1CA217. Hemos explorado:

1. **Fundamentos**: Qué son las bases vectoriales y por qué existen
2. **Embeddings**: Cómo representar semántica como vectores numéricos
3. **ANN**: Algoritmos para búsqueda eficiente en alta dimensionalidad
4. **IA Generativa**: LLMs, GPT, y su relación con bases vectoriales
5. **RAG**: Arquitectura para respuestas fundamentadas en documentos
6. **Comparativas**: Qdrant vs PostgreSQL en detalle
7. **Arquitectura**: Cómo está construido este proyecto
8. **Pipelines**: Procesamiento desde PDFs hasta respuestas
9. **Técnicas avanzadas**: Multi-Query, HyDE, Hybrid, Iterative, Orchestrated
10. **Optimización**: Métricas, tuning, performance
11. **Cloud**: Deployment en AWS/GCP/Azure
12. **Aplicaciones**: Casos de uso reales en industria
13. **Recomendaciones**: Mejores prácticas y decisiones arquitectónicas

**Próximos pasos sugeridos:**

1. 📖 **Leer este documento** sección por sección
2. 🧪 **Experimentar** con los endpoints del proyecto
3. 📊 **Comparar** resultados de Qdrant vs PostgreSQL
4. 🔬 **Medir** impacto de diferentes técnicas avanzadas
5. 📝 **Documentar** tus hallazgos para la presentación

**Recursos adicionales:**

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [E5 Model Paper](https://arxiv.org/abs/2212.03533)
- [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)

---

_Documento creado para el curso 1CA217 - Sistemas de Base de Datos Avanzadas_
_Universidad de Costa Rica - 2025_
