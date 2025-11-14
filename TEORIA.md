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

## 6. Estrategias Avanzadas de RAG

Esta sección explora en profundidad las cinco estrategias avanzadas de Retrieval-Augmented Generation (RAG) implementadas en este proyecto. Cada técnica aborda limitaciones específicas del RAG básico y ha demostrado mejoras significativas en escenarios particulares.

### 6.1 Contexto: Limitaciones del RAG Básico

El RAG básico (búsqueda vectorial simple + LLM) tiene limitaciones conocidas:

1. **Vocabulary Mismatch**: El usuario y los documentos pueden usar términos diferentes para el mismo concepto
2. **Query Ambiguity**: Consultas cortas o ambiguas pueden no capturar la intención real
3. **Query Complexity**: Preguntas multi-parte requieren información de múltiples fuentes
4. **Semantic Gap**: Diferencia entre cómo se formula una pregunta vs. cómo se escribe la respuesta
5. **Lexical Blindness**: Búsqueda puramente semántica puede perder coincidencias de keywords exactas

Las estrategias avanzadas surgieron de investigación académica y práctica industrial para abordar estas limitaciones específicas.

---

### 6.2 Multi-Query Rephrasing (Reformulación Multi-Consulta)

#### 6.2.1 Fundamento Teórico

**Problema central**: Una sola formulación del query puede no capturar todas las formas en que la información relevante está expresada en los documentos.

**Hipótesis**: Si generamos múltiples reformulaciones del query y fusionamos sus resultados, aumentamos la probabilidad de recuperar todos los documentos relevantes (mejor **recall**).

**Origen**: Inspirado en técnicas de Query Expansion clásicas de Information Retrieval, adaptadas para la era de LLMs.

**Referencias académicas:**

- Ma et al. (2023) - "Query Rewriting for Retrieval-Augmented Large Language Models" ([arXiv:2305.14283](https://arxiv.org/abs/2305.14283))
- Wang et al. (2023) - "Query2doc: Query Expansion with Large Language Models" ([arXiv:2303.07678](https://arxiv.org/abs/2303.07678))

#### 6.2.2 Flujo de Ejecución

```
┌─────────────────────────────────────────────────────────────┐
│ Fase 1: Generación de Variaciones                           │
└─────────────────────────────────────────────────────────────┘

Query Original: "¿Qué es pgvector?"

    ↓ [LLM: Genera variaciones manteniendo intención semántica]

Variación 1: "Explica la extensión pgvector de PostgreSQL"
Variación 2: "¿Cómo funciona el soporte de vectores en Postgres?"
Variación 3: "Definición y características de pgvector"

┌─────────────────────────────────────────────────────────────┐
│ Fase 2: Búsqueda Independiente                              │
└─────────────────────────────────────────────────────────────┘

Cada variación → Embedding → Búsqueda Vectorial

Query Original    → Top-50 resultados (R0)
Variación 1       → Top-50 resultados (R1)
Variación 2       → Top-50 resultados (R2)
Variación 3       → Top-50 resultados (R3)

Total: 4 listas de resultados (con posible overlap)

┌─────────────────────────────────────────────────────────────┐
│ Fase 3: Fusión con RRF (Reciprocal Rank Fusion)            │
└─────────────────────────────────────────────────────────────┘

Para cada documento d que aparece en al menos una lista:

    RRF_score(d) = Σ [ 1 / (k + rank_i(d)) ]

    Donde:
    - k = constante (típicamente 60)
    - rank_i(d) = posición de d en lista i (1-indexed)
    - Si d no aparece en lista i, su contribución es 0

Ejemplo concreto:

Documento A:
- R0: posición 1  → 1/(60+1)  = 0.0164
- R1: posición 5  → 1/(60+5)  = 0.0154
- R2: posición 2  → 1/(60+2)  = 0.0161
- R3: no aparece  → 0
- RRF_score(A) = 0.0479

Documento B:
- R0: posición 10 → 1/(60+10) = 0.0143
- R1: no aparece  → 0
- R2: posición 8  → 1/(60+8)  = 0.0147
- R3: posición 15 → 1/(60+15) = 0.0133
- RRF_score(B) = 0.0423

Documento A rankeado más alto (aparece en más listas y posiciones altas)

┌─────────────────────────────────────────────────────────────┐
│ Fase 4: Re-ranking y Selección Final                        │
└─────────────────────────────────────────────────────────────┘

Ordenar todos los documentos únicos por RRF_score descendente
Retornar Top-K (típicamente 5-10)
```

#### 6.2.3 Concepto Clave: RRF (Reciprocal Rank Fusion)

**RRF** es un algoritmo de fusión de rankings que no requiere normalización de scores entre diferentes sistemas.

**Ventajas sobre otras fusiones:**

- **CombSUM/CombMNZ**: Requieren scores normalizados (problemático con diferentes backends)
- **Borda Count**: Sensible a la longitud de las listas
- **RRF**: Invariante a la escala de scores, robusto, teóricamente fundamentado

**Paper original**: Cormack et al. (2009) - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (SIGIR)

**Por qué funciona:**

1. **Penaliza posiciones bajas**: 1/61 >> 1/160
2. **Recompensa consistencia**: Documentos que aparecen en múltiples listas suman más
3. **Suaviza errores**: Un ranking malo no domina el resultado final

#### 6.2.4 Resultados Empíricos

**Benchmarks publicados** (Ma et al., 2023):

| Dataset                | RAG Básico     | Multi-Query (3 vars) | Multi-Query (5 vars) | Mejora     |
| ---------------------- | -------------- | -------------------- | -------------------- | ---------- |
| NQ (Natural Questions) | 0.412 Recall@5 | 0.498 Recall@5       | 0.521 Recall@5       | **+26.5%** |
| HotpotQA               | 0.385 Recall@5 | 0.461 Recall@5       | 0.478 Recall@5       | **+24.2%** |
| MS MARCO               | 0.356 Recall@5 | 0.428 Recall@5       | 0.447 Recall@5       | **+25.6%** |

**Costo computacional**: ~3-5x latencia (3-5 llamadas a embedding + búsqueda)

#### 6.2.5 Cuándo Usar Multi-Query

**✅ Escenarios ideales:**

- **Queries ambiguos**: "bases vectoriales" (¿definición? ¿implementación? ¿comparación?)
- **Queries cortos**: "HNSW" (necesita expansión de contexto)
- **Score gap bajo**: Top-10 resultados tienen scores muy similares (poca confianza)
- **Vocabulario diverso**: Documentos usan múltiples terminologías para mismo concepto
- **Recall crítico**: Mejor perder 1-2 documentos irrelevantes que perder 1 relevante

**❌ Evitar cuando:**

- **Queries específicos**: "configuración M=16 en HNSW" (ya es específico)
- **Latencia crítica**: Sistema de tiempo real (<200ms)
- **Presupuesto limitado**: Cada query consume 3-5x API calls

---

### 6.3 Query Decomposition (Descomposición de Consultas)

#### 6.3.1 Fundamento Teórico

**Problema central**: Preguntas complejas contienen múltiples sub-preguntas que requieren información de diferentes fuentes.

**Hipótesis**: Dividir una pregunta compleja en sub-preguntas más simples permite búsquedas más precisas y respuestas más completas.

**Inspiración**: Estrategia humana de "divide y conquista" para problemas complejos.

**Referencias académicas:**

- Press et al. (2023) - "Measuring and Narrowing the Compositionality Gap in Language Models" ([arXiv:2210.03350](https://arxiv.org/abs/2210.03350))
- Khot et al. (2023) - "Decomposed Prompting: A Modular Approach for Solving Complex Tasks" ([arXiv:2210.02406](https://arxiv.org/abs/2210.02406))

#### 6.3.2 Flujo de Ejecución

```
┌─────────────────────────────────────────────────────────────┐
│ Fase 1: Análisis y Descomposición                           │
└─────────────────────────────────────────────────────────────┘

Query Complejo:
"¿Cuáles son las ventajas y desventajas de usar Qdrant vs
PostgreSQL para búsqueda vectorial, y cuál es mejor para
un startup con presupuesto limitado?"

    ↓ [LLM: Identifica componentes de la pregunta]

Componentes detectados:
- Comparación de dos sistemas (Qdrant, PostgreSQL)
- Aspectos múltiples (ventajas, desventajas)
- Contexto específico (startup, presupuesto limitado)

    ↓ [LLM: Genera sub-preguntas atómicas]

Sub-Q1: "¿Cuáles son las principales ventajas de Qdrant?"
Sub-Q2: "¿Cuáles son las desventajas o limitaciones de Qdrant?"
Sub-Q3: "¿Cuáles son las ventajas de PostgreSQL con pgvector?"
Sub-Q4: "¿Cuáles son las desventajas de PostgreSQL+pgvector?"
Sub-Q5: "¿Cuál es el costo de operación de Qdrant?"
Sub-Q6: "¿Cuál es el costo de operación de PostgreSQL?"

┌─────────────────────────────────────────────────────────────┐
│ Fase 2: Búsqueda Independiente por Sub-Query                │
└─────────────────────────────────────────────────────────────┘

Para cada sub-pregunta:
    Embedding → Búsqueda Vectorial → Top-K resultados

Sub-Q1 → Resultados sobre ventajas Qdrant
Sub-Q2 → Resultados sobre desventajas Qdrant
Sub-Q3 → Resultados sobre ventajas PostgreSQL
...

Cada sub-query recupera fragmentos específicos a su aspecto

┌─────────────────────────────────────────────────────────────┐
│ Fase 3: Agregación de Evidencia                             │
└─────────────────────────────────────────────────────────────┘

Agrupar resultados por tema:

Qdrant:
  Ventajas: [Doc1, Doc5, Doc8]
  Desventajas: [Doc3, Doc12]
  Costos: [Doc15, Doc20]

PostgreSQL:
  Ventajas: [Doc2, Doc6, Doc9]
  Desventajas: [Doc4, Doc7]
  Costos: [Doc16, Doc18]

┌─────────────────────────────────────────────────────────────┐
│ Fase 4: Síntesis (Opcional pero recomendado)                │
└─────────────────────────────────────────────────────────────┘

Opción A: Retornar respuestas individuales
    - "Ventajas de Qdrant: ..."
    - "Desventajas de Qdrant: ..."
    - ...

Opción B: LLM sintetiza respuesta integrada

    Prompt:
    "Basándote en las siguientes respuestas parciales,
     genera una respuesta coherente y estructurada que
     responda la pregunta original:

     [Sub-respuestas organizadas]

     Pregunta original: [query complejo]"

    ↓ [LLM]

    Respuesta Final:
    "Para un startup con presupuesto limitado, la elección
     entre Qdrant y PostgreSQL depende de varios factores:

     Qdrant ofrece [ventajas], pero [desventajas]...
     PostgreSQL provee [ventajas], aunque [desventajas]...

     En términos de costo, [análisis comparativo]...

     Recomendación: [síntesis basada en contexto]"
```

#### 6.3.3 Estrategias de Descomposición

**1. Sequential Decomposition** (Secuencial):

```
P: "¿Qué institución fundó el creador de Python?"

Sub-Q1: "¿Quién creó Python?" → Respuesta: "Guido van Rossum"
Sub-Q2: "¿Qué institución fundó Guido van Rossum?" → [Búsqueda]

Útil para: Preguntas multi-hop donde respuesta de Q1 es input de Q2
```

**2. Parallel Decomposition** (Paralelo):

```
P: "Compara HNSW, IVF y Scalar Quantization"

Sub-Q1: "¿Cómo funciona HNSW?"
Sub-Q2: "¿Cómo funciona IVF?"
Sub-Q3: "¿Cómo funciona Scalar Quantization?"

[Todas se buscan en paralelo, luego se comparan]

Útil para: Comparaciones, análisis multi-aspecto
```

**3. Hierarchical Decomposition** (Jerárquico):

```
P: "Explica el estado del arte en búsqueda vectorial"

Sub-Q1: "¿Qué algoritmos de búsqueda vectorial existen?"
    Sub-Q1.1: "¿Qué es HNSW?"
    Sub-Q1.2: "¿Qué es IVF?"
Sub-Q2: "¿Cuáles son las métricas de evaluación?"
Sub-Q3: "¿Qué papers recientes destacan?"

Útil para: Temas amplios que requieren estructura
```

#### 6.3.4 Resultados Empíricos

**Benchmarks** (Khot et al., 2023):

| Dataset                | RAG Básico    | Query Decomposition | Mejora     |
| ---------------------- | ------------- | ------------------- | ---------- |
| StrategyQA (multi-hop) | 0.58 Accuracy | 0.73 Accuracy       | **+25.9%** |
| 2WikiMultihopQA        | 0.42 F1       | 0.59 F1             | **+40.5%** |
| HotpotQA (fullwiki)    | 0.38 EM       | 0.51 EM             | **+34.2%** |

**Observación clave**: Las mejoras son más dramáticas en preguntas que requieren razonamiento multi-hop.

#### 6.3.5 Cuándo Usar Query Decomposition

**✅ Escenarios ideales:**

- **Preguntas comparativas**: "A vs B", "ventajas y desventajas"
- **Queries con conjunciones**: "¿Qué es X y cómo se relaciona con Y?"
- **Multi-hop reasoning**: Respuesta requiere información de múltiples documentos relacionados
- **Listas o enumeraciones**: "Lista 5 características de X"
- **Análisis estructurados**: Cuando quieres respuesta con secciones claras

**❌ Evitar cuando:**

- **Preguntas simples**: "¿Qué es HNSW?" (no requiere descomposición)
- **Latencia crítica**: Múltiples búsquedas + síntesis añade latencia significativa
- **Preguntas atómicas**: Ya son indivisibles

---

### 6.4 HyDE (Hypothetical Document Embeddings)

#### 6.4.1 Fundamento Teórico

**Problema central**: Existe un **domain gap** entre cómo los usuarios formulan preguntas (lenguaje casual, incompleto) y cómo están escritos los documentos (lenguaje formal, técnico, completo).

**Hipótesis clave**: Si generamos un documento hipotético que respondería perfectamente la pregunta, su embedding estará más cerca de los documentos reales que el embedding de la pregunta directa.

**Intuición**: Los documentos se parecen más entre sí que una pregunta a un documento.

**Referencias académicas:**

- Gao et al. (2023) - "Precise Zero-Shot Dense Retrieval without Relevance Labels" ([arXiv:2212.10496](https://arxiv.org/abs/2212.10496))
  - Paper original que introduce HyDE
  - Demuestran mejoras del 5-10% en MS MARCO y Natural Questions

#### 6.4.2 Flujo de Ejecución

```
┌─────────────────────────────────────────────────────────────┐
│ Fase 1: Generación de Documento Hipotético                  │
└─────────────────────────────────────────────────────────────┘

Query del Usuario: "¿Qué es HNSW?"

    ↓ [LLM: Genera documento hipotético que respondería la pregunta]

Prompt al LLM:
"Genera un párrafo técnico y detallado que responda perfectamente
esta pregunta. Usa terminología precisa y estructura de artículo
académico.

Pregunta: ¿Qué es HNSW?

Respuesta técnica:"

    ↓ [LLM genera]

Documento Hipotético:
"HNSW (Hierarchical Navigable Small World) es un algoritmo de
búsqueda aproximada de vecinos más cercanos basado en grafos.
Construye una estructura multicapa donde cada capa contiene un
subconjunto de nodos conectados a sus vecinos más cercanos.
Durante la búsqueda, el algoritmo navega desde capas superiores
(dispersas) hacia capas inferiores (densas), logrando complejidad
logarítmica O(log N). Los parámetros clave son M (conexiones por
nodo) y ef_construction (calidad del índice). HNSW es el algoritmo
preferido en bases de datos vectoriales modernas por su balance
entre velocidad y precisión, alcanzando recall@10 > 95% con
latencias bajo 10ms en datasets de millones de vectores..."

┌─────────────────────────────────────────────────────────────┐
│ Fase 2: Embedding del Documento Hipotético                  │
└─────────────────────────────────────────────────────────────┘

IMPORTANTE: Usar prefijo "passage", NO "query"

    documento_hipotetico → encode("passage: " + texto)

    ↓

    hyp_embedding [768 floats]

Razón: El documento hipotético es un documento, no un query

┌─────────────────────────────────────────────────────────────┐
│ Fase 3: Búsqueda Vectorial con Embedding Hipotético         │
└─────────────────────────────────────────────────────────────┘

Búsqueda usando hyp_embedding (NO el query original)

    Vector DB search(hyp_embedding, k=50)

    ↓

    Resultados: Documentos más similares al hipotético

Estos documentos probablemente:
- Usan terminología técnica similar
- Tienen estructura similar (definición → características → ejemplo)
- Están en el mismo "espacio semántico"

┌─────────────────────────────────────────────────────────────┐
│ Fase 4: Generación de Respuesta Final (Opcional)            │
└─────────────────────────────────────────────────────────────┘

Opción A: Retornar documentos recuperados directamente

Opción B: LLM genera respuesta usando documentos reales
    "Basándote en estos documentos [resultados], responde: [query original]"

    Ventaja: Respuesta fundamentada en docs reales, no en hipotético (que puede tener "alucinaciones")
```

#### 6.4.3 Visualización del Domain Gap

```
Espacio de Embeddings (simplificado a 2D):

                Documentos Técnicos
                (lenguaje formal)
                      ●
                  ●       ●
              ●               ●
          ●                       ●
              ●   [Cluster denso]
                  ●       ●
                      ●

    Distancia grande ↑
                     │
                     │
    Query Usuario    ●   "¿Qué es HNSW?"
    (lenguaje casual)

──────────────────────────────────────────────────

Con HyDE:

                Documentos Técnicos
                      ●
                  ●   ★   ●    ★ = Documento Hipotético
              ●       ↑       ●    (genera LLM)
          ●       Distancia corta
              ●                 ●
                  ●       ●
                      ●

    Query Usuario    ●   "¿Qué es HNSW?"
    (se usa solo para generar hipotético, no para búsqueda)
```

#### 6.4.4 Por Qué Funciona: Análisis Profundo

**1. Cierre del Domain Gap:**

Los modelos de embedding (E5, BERT, etc.) capturan:

- **Estilo**: Formal vs. casual
- **Densidad**: Párrafos completos vs. frases cortas
- **Terminología**: Técnica vs. coloquial

Un documento hipotético generado por LLM "habla el mismo idioma" que los documentos reales.

**2. Expansión Implícita:**

El documento hipotético contiene:

- Términos relacionados que no estaban en el query
- Contexto adicional
- Sinónimos y variaciones

Ejemplo:

- Query: "¿Qué es HNSW?"
- Hipotético incluye: "grafos", "vecinos", "multicapa", "logarítmico", "recall", "latencia"
- Búsqueda captura documentos con estos términos relacionados

**3. Robustez a Alucinaciones:**

Incluso si el LLM "alucina" en el documento hipotético, **solo se usa para búsqueda**, no como respuesta final. Los documentos reales recuperados son la fuente de verdad.

#### 6.4.5 Resultados Empíricos

**Paper original** (Gao et al., 2023):

| Dataset           | Dense Retrieval | Dense + HyDE    | Mejora     |
| ----------------- | --------------- | --------------- | ---------- |
| MS MARCO (Dev)    | 0.352 MRR@10    | 0.389 MRR@10    | **+10.5%** |
| Natural Questions | 0.434 Recall@20 | 0.478 Recall@20 | **+10.1%** |
| TriviaQA          | 0.512 Recall@20 | 0.551 Recall@20 | **+7.6%**  |
| TREC-COVID        | 0.656 NDCG@10   | 0.702 NDCG@10   | **+7.0%**  |

**Hallazgo clave**: HyDE funciona mejor en dominios técnicos/especializados donde el domain gap es más pronunciado.

#### 6.4.6 Cuándo Usar HyDE

**✅ Escenarios ideales:**

- **Queries muy cortos**: "HNSW", "pgvector" (necesitan expansión)
- **Domain gap conocido**: Usuarios casuales buscando en docs técnicos
- **Queries abstractos**: "Mejor algoritmo para vectores" (HyDE concretiza)
- **Dominio especializado**: Médico, legal, científico (terminología específica)
- **Zero-shot scenarios**: Sin ejemplos de queries para el dominio

**❌ Evitar cuando:**

- **Queries ya técnicos**: "Configuración M=16, ef_construction=200 en HNSW" (ya es específico)
- **LLM no confiable**: Si LLM alucina mucho, documento hipotético será ruido
- **Latencia crítica**: Generación de documento hipotético añade 1-3 segundos
- **Presupuesto limitado**: Requiere llamada adicional a LLM

---

### 6.5 Hybrid Search (Búsqueda Híbrida: BM25 + Vectorial)

#### 6.5.1 Fundamento Teórico

**Problema central**: La búsqueda vectorial pura es "ciega" a coincidencias lexicales exactas. Puede perder documentos que contienen keywords exactas que el usuario menciona.

**Hipótesis**: Combinar búsqueda lexical (keyword-based) con búsqueda semántica (vector-based) ofrece lo mejor de ambos mundos.

**Inspiración**: Motores de búsqueda tradicionales (Google, Elasticsearch) siempre han combinado múltiples señales.

**Referencias académicas:**

- Lin et al. (2021) - "A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques" ([arXiv:2106.14807](https://arxiv.org/abs/2106.14807))
- Robertson & Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond" (Foundations and Trends in Information Retrieval)

#### 6.5.2 Componentes: BM25 Explicado

**BM25 (Best Matching 25)** es el algoritmo estándar de facto para búsqueda lexical, evolución de TF-IDF.

**Fórmula completa:**

```
score(D, Q) = Σ IDF(qi) × [ f(qi, D) × (k1 + 1) ] / [ f(qi, D) + k1 × (1 - b + b × |D| / avgdl) ]

Donde:
- D: Documento
- Q: Query (conjunto de términos {q1, q2, ..., qn})
- f(qi, D): Frecuencia del término qi en documento D
- |D|: Longitud del documento D (número de palabras)
- avgdl: Longitud promedio de todos los documentos
- k1: Parámetro de saturación de frecuencia del término (típicamente 1.2-2.0)
- b: Parámetro de normalización de longitud (típicamente 0.75)

IDF(qi) = log[ (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 ]
- N: Número total de documentos
- n(qi): Número de documentos que contienen qi
```

**Intuición de cada componente:**

1. **Term Frequency** `f(qi, D)`:

   - Más apariciones del término → mayor relevancia
   - Pero con **saturación**: 10 apariciones no es 10x más relevante que 1

2. **IDF (Inverse Document Frequency)**:

   - Términos raros (aparecen en pocos docs) son más informativos
   - "the", "is", "a" tienen IDF bajo (aparecen en todos lados)
   - "HNSW", "pgvector" tienen IDF alto (términos específicos)

3. **Document Length Normalization** `(1 - b + b × |D| / avgdl)`:

   - Documentos largos no deben dominar solo por ser largos
   - b=0: sin normalización
   - b=1: normalización completa por longitud

4. **k1 (Saturación)**:
   - Controla cuán rápido la frecuencia satura
   - k1=1.2 (típico): después de ~5 apariciones, retornos decrecientes
   - k1=0: frecuencia binaria (presente o no)
   - k1=∞: frecuencia lineal (no satura)

**Ejemplo concreto:**

```
Corpus:
Doc1: "HNSW es un algoritmo de grafos para búsqueda vectorial"
Doc2: "El algoritmo HNSW usa grafos navegables"
Doc3: "PostgreSQL es una base de datos relacional"

Query: "algoritmo HNSW"

Cálculos:

Para Doc1:
- "algoritmo": f=1, IDF=log[(3-2+0.5)/(2+0.5)+1]=0.405
- "HNSW": f=1, IDF=log[(3-2+0.5)/(2+0.5)+1]=0.405
- |D1|=9, avgdl=8.67
- TF_component("algoritmo") = [1×2.2]/[1+1.2×(0.25+0.75×9/8.67)] = 1.05
- TF_component("HNSW") = 1.05
- score(Doc1) = 0.405×1.05 + 0.405×1.05 = 0.85

Para Doc2: score = 0.82 (similar cálculo)
Para Doc3: score = 0.14 (solo "algoritmo" pero no "HNSW")

Ranking BM25: Doc1 > Doc2 >> Doc3
```

#### 6.5.3 Flujo de Ejecución Hybrid Search

```
┌─────────────────────────────────────────────────────────────┐
│ Fase 1: Búsqueda Semántica (Dense Retrieval)                │
└─────────────────────────────────────────────────────────────┘

Query: "mejor algoritmo para millones de vectores"

    ↓ [Embedding Model]

    query_vector [768 floats]

    ↓ [Vector Database: Cosine Similarity]

    Resultados Semánticos (Top-50):

    1. Doc_A (score=0.89): "HNSW escala a millones de vectores..."
    2. Doc_B (score=0.85): "IVF es eficiente para grandes datasets..."
    3. Doc_C (score=0.82): "Scalar quantization reduce memoria..."
    ...
    50. Doc_Z (score=0.45)

┌─────────────────────────────────────────────────────────────┐
│ Fase 2: Búsqueda Lexical (Sparse Retrieval / BM25)          │
└─────────────────────────────────────────────────────────────┘

Query: "mejor algoritmo para millones de vectores"

    ↓ [Tokenización + Filtrado stopwords]

    tokens_relevantes: ["algoritmo", "millones", "vectores"]

    ↓ [BM25 sobre todo el corpus]

    Resultados BM25 (Top-50):

    1. Doc_D (score=12.3): "Algoritmo IVF para millones de vectores..."
    2. Doc_A (score=11.8): "HNSW escala a millones de vectores..."
    3. Doc_E (score=10.5): "Comparación de algoritmos vectoriales..."
    ...
    50. Doc_Y (score=2.1)

Nota: Doc_A aparece en ambas listas (buena señal!)

┌─────────────────────────────────────────────────────────────┐
│ Fase 3: Fusión con RRF (Reciprocal Rank Fusion)             │
└─────────────────────────────────────────────────────────────┘

Opción A: RRF (recomendado - usado en este proyecto)

Para cada documento único en cualquier lista:

    RRF_score(d) = α × [1/(60+rank_semantic(d))] +
                   (1-α) × [1/(60+rank_bm25(d))]

    Donde α = peso semántico (típicamente 0.7)

Ejemplo:

Doc_A:
- Rank semántico: 1  → 1/(60+1)  = 0.0164
- Rank BM25: 2       → 1/(60+2)  = 0.0161
- RRF = 0.7×0.0164 + 0.3×0.0161 = 0.0163

Doc_D:
- Rank semántico: 12 → 1/(60+12) = 0.0139
- Rank BM25: 1       → 1/(60+1)  = 0.0164
- RRF = 0.7×0.0139 + 0.3×0.0164 = 0.0146

Doc_C:
- Rank semántico: 3  → 1/(60+3)  = 0.0159
- Rank BM25: no aparece → 0
- RRF = 0.7×0.0159 + 0.3×0 = 0.0111

Ranking Final: Doc_A > Doc_D > Doc_C > ...

Opción B: Weighted Sum (alternativa)

    final_score(d) = α × normalize(semantic_score(d)) +
                     (1-α) × normalize(bm25_score(d))

    Problema: Requiere normalización de scores (MinMax, Z-score, etc.)
    Ventaja: Usa scores originales (más información)

┌─────────────────────────────────────────────────────────────┐
│ Fase 4: Re-ranking y Selección                              │
└─────────────────────────────────────────────────────────────┘

Ordenar todos documentos únicos por score fusionado
Retornar Top-K final (típicamente 5-10)

Opcionalmente: Aplicar MMR para diversidad
```

#### 6.5.4 Ajuste del Parámetro α (Peso Semántico)

El parámetro α controla el balance semántico vs. lexical:

```
α = 0.0: Solo BM25 (búsqueda lexical pura)
α = 0.3: Favorece keywords (30% semántico, 70% lexical)
α = 0.5: Balance equitativo
α = 0.7: Favorece semántica (70% semántico, 30% lexical) ← Default recomendado
α = 1.0: Solo vectorial (búsqueda semántica pura)
```

**Heurística para elegir α:**

| Escenario                             | α Recomendado | Razón                             |
| ------------------------------------- | ------------- | --------------------------------- |
| Queries con nombres propios, IDs      | 0.3-0.5       | Coincidencia exacta crítica       |
| Queries técnicos con jerga específica | 0.4-0.6       | Terminología exacta importa       |
| Queries conceptuales/abstractos       | 0.7-0.9       | Semántica domina                  |
| Queries bien formulados (oraciones)   | 0.6-0.8       | Balance con inclinación semántica |
| Queries de 1-2 palabras               | 0.5-0.6       | Keywords son toda la información  |

**Experimento recomendado:**

```python
# Probar diferentes α con tus queries de validación
alphas = [0.3, 0.5, 0.7, 0.9]
for α in alphas:
    results = hybrid_search(query, semantic_weight=α)
    evaluate_recall(results, ground_truth)

# Seleccionar α que maximiza recall/precision en tu dataset
```

#### 6.5.5 Resultados Empíricos

**Benchmarks** (Lin et al., 2021):

| Dataset           | Dense Only      | BM25 Only       | Hybrid (α=0.7)      | Mejora vs Mejor Individual |
| ----------------- | --------------- | --------------- | ------------------- | -------------------------- |
| MS MARCO Dev      | 0.335 MRR@10    | 0.187 MRR@10    | **0.369 MRR@10**    | +10.1% vs Dense            |
| TREC DL 2019      | 0.645 NDCG@10   | 0.497 NDCG@10   | **0.682 NDCG@10**   | +5.7% vs Dense             |
| Natural Questions | 0.452 Recall@20 | 0.328 Recall@20 | **0.487 Recall@20** | +7.7% vs Dense             |

**Observación clave**: Hybrid search casi siempre iguala o supera al mejor sistema individual.

#### 6.5.6 Cuándo Usar Hybrid Search

**✅ Escenarios ideales:**

- **General purpose**: Excelente default para casi todos los casos
- **Queries con nombres propios**: "PostgreSQL 15", "HNSW paper 2016"
- **Queries con IDs o códigos**: "error CODE-1234", "RFC-9293"
- **Queries con números**: "vector de 768 dimensiones", "latencia bajo 10ms"
- **Acronyms y abreviaciones**: "NLP", "RAG", "ANN"
- **Términos técnicos específicos**: Donde coincidencia exacta es valiosa

**❌ No es necesario cuando:**

- **Solo búsqueda semántica profunda**: Análisis de sentimiento, conceptos abstractos
- **Corpus muy pequeño** (<1000 docs): Overhead no justificado
- **Todos los queries son largos y descriptivos**: Semántica pura funciona bien

---

### 6.6 Iterative Retrieval (Recuperación Iterativa Multi-Ronda)

#### 6.6.1 Fundamento Teórico

**Problema central**: Una sola ronda de búsqueda puede no capturar toda la información necesaria para responder preguntas complejas o multi-hop.

**Hipótesis**: Iterativamente generar nuevas búsquedas basadas en gaps de información identificados permite construir contexto completo progresivamente.

**Inspiración**:

- Estrategia humana de investigación: buscar → leer → identificar gaps → buscar más
- Self-RAG y ReACT patterns en LLM research

**Referencias académicas:**

- Asai et al. (2023) - "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511))
- Yao et al. (2023) - "ReAct: Synergizing Reasoning and Acting in Language Models" ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629))
- Shao et al. (2023) - "Enhancing Retrieval-Augmented LMs with Iterative Retrieval-Generation Synergy" ([arXiv:2305.15294](https://arxiv.org/abs/2305.15294))

#### 6.6.2 Flujo de Ejecución

```
┌─────────────────────────────────────────────────────────────┐
│ ROUND 1: Búsqueda Inicial                                   │
└─────────────────────────────────────────────────────────────┘

Query Original:
"¿Cómo configurar HNSW en Qdrant y PostgreSQL para óptimo rendimiento?"

    ↓ [Búsqueda estándar]

    Documentos Recuperados (Top-10):
    - Doc1: "HNSW es un algoritmo de grafos..."
    - Doc2: "Qdrant usa HNSW por defecto..."
    - Doc3: "PostgreSQL soporta múltiples índices..."
    - ... (8 docs más, contenido general)

    ↓ [Construir contexto]

    Context_R1 = concatenate(docs)

    ↓ [LLM: Evaluar answerability]

    Prompt:
    "¿Puedes responder completamente esta pregunta con el contexto dado?

     Pregunta: [query original]
     Contexto: [Context_R1]

     Responde:
     RESPUESTA: [SI/NO/PARCIAL]
     CONFIANZA: [0-100]
     FALTA: [qué información específica falta]"

    ↓ [LLM responde]

    Análisis:
    RESPUESTA: PARCIAL
    CONFIANZA: 45
    FALTA: "parámetros específicos de configuración HNSW (M, ef_construction),
            valores recomendados, diferencias entre Qdrant y PostgreSQL"

Decisión: NO answerable → Continuar a Round 2

┌─────────────────────────────────────────────────────────────┐
│ ROUND 2: Búsqueda Enfocada (Followup Query)                 │
└─────────────────────────────────────────────────────────────┘

    ↓ [LLM: Generar followup query]

    Prompt:
    "Genera UNA pregunta específica para buscar la información que falta.

     Pregunta original: [query]
     Información que falta: [gaps identificados]

     Pregunta específica:"

    ↓ [LLM genera]

    Followup Query:
    "¿Cuáles son los parámetros M y ef_construction de HNSW y sus valores recomendados?"

    ↓ [Búsqueda con followup]

    Nuevos Documentos:
    - Doc11: "M controla conexiones por nodo, típicamente 16..."
    - Doc12: "ef_construction entre 100-400, trade-off..."
    - Doc13: "Qdrant recomienda M=16, ef=200 para general..."
    - ... (más docs específicos)

    ↓ [Acumular contexto]

    Context_R2 = Context_R1 + new_docs
    (todos los docs de R1 + nuevos de R2)

    ↓ [Re-evaluar answerability]

    Análisis:
    RESPUESTA: PARCIAL
    CONFIANZA: 70
    FALTA: "configuración específica de PostgreSQL pgvector"

Decisión: Confianza aumentó pero aún <75% → Continuar a Round 3

┌─────────────────────────────────────────────────────────────┐
│ ROUND 3: Refinamiento Final                                 │
└─────────────────────────────────────────────────────────────┘

    Followup Query 2:
    "Configuración de índice HNSW en PostgreSQL pgvector"

    ↓ [Búsqueda]

    Más Documentos:
    - Doc21: "CREATE INDEX USING hnsw (embedding vector_cosine_ops)
              WITH (m = 16, ef_construction = 200)..."
    - Doc22: "pgvector desde v0.5 soporta HNSW..."

    ↓ [Acumular]

    Context_R3 = Context_R2 + new_docs

    ↓ [Re-evaluar]

    Análisis:
    RESPUESTA: SI
    CONFIANZA: 85
    FALTA: nada

Decisión: ANSWERABLE → Stop, generar respuesta final

┌─────────────────────────────────────────────────────────────┐
│ Generación de Respuesta Final                               │
└─────────────────────────────────────────────────────────────┘

    ↓ [LLM con contexto completo]

    Prompt:
    "Basándote en el siguiente contexto completo, responde la pregunta:

     Contexto: [Context_R3 - todos los docs de 3 rondas]
     Pregunta: [query original]

     Respuesta clara y estructurada:"

    ↓ [LLM genera respuesta final]

    Respuesta:
    "Para configurar HNSW óptimamente:

     En Qdrant:
     - M=16 (conexiones por nodo)
     - ef_construction=200 (calidad del índice)
     - Configuración en collection_create:
       [ejemplo con valores]

     En PostgreSQL:
     - Requiere pgvector v0.5+
     - CREATE INDEX ... WITH (m=16, ef_construction=200)
     - [ejemplo específico]

     Valores recomendados:
     - Dataset pequeño (<100k): M=16, ef=100
     - Dataset mediano (100k-1M): M=16, ef=200
     - Dataset grande (>1M): M=24, ef=400

     Trade-off: Mayor M/ef = mejor precisión pero más memoria/tiempo"

Metadata retornado:
- rounds_executed: 3
- retrieval_calls: 3
- confidence_progression: [45, 70, 85]
- total_documents: 28
- unique_documents: 24
```

#### 6.6.3 Conceptos Clave

**1. Answerability Check:**

Evaluar si el contexto actual es suficiente para responder. Criterios:

- **Coverage**: ¿Todos los aspectos de la pregunta están cubiertos?
- **Specificity**: ¿La información es suficientemente específica?
- **Consistency**: ¿Hay contradicciones en los documentos?

**2. Gap Identification:**

Identificar qué información específica falta. Tipos de gaps:

- **Factual gaps**: "No sé el valor de X"
- **Aspect gaps**: "Cubrí A pero no B"
- **Depth gaps**: "Tengo definición pero no ejemplos"

**3. Followup Query Generation:**

Generar query específico para llenar gap. Características:

- **Específico**: No vago como "más información"
- **Accionable**: Puede ser buscado directamente
- **Complementario**: No repite información ya obtenida

**4. Stopping Criteria:**

Cuándo parar la iteración:

- **Answerability achieved**: Confidence > threshold (típicamente 75-80%)
- **Max rounds reached**: Límite presupuestario (típicamente 3-5 rondas)
- **No progress**: Confidence no aumenta entre rondas
- **No followup possible**: LLM no puede generar followup específico

#### 6.6.4 Variantes de Iterative Retrieval

**1. Sequential (usado en este proyecto):**

```
R1 → Eval → R2 → Eval → R3 → Answer

Ventaja: Control fino, puede parar temprano
Desventaja: Latencia acumulativa
```

**2. Parallel with Refinement:**

```
R1 (broad query)
↓
Generate 3 followups in parallel
↓
R2a, R2b, R2c (execute simultaneously)
↓
Merge and answer

Ventaja: Menor latencia
Desventaja: Puede buscar info redundante
```

**3. Hierarchical:**

```
R1: Top-level search
↓
Identify sub-topics
↓
R2: Parallel searches for each sub-topic
↓
R3: Deep-dive on most relevant sub-topic

Ventaja: Estructura clara
Desventaja: Complejo de implementar
```

#### 6.6.5 Resultados Empíricos

**Self-RAG Paper** (Asai et al., 2023):

| Dataset                   | Standard RAG  | Iterative (2-3 rounds) | Mejora     |
| ------------------------- | ------------- | ---------------------- | ---------- |
| 2WikiMultihopQA           | 0.42 F1       | 0.58 F1                | **+38.1%** |
| HotpotQA (fullwiki)       | 0.38 EM       | 0.52 EM                | **+36.8%** |
| FEVER (fact verification) | 0.71 Accuracy | 0.81 Accuracy          | **+14.1%** |
| StrategyQA                | 0.58 Accuracy | 0.70 Accuracy          | **+20.7%** |

**Observación**: Mejoras son dramáticas en datasets multi-hop. Beneficio marginal en preguntas simples.

**Costo computacional**:

- 3 rounds promedio
- ~3x latencia vs. RAG básico
- ~3x costos de API LLM (si usa API externa)

#### 6.6.6 Cuándo Usar Iterative Retrieval

**✅ Escenarios ideales:**

- **Multi-hop questions**: "¿Quién fundó la empresa del creador de Python?"
- **Preguntas complejas exhaustivas**: Requieren información de múltiples fuentes
- **Análisis profundos**: "Análisis completo de pros/cons de X"
- **Investigación exploratoria**: Usuario no sabe exactamente qué buscar
- **Alta precisión crítica**: Vale la pena latencia extra por respuesta completa

**❌ Evitar cuando:**

- **Preguntas simples factuales**: "¿Qué es HNSW?" (no requiere iteración)
- **Latencia crítica**: Sistema tiempo real (<500ms)
- **Presupuesto limitado**: 3-5x costo vs. RAG básico
- **Queries bien scoped**: Ya son específicos y dirigidos

---

### 6.7 Comparativa de las 5 Estrategias

| Estrategia        | Complejidad | Latencia Típica | Mejora Recall | Mejora Precision | Costo LLM | Mejor Para                                |
| ----------------- | ----------- | --------------- | ------------- | ---------------- | --------- | ----------------------------------------- |
| **Multi-Query**   | 🟡 Media    | ~200ms          | +15-25%       | +10-15%          | 3-5x      | Queries ambiguos, vocabulario diverso     |
| **Decomposition** | 🟡 Media    | ~300ms          | +20-40%       | +15-25%          | 2-3x      | Comparaciones, preguntas multi-parte      |
| **HyDE**          | 🟡 Media    | ~250ms          | +5-10%        | +10-15%          | 1x        | Domain gap, queries abstractos            |
| **Hybrid Search** | 🟢 Baja     | ~80ms           | +5-15%        | +10-20%          | 0x        | **General purpose**, keywords importantes |
| **Iterative**     | 🔴 Alta     | ~1000ms         | +30-40%       | +20-30%          | 3-5x      | Multi-hop, exhaustividad crítica          |

**Recomendación estratégica:**

```
Startup/MVP:
  ├─ Implementar: Hybrid Search (mejor ROI)
  └─ Opcional: HyDE si domain gap es evidente

Producción General:
  ├─ Default: Hybrid Search
  ├─ Queries cortos/ambiguos: + Multi-Query
  └─ Monitoreo: Identificar patrones de queries que fallan

Alto Performance:
  ├─ Orchestrated RAG (auto-selecciona técnica)
  └─ A/B testing continuo de técnicas

Investigación/Academia:
  └─ Implementar todas, benchmarking exhaustivo
```

### 6.8 Implementación en Este Proyecto

Todas las 5 estrategias están implementadas en:

- **`app/advanced_rag.py`**: Implementaciones individuales de cada técnica
- **`app/orchestrated_rag.py`**: Sistema inteligente que auto-selecciona técnicas basado en señales del query

**Endpoints disponibles:**

```bash
# Probar técnicas individuales
/advanced/multi-query?q=tu_query
/advanced/decompose?q=tu_query
/advanced/hyde?q=tu_query
/advanced/hybrid?q=tu_query
/advanced/iterative?q=tu_query

# Sistema automático (recomendado)
/orchestrated?q=tu_query
```

Consulta las **Secciones 9 y 7.3** de este documento para detalles de implementación y arquitectura.

---

## 7. Comparativa de Bases de Datos Vectoriales

### 7.1 Qdrant

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

### 7.2 PostgreSQL + pgvector

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

### 7.3 Otros Sistemas Populares

#### 7.3.1 Pinecone

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

#### 7.3.2 Weaviate

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

#### 7.3.3 Milvus

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

### 7.4 Comparativa: Qdrant vs. PostgreSQL+pgvector

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

## 8. Arquitectura del Proyecto

### 8.1 Visión General

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

### 8.2 Estructura de Directorios

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

### 8.3 Componentes Principales

#### 8.3.1 FastAPI Application (app/main.py)

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

#### 8.3.2 RAG Engine (app/rag.py)

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

#### 8.3.3 Advanced RAG (app/advanced_rag.py)

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

#### 8.3.4 Orchestrated RAG (app/orchestrated_rag.py)

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

#### 8.3.5 Backends (qdrant_backend.py, pgvector_backend.py)

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

### 8.4 Flujo de Datos

#### 8.4.1 Indexación (Offline)

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

#### 8.4.2 Consulta (Runtime)

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

### 8.5 Configuración Multi-Algoritmo

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

## 9. Pipeline de Procesamiento

### 9.1 Pipeline Básico (main_pipeline.py)

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

### 9.2 Fases del Pipeline

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

### 9.3 Pipeline con Procesamiento Paralelo

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

### 9.4 Monitoreo y Estadísticas

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

## 10. Técnicas Avanzadas de RAG

### 10.1 Multi-Query Rephrasing

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

### 10.2 Query Decomposition

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

### 10.3 HyDE (Hypothetical Document Embeddings)

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

### 10.4 Hybrid Search (BM25 + Dense)

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

### 10.5 Iterative Retrieval (Multi-Round)

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

### 10.6 Orchestrated RAG (Sistema Inteligente)

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

### 10.7 Comparativa de Técnicas

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

## 11. Optimización y Métricas

### 11.1 Métricas de Evaluación

#### 11.1.1 Retrieval Metrics

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

### 11.2 Optimización de Búsqueda Vectorial

#### 11.2.1 Tuning de HNSW

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

### 11.3 Optimización de LLM

#### 11.3.1 Selección de Modelo

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

### 11.4 Monitoreo en Producción

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

## 12. Despliegue en la Nube

### 12.1 Arquitectura Cloud-Native

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

### 12.2 Opciones por Proveedor

#### 12.2.1 AWS Deployment

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

### 12.3 Managed Services vs. Self-Hosted

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

### 12.4 CI/CD Pipeline

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

### 12.5 Costos Comparativos

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

## 13. Aplicaciones Empresariales

### 13.1 Casos de Uso por Industria

#### 13.1.1 Servicio al Cliente

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

### 13.2 ROI y Justificación de Inversión

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

### 13.3 Consideraciones de Implementación

#### 13.3.1 Data Privacy y Seguridad

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

## 14. Recomendaciones y Mejores Prácticas

### 14.1 Para Estudiantes y Aprendizaje Autónomo

**Ruta de aprendizaje progresivo:**

Esta sección está diseñada para cualquier persona que quiera aprender sobre bases de datos vectoriales y RAG, independientemente de si está en un curso formal o aprendiendo por cuenta propia.

#### **Nivel 1: Fundamentos (1-2 semanas)**

**Objetivos:**

- Comprender qué es un embedding y cómo se genera
- Entender cómo funciona la similitud vectorial
- Realizar tu primera búsqueda semántica

**Ejercicios prácticos:**

1. **Generar tu primer embedding:**

   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('intfloat/multilingual-e5-base')

   # Textos de ejemplo
   textos = [
       "El perro corre en el parque",
       "Un canino juega en el jardín",
       "El auto está en el garaje"
   ]

   # Generar embeddings
   embeddings = model.encode(textos)
   print(f"Forma del vector: {embeddings.shape}")  # (3, 768)

   # Calcular similitud
   from sklearn.metrics.pairwise import cosine_similarity
   similitud = cosine_similarity(embeddings)
   print(similitud)  # Observa que texto 1 y 2 son muy similares
   ```

2. **Explorar el espacio vectorial:**

   ```python
   # Visualizar embeddings en 2D usando t-SNE
   from sklearn.manifold import TSNE
   import matplotlib.pyplot as plt

   # Genera embeddings de varios conceptos
   conceptos = [
       "perro", "gato", "ratón",  # Animales
       "auto", "camión", "bicicleta",  # Vehículos
       "manzana", "naranja", "plátano"  # Frutas
   ]

   embeddings = model.encode(conceptos)

   # Reducir a 2D
   tsne = TSNE(n_components=2, random_state=42)
   coords_2d = tsne.fit_transform(embeddings)

   # Graficar
   plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
   for i, concepto in enumerate(conceptos):
       plt.annotate(concepto, (coords_2d[i, 0], coords_2d[i, 1]))
   plt.title("Espacio de Embeddings (2D)")
   plt.show()
   # Verás que conceptos similares se agrupan
   ```

3. **Experimentar con diferentes métricas de distancia:**

   ```python
   import numpy as np
   from scipy.spatial import distance

   vec1 = embeddings[0]  # "perro"
   vec2 = embeddings[1]  # "gato"
   vec3 = embeddings[3]  # "auto"

   # Comparar métricas
   print(f"Cosine (perro-gato): {1 - distance.cosine(vec1, vec2):.4f}")
   print(f"Cosine (perro-auto): {1 - distance.cosine(vec1, vec3):.4f}")

   print(f"Euclidean (perro-gato): {distance.euclidean(vec1, vec2):.4f}")
   print(f"Euclidean (perro-auto): {distance.euclidean(vec1, vec3):.4f}")

   # Pregunta: ¿Qué métrica diferencia mejor conceptos relacionados vs no relacionados?
   ```

**Recursos de estudio:**

- 📖 [Sentence Transformers Documentation](https://www.sbert.net/)
- 🎥 [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- 📄 Sección 2 de este documento (Embeddings y Representación Vectorial)

#### **Nivel 2: Implementación Básica de RAG (2-3 semanas)**

**Objetivos:**

- Construir un sistema RAG funcional desde cero
- Entender el flujo completo: indexación → búsqueda → generación
- Comparar diferentes backends vectoriales

**Proyecto guiado: "Mi primera base de conocimiento"**

```python
# Paso 1: Preparar datos
documentos = [
    {"id": 1, "contenido": "Python es un lenguaje de programación interpretado..."},
    {"id": 2, "contenido": "JavaScript es esencial para desarrollo web..."},
    {"id": 3, "contenido": "SQL se utiliza para gestionar bases de datos..."},
    # Añade 20-50 documentos sobre un tema que te interese
]

# Paso 2: Generar embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-base')

for doc in documentos:
    doc['embedding'] = model.encode(f"passage: {doc['contenido']}")

# Paso 3A: Indexar en Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="mi_conocimiento",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

points = [
    PointStruct(
        id=doc['id'],
        vector=doc['embedding'].tolist(),
        payload={"contenido": doc['contenido']}
    )
    for doc in documentos
]

client.upsert(collection_name="mi_conocimiento", points=points)

# Paso 4: Búsqueda
def buscar(pregunta, k=3):
    query_emb = model.encode(f"query: {pregunta}")

    resultados = client.search(
        collection_name="mi_conocimiento",
        query_vector=query_emb.tolist(),
        limit=k
    )

    return [(r.score, r.payload['contenido']) for r in resultados]

# Paso 5: Generación (con Ollama local)
import ollama

def preguntar(pregunta):
    # Buscar contexto relevante
    resultados = buscar(pregunta, k=3)
    contexto = "\n\n".join([f"[{i+1}] {cont}" for i, (score, cont) in enumerate(resultados)])

    # Generar respuesta
    prompt = f"""Responde basándote solo en el siguiente contexto:

{contexto}

Pregunta: {pregunta}

Respuesta:"""

    respuesta = ollama.generate(model='phi3:mini', prompt=prompt)
    return respuesta['response']

# Probar
print(preguntar("¿Qué es Python?"))
```

**Ejercicios de experimentación:**

1. **Comparar backends:**

   - Implementa lo mismo con PostgreSQL+pgvector
   - Mide tiempo de indexación y búsqueda
   - Compara resultados de búsqueda

2. **Experimentar con chunking:**

   - Prueba diferentes tamaños de chunk (100, 200, 500 tokens)
   - Mide el impacto en calidad de respuestas
   - ¿Chunks más grandes = mejores respuestas? ¿Por qué o por qué no?

3. **A/B testing de modelos embedding:**
   - Prueba `all-MiniLM-L6-v2` (más rápido, 384 dims)
   - Prueba `multilingual-e5-base` (multilingüe, 768 dims)
   - Compara precision de búsqueda

**Recursos:**

- 📖 Sección 5 de este documento (RAG completo)
- 🔗 [Qdrant Tutorial](https://qdrant.tech/documentation/tutorials/)
- 🔗 [pgvector Examples](https://github.com/pgvector/pgvector-python)

#### **Nivel 3: Técnicas Avanzadas (2-4 semanas)**

**Objetivos:**

- Implementar y comparar técnicas avanzadas de RAG
- Entender cuándo usar cada técnica
- Optimizar rendimiento y calidad

**Proyectos prácticos:**

1. **Implementar Multi-Query Rephrasing:**

   ```python
   def multi_query_search(pregunta_original, num_variaciones=3):
       # Generar variaciones con LLM
       prompt = f"""Genera {num_variaciones} formas alternativas de preguntar:
       "{pregunta_original}"

       Responde solo con las preguntas, una por línea."""

       variaciones = ollama.generate(model='phi3:mini', prompt=prompt)
       queries = [pregunta_original] + variaciones['response'].strip().split('\n')

       # Buscar con cada variación
       todos_resultados = []
       for q in queries:
           resultados = buscar(q, k=20)
           todos_resultados.append(resultados)

       # Fusionar con RRF (Reciprocal Rank Fusion)
       return fusionar_rrf(todos_resultados)

   # Pregunta: ¿Cuándo mejora Multi-Query vs búsqueda simple?
   # Experimento: Compara con queries ambiguos vs específicos
   ```

2. **Implementar Hybrid Search:**

   ```python
   from rank_bm25 import BM25Okapi

   def hybrid_search(pregunta, lambda_weight=0.7):
       # Búsqueda semántica (densa)
       resultados_densos = buscar(pregunta, k=50)

       # Búsqueda keyword (BM25)
       corpus = [doc['contenido'] for doc in documentos]
       tokenized_corpus = [doc.split() for doc in corpus]
       bm25 = BM25Okapi(tokenized_corpus)

       query_tokens = pregunta.split()
       scores_bm25 = bm25.get_scores(query_tokens)

       # Combinar scores
       # Score final = lambda * score_denso + (1-lambda) * score_bm25
       ...

       return resultados_combinados

   # Experimento: Varía lambda de 0.0 a 1.0
   # ¿Cuál es el valor óptimo para tu dataset?
   ```

3. **Implementar HyDE:**

   ```python
   def hyde_search(pregunta):
       # Generar documento hipotético
       prompt = f"""Genera un párrafo que respondería perfectamente esta pregunta:
       {pregunta}

       Escribe solo el párrafo informativo."""

       doc_hipotetico = ollama.generate(model='phi3:mini', prompt=prompt)

       # Buscar usando embedding del documento hipotético (no de la pregunta)
       emb_hipotetico = model.encode(f"passage: {doc_hipotetico['response']}")

       resultados = client.search(
           collection_name="mi_conocimiento",
           query_vector=emb_hipotetico.tolist(),
           limit=5
       )

       return resultados

   # Experimento: ¿Cuándo HyDE supera a búsqueda directa?
   ```

**Desafíos de evaluación:**

1. **Crear benchmark personalizado:**

   ```python
   # Define 20-30 preguntas con respuestas gold standard
   benchmark = [
       {
           "pregunta": "¿Qué es Python?",
           "respuesta_esperada": "lenguaje de programación interpretado...",
           "docs_relevantes": [1, 5, 12]  # IDs de docs que deberían recuperarse
       },
       # ... más preguntas
   ]

   # Medir Recall@K
   def evaluar_recall(metodo_busqueda):
       recalls = []
       for item in benchmark:
           resultados = metodo_busqueda(item['pregunta'], k=5)
           ids_recuperados = [r['id'] for r in resultados]

           hits = len(set(ids_recuperados) & set(item['docs_relevantes']))
           recall = hits / len(item['docs_relevantes'])
           recalls.append(recall)

       return np.mean(recalls)

   # Compara: búsqueda_basica vs multi_query vs hyde vs hybrid
   ```

2. **Optimización de parámetros:**
   - HNSW: Prueba M=[8, 16, 32], ef_construction=[100, 200, 400]
   - Hybrid: Prueba lambda=[0.5, 0.6, 0.7, 0.8, 0.9]
   - Grafica precision vs latencia

**Recursos:**

- 📖 Sección 9 de este documento (Técnicas Avanzadas)
- 📄 [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- 🔗 [Advanced RAG Patterns](https://blog.langchain.dev/deconstructing-rag/)

#### **Nivel 4: Optimización y Producción (3-6 semanas)**

**Objetivos:**

- Optimizar rendimiento (latencia, throughput)
- Implementar monitoreo y observabilidad
- Preparar para despliegue en producción

**Proyectos avanzados:**

1. **Benchmarking sistemático:**

   ```python
   import time
   import statistics

   def benchmark_latencia(metodo, queries, repeticiones=10):
       latencias = []

       for query in queries:
           tiempos = []
           for _ in range(repeticiones):
               inicio = time.time()
               _ = metodo(query)
               fin = time.time()
               tiempos.append((fin - inicio) * 1000)  # ms

           latencias.append(statistics.median(tiempos))

       return {
           "p50": statistics.median(latencias),
           "p95": sorted(latencias)[int(0.95 * len(latencias))],
           "p99": sorted(latencias)[int(0.99 * len(latencias))],
           "mean": statistics.mean(latencias)
       }

   # Compara diferentes configuraciones
   configs = [
       ("Básico", busqueda_basica),
       ("Multi-Query", multi_query_search),
       ("Hybrid", hybrid_search)
   ]

   for nombre, metodo in configs:
       stats = benchmark_latencia(metodo, test_queries)
       print(f"{nombre}: P50={stats['p50']:.1f}ms, P95={stats['p95']:.1f}ms")
   ```

2. **Implementar caching:**

   ```python
   from functools import lru_cache
   import hashlib

   @lru_cache(maxsize=1000)
   def buscar_con_cache(pregunta_hash, k):
       # Cache de búsquedas frecuentes
       return buscar(pregunta_hash, k)

   def buscar_inteligente(pregunta, k=5):
       # Hash de la pregunta normalizada
       pregunta_norm = pregunta.lower().strip()
       pregunta_hash = hashlib.md5(pregunta_norm.encode()).hexdigest()

       return buscar_con_cache(pregunta_hash, k)

   # Mide el hit rate del cache después de 1000 queries
   ```

3. **Monitoreo y logging:**

   ```python
   import logging
   import json
   from datetime import datetime

   # Configurar logging estructurado
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def buscar_con_telemetria(pregunta, k=5):
       inicio = time.time()

       try:
           resultados = buscar(pregunta, k)
           latencia = (time.time() - inicio) * 1000

           # Log estructurado
           logger.info(json.dumps({
               "evento": "busqueda_exitosa",
               "timestamp": datetime.now().isoformat(),
               "pregunta_length": len(pregunta),
               "num_resultados": len(resultados),
               "latencia_ms": latencia,
               "top_score": resultados[0][0] if resultados else 0
           }))

           return resultados

       except Exception as e:
           logger.error(json.dumps({
               "evento": "busqueda_fallida",
               "timestamp": datetime.now().isoformat(),
               "error": str(e)
           }))
           raise

   # Analiza los logs para identificar patrones
   ```

**Desafío final: Sistema RAG de producción**

Construye un sistema completo con:

- ✅ API REST (FastAPI)
- ✅ Autenticación (API keys)
- ✅ Rate limiting
- ✅ Caching
- ✅ Logging estructurado
- ✅ Métricas (Prometheus)
- ✅ Tests unitarios e integración
- ✅ Docker Compose para deployment local
- ✅ CI/CD básico (GitHub Actions)

**Recursos:**

- 📖 Secciones 10-11 de este documento (Optimización y Cloud)
- 🔗 [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- 🔗 [Production ML Systems](https://madewithml.com/)

#### **Proyecto Capstone: RAG Especializado**

Aplica todo lo aprendido para construir un sistema RAG en un dominio específico de tu interés:

**Ideas de proyectos:**

1. **Asistente académico personalizado**

   - Indexa tus apuntes, papers, libros
   - Ayuda a estudiar con preguntas y respuestas
   - Genera resúmenes y flashcards

2. **Buscador de código interno**

   - Indexa repositorios de código
   - Búsqueda semántica de funciones
   - Explica código con contexto

3. **Asistente de documentación técnica**

   - Indexa docs de frameworks/libraries
   - Responde preguntas de implementación
   - Sugiere ejemplos relevantes

4. **Analizador de contenido (Reddit, Twitter, blogs)**
   - Indexa posts/artículos sobre un tema
   - Detecta tendencias
   - Resume discusiones

**Rúbrica de evaluación autónoma:**

| Criterio          | Básico                      | Intermedio                      | Avanzado                 |
| ----------------- | --------------------------- | ------------------------------- | ------------------------ |
| **Funcionalidad** | RAG básico funciona         | Multiple técnicas implementadas | Sistema producción-ready |
| **Rendimiento**   | Funciona en dataset pequeño | Optimizado para dataset mediano | Escala a 100k+ docs      |
| **Código**        | Scripts funcionales         | Código estructurado             | Tests, CI/CD, docs       |
| **Evaluación**    | Pruebas manuales            | Benchmark básico                | Métricas completas       |
| **Innovación**    | Implementación estándar     | Adaptación al dominio           | Técnicas novedosas       |

---

**Comunidad y recursos adicionales:**

- 💬 [Qdrant Discord](https://discord.gg/qdrant)
- 💬 [LangChain Discord](https://discord.gg/langchain)
- 📚 [Hugging Face Forums](https://discuss.huggingface.co/)
- 📺 [RAG YouTube Tutorials](https://www.youtube.com/@LangChain)
- 📖 [Este proyecto GitHub](https://github.com/aleistrying/basic-py-rag) - código completo funcionando

**Tip final:** La mejor forma de aprender es **construyendo**. Empieza simple, itera, mide resultados, y mejora progresivamente. No intentes implementar todo a la vez.

### 14.2 Para Desarrolladores

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

### 14.3 Para Arquitectos de Sistemas

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

### 14.4 Recomendación Final: ¿Qué Base de Datos Elegir?

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
