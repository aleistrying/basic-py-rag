# 🚀 Sistema RAG con Bases de Datos Vectoriales

**Sistema RAG en español que compara Qdrant vs PostgreSQL para búsqueda semántica.**

Perfecto para cursos académicos o para aprender sobre bases de datos vectoriales. Procesa PDFs en español y proporciona búsqueda semántica con respuestas generadas por IA.

> 📚 **¿Nuevo en vectores y RAG?** Lee primero: **[TEORIA.md](./TEORIA.md)** - Guía completa desde fundamentos hasta implementación avanzada.

## Inicio Rápido (5 minutos)

Choose your preferred approach:

### 🐳 Opción A: Solo Docker + Interfaz Web (Recomendado)

**¿Sin Python local? ¡Perfecto! Todo funciona desde el navegador.**

```bash
git clone https://github.com/aleistrying/basic-py-rag.git
cd base-de-datos-avanzadas

# Iniciar TODO el sistema (bases de datos + API + interfaz)
docker compose up -d
```

**Usar desde el navegador:**

- 🎓 **Demo Pipeline Completo:** http://localhost:8080/demo/pipeline
- 🔍 **Búsqueda Simple:** http://localhost:8080/
- 📊 **Comparación BD:** http://localhost:8080/compare
- 🚀 **Gestión Pipeline:** http://localhost:8080/pipeline
- 🤖 **Chat IA:** http://localhost:8080/ai
- 📁 **Gestión de Archivos:** Disponible desde la página principal

> 📄 **Subir documentos:** Usa el botón "📁 Subir Archivo" en la página principal, o coloca archivos en `data/raw/`
> 🗂️ **Gestionar archivos:** Usa el botón "📂 Ver Archivos" para listar, visualizar y eliminar archivos subidos

### 🔧 Opción B: Docker + Scripts Python + API Directa

**¿Developer que quiere control total? Esta es tu opción.**

```bash
git clone https://github.com/aleistrying/basic-py-rag.git
cd base-de-datos-avanzadas

# 1. Bases de datos + API
docker compose up -d

# 2. Instalar dependencias Python (local) segurate de tener Python 3.11+
uv venv

source venv/bin/activate

uv pip install -r requirements.txt

# 3. Procesar documentos con scripts
python scripts/main_pipeline.py --clear
```

> 🔍 **Detalles del pipeline:** Ver [Sección 9: Pipeline de Procesamiento](./TEORIA.md#9-de-pdf-a-respuesta-inteligente-en-6-pasos) en TEORIA.md

### 🧪 Probar que Funciona con curl

```bash
# Búsqueda básica en JSON
curl "http://localhost:8080/ask?q=vectores&backend=qdrant&format=json"

# Respuesta generada por IA en JSON
curl "http://localhost:8080/ai?q=¿Qué%20son%20las%20bases%20de%20datos%20vectoriales?&format=json"

# Comparar ambas bases de datos en JSON
curl "http://localhost:8080/compare?q=bases%20de%20datos%20vectoriales&format=json"

# === GESTIÓN DE ARCHIVOS ===

# Subir archivo (PDF, TXT, MD, YAML, YML - máx 10MB)
curl -X POST -F "file=@documento.pdf" "http://localhost:8080/upload"

# Listar todos los archivos subidos (con detalles)
curl "http://localhost:8080/upload/list"

# Eliminar archivo específico
curl -X DELETE "http://localhost:8080/upload/documento.pdf"
```

## 🎓 Demostración para Aulas

```bash
# Demo interactiva paso a paso:
http://localhost:8080/demo/pipeline?q=bases%20de%20datos%20vectoriales
```

> 📖 **Teoría detrás de la demo:** [Sección 2: El Arte de Convertir Palabras en Números](./TEORIA.md#2-el-arte-de-convertir-palabras-en-números) + [Sección 3: Búsqueda en 768 Dimensiones](./TEORIA.md#3-encontrar-agujas-en-pajares-de-768-dimensiones)

## 🔌 API Endpoints Completos

### 📋 Obtener información del sistema

```bash
# Estado general del sistema
curl "http://localhost:8080/"
# Respuesta JSON:
{
  "message": "RAG Vector Search API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {
    "search": "/ask",
    "ai_chat": "/ai",
    "compare": "/compare",
    "demo": "/demo/pipeline"
  }
}
```

### 🔍 Búsqueda vectorial básica

```bash
# Búsqueda en Qdrant (JSON)
curl "http://localhost:8080/ask?q=bases%20de%20datos&backend=qdrant&format=json"

# Búsqueda en PostgreSQL (JSON)
curl "http://localhost:8080/ask?q=bases%20de%20datos&backend=pgvector&format=json"

# Respuesta JSON:
{
  "query": "bases de datos",
  "backend": "qdrant",
  "results": [
    {
      "content": "Las bases de datos vectoriales permiten...",
      "score": 0.89,
      "metadata": {
        "source": "documento.pdf",
        "page": 15,
        "section": "introduccion"
      }
    }
  ],
  "total_found": 5,
  "time_ms": 45
}
```

### ⚖️ Comparación entre bases de datos

```bash
curl "http://localhost:8080/compare?q=vectores&limit=3&format=json"

# Respuesta JSON:
{
  "query": "vectores",
  "comparison": {
    "qdrant": {
      "results": [...],
      "time_ms": 35,
      "backend": "qdrant"
    },
    "pgvector": {
      "results": [...],
      "time_ms": 67,
      "backend": "pgvector"
    }
  },
  "performance": {
    "winner": "qdrant",
    "time_difference_ms": 32
  }
}
```

### 🤖 Chat con IA (RAG)

```bash
curl "http://localhost:8080/ai?q=¿Qué%20es%20pgvector?&format=json"

# Respuesta JSON:
{
  "query": "¿Qué es pgvector?",
  "answer": "pgvector es una extensión de PostgreSQL que permite...",
  "sources": [
    {
      "content": "PostgreSQL con pgvector permite almacenar vectores...",
      "source": "manual_pgvector.pdf",
      "page": 12,
      "relevance": 0.92
    }
  ],
  "model": "phi3:mini",
  "time_ms": 1250,
  "tokens_used": 456
}
```

### 📊 Pipeline y gestión de documentos

```bash
# Estado del pipeline
curl "http://localhost:8080/pipeline/stats"

# === GESTIÓN DE ARCHIVOS ===

# Subir archivo (POST con FormData)
curl -X POST "http://localhost:8080/upload" -F "file=@documento.pdf"

# Listar archivos subidos con detalles completos
curl "http://localhost:8080/upload/list"

# Eliminar archivo específico
curl -X DELETE "http://localhost:8080/upload/documento.pdf"

# Eliminar archivo específico
curl -X DELETE "http://localhost:8080/upload/documento.pdf"

# Respuesta JSON:
{
  "success": true,
  "filename": "documento.pdf",
  "file_size": 1048576,
  "message": "Archivo guardado exitosamente...",
  "saved_path": "/app/data/raw/documento.pdf"
}

# Listar archivos:
{
  "success": true,
  "files": [
    {
      "filename": "documento.pdf",
      "size": 1048576,
      "size_mb": 1.0,
      "created": 1763604424.6123478,
      "modified": 1763604424.6123478,
      "extension": ".pdf"
    }
  ],
  "total": 1
}

# Eliminar archivo:
{
  "success": true,
  "message": "Archivo documento.pdf eliminado exitosamente",
  "filename": "documento.pdf",
  "size_deleted": 1048576
}
```

### 🔧 Filtros avanzados

```bash
# Filtrar por tipo de documento (JSON)
curl "http://localhost:8080/ask?q=proyecto&document_type=pdf&format=json"

# Filtrar por página específica (JSON)
curl "http://localhost:8080/ask?q=objetivos&page=1&format=json"

# Múltiples filtros (JSON)
curl "http://localhost:8080/ask?q=evaluacion&document_type=pdf&contains=proyecto&limit=5&format=json"

# Filtros disponibles: document_type, page, contains
# Ejemplos de uso:
# - document_type: pdf, txt, md, yaml
# - page: número de página (solo PDFs)
# - contains: texto que debe estar presente
```

### 📈 Estadísticas y métricas

```bash
# Estadísticas generales
curl "http://localhost:8080/pipeline/stats"

# Respuesta JSON:
{
  "documents": {
    "total": 15,
    "total_chunks": 1204,
    "avg_chunk_size": 245
  },
  "vectors": {
    "qdrant_collection": "course_docs_clean",
    "qdrant_count": 1204,
    "pgvector_table": "vectors",
    "pgvector_count": 1204
  },
  "performance": {
    "avg_search_time_qdrant": "35ms",
    "avg_search_time_pgvector": "67ms",
    "total_searches": 2847
  }
}
```

### 🎯 Demo paso a paso

```bash
# Pipeline completo interactivo
curl "http://localhost:8080/demo/pipeline?q=machine%20learning&model=phi3:mini"

# Solo embeddings
curl "http://localhost:8080/demo/embeddings?text=Hola%20mundo"

# Solo búsqueda
curl "http://localhost:8080/demo/search?q=vectores&backend=both"
```

## ✨ Características Principales

- 📄 **Procesamiento PDFs** → 🔍 **Búsqueda Semántica** → 🤖 **Respuestas IA**
- ⚖️ **Comparación Qdrant vs PostgreSQL** lado a lado
- 🎓 **Demos interactivas** para enseñanza
- 📁 **Gestión completa de archivos** con interfaz web intuitiva
- 🐳 **100% Docker** - no requiere Python local
- 🔌 **API REST completa** con documentación JSON
- 🎯 **Upload drag & drop** con validación y feedback en tiempo real

> 🏗️ **Arquitectura completa:** [Sección 8: Anatomía del Sistema](./TEORIA.md#8-anatomía-de-nuestro-sistema-rag)

## 📁 Gestión de Archivos - Guía Completa

### 🖱️ **Interfaz Web (Recomendado)**

**Flujo completo desde el navegador:**

1. **🌐 Ir a:** http://localhost:8080/
2. **📤 Subir:** Click en "📁 Subir Archivo" → Seleccionar archivo → ✅ Confirmación automática
3. **👀 Ver archivos:** Click en "📂 Ver Archivos" → Modal con lista completa
4. **🗑️ Eliminar:** Click en 🗑️ junto al archivo → Confirmación → ✅ Eliminado

**Características de la interfaz:**

- ✅ **Validación en tiempo real:** PDF, TXT, MD, YAML, YML (máx 10MB)
- 📊 **Información detallada:** Nombre, tamaño, tipo, fecha de modificación
- 🎯 **Feedback visual:** Progreso de subida, mensajes de éxito/error
- 🔄 **Actualización automática:** Lista se actualiza tras cada operación

### 🔧 **API REST (Desarrolladores)**

```bash
# 📤 SUBIR archivo
curl -X POST "http://localhost:8080/upload" \\
  -F "file=@documento.pdf"

# 📋 LISTAR archivos (con detalles)
curl "http://localhost:8080/upload/list"

# 🗑️ ELIMINAR archivo específico
curl -X DELETE "http://localhost:8080/upload/documento.pdf"

# 📊 Respuesta de listado:
{
  "success": true,
  "files": [
    {
      "filename": "documento.pdf",
      "size": 1048576,
      "size_mb": 1.0,
      "created": 1763604424.61,
      "modified": 1763604424.61,
      "extension": ".pdf"
    }
  ],
  "total": 1
}
```

### 🔄 **Flujo Completo: Subir → Procesar → Buscar**

```bash
# 1️⃣ Subir documento
curl -X POST "http://localhost:8080/upload" -F "file=@mi_documento.pdf"

# 2️⃣ Procesar y añadir a base vectorial
docker compose exec app python3 scripts/main_pipeline.py

# 3️⃣ ¡Buscar en el documento!
curl "http://localhost:8080/ai?q=¿De qué trata mi documento?&format=json"
```

> 💡 **Tip:** Los archivos subidos van a `/app/data/raw/` y se procesan automáticamente cuando ejecutas el pipeline.

## Características Avanzadas

### 🤖 Búsqueda con IA (Ollama integrado)

**El sistema incluye Ollama pre-configurado** - solo necesitas descargar un modelo:

```bash
# Opción 1: Desde el contenedor
docker compose exec ollama ollama pull phi3:mini

# Opción 2: Instalación local (opcional)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# Probar chat IA
curl "http://localhost:8080/ai?q=¿Cuáles son los objetivos del curso?"
```

### 🔧 Pipeline de documentos desde Docker

**Sin Python local - todo desde contenedores:**

```bash
# Procesar documentos desde el contenedor
docker compose exec app python3 scripts/main_pipeline.py --clear

# Subir documentos vía API
curl -X POST "http://localhost:8080/pipeline/upload" -F "file=@documento.pdf"

# Ver estado del pipeline
curl "http://localhost:8080/pipeline/status"
```

> 🧠 **Cómo funciona RAG:** [Sección 5: RAG - Cuando la IA Busca Antes de Responder](./TEORIA.md#5-rag-cuando-la-ia-busca-antes-de-responder)

## Archivos Principales

```
├── docker-compose.yml          # 🐳 Todo el sistema (bases de datos + API + Ollama)
├── app/                        # 🚀 API REST y demos interactivos
│   ├── main.py                 #     Servidor principal
│   ├── demo_pipeline.py        #     Pipeline de demostración
│   └── embedding_service.py    #     Servicio de embeddings (local + OpenAI)
├── scripts/                    # 🔧 Scripts de procesamiento (opcional)
│   ├── main_pipeline.py        #     Pipeline principal para documentos
│   └── embedding_database.py   #     Gestión de bases de datos vectoriales
├── requirements.txt            # 📦 Dependencias Python (solo si usas scripts locales)
├── .env.example               # ⚙️ Configuración (embeddings local/OpenAI)
├── EMBEDDING_GUIDE.md         # 📖 Guía completa de embeddings
└── data/raw/                  # 📄 Coloca tus PDFs aquí
```

## 🚀 Dos Workflows Completos

### Workflow A: Solo Docker (Recomendado)

1. `docker compose up -d` → Sistema completo funcionando
2. http://localhost:8080 → Interfaz web para subir documentos
3. http://localhost:8080/demo/pipeline → Ver demo completo
4. Usar API endpoints con `curl` para integrar con otros sistemas

### Workflow B: Docker + Scripts Python

1. `docker compose up -d` → Bases de datos + API
2. `pip install -r requirements.txt` → Instalar dependencias locales
3. `python scripts/main_pipeline.py` → Procesar documentos con scripts
4. Usar tanto interfaz web como scripts según necesidad

## 📈 Rendimiento

- ⚡ **Paralelo + Batch processing** para velocidad
- 💾 **Dual storage:** Qdrant (velocidad) + PostgreSQL (SQL)

> 🔧 **Optimizaciones detalladas:** [Sección 10: Optimizar para el Mundo Real](./TEORIA.md#10-optimizar-para-el-mundo-real)

## 🎯 Casos de Uso

- 🔬 **Investigación académica** y análisis de documentos
- 🤖 **Asistentes IA** con contexto empresarial
- 🎓 **Enseñanza** de conceptos vectoriales

> 💼 **Aplicaciones empresariales reales:** [Sección 12: Casos que Cambian Industrias](./TEORIA.md#12-casos-de-uso-que-cambian-industrias)

## ⚖️ Qdrant vs PostgreSQL

| Aspecto              | Qdrant | PostgreSQL+pgvector |
| -------------------- | ------ | ------------------- |
| 🚀 Velocidad         | ⭐⭐⭐ | ⭐⭐                |
| 🔧 Integración SQL   | ❌     | ✅                  |
| 📚 Curva aprendizaje | Nueva  | Familiar            |

> 📊 **Comparación detallada:** [Sección 7: La Gran Comparación](./TEORIA.md#7-qdrant-vs-postgresql-la-gran-comparación) + [Guía Práctica](./TEORIA.md#36-guía-práctica-algoritmos-y-distancias-por-base-de-datos)

## 🔧 Solución de Problemas

### Workflow Docker-only

- 🐳 **Sistema no inicia:** `docker compose ps` - verificar que todos los contenedores estén corriendo
- 🌐 **API no responde:** Verificar puerto 8080 libre: `netstat -tulpn | grep 8080`
- 📄 **No ve documentos:** Colocar PDFs en `data/raw/` o usar interfaz web
- 🔍 **Búsqueda vacía:** Procesar documentos primero vía interfaz web

### Workflow con Scripts Python

- 🐍 **Dependencias faltan:** `pip install -r requirements.txt` en entorno virtual
- 📄 **Documentos no se procesan:** Verificar PDFs en `data/raw/`, usar `--force`
- 🤖 **Ollama no responde:** `docker compose logs ollama` para ver logs del modelo

### Problemas Comunes (ambos workflows)

- 🔍 **Búsqueda pobre:** Probar diferentes backends: `&backend=pgvector` vs `&backend=qdrant`
- 💾 **Error base de datos:** Reiniciar contenedores: `docker compose restart`
- 🌐 **CORS errors:** API está en puerto 8080, verificar configuración

**Logs detallados:**

```bash
# Ver todos los servicios
docker compose logs -f

# Ver solo la API
docker compose logs -f app

# Ver base de datos específica
docker compose logs -f qdrant
docker compose logs -f pgvector_db

# Estadísticas del sistema
curl "http://localhost:8080/stats"
```

**Comandos útiles:**

```bash
# Reiniciar todo
docker compose restart

# Reconstruir si hay cambios de código
docker compose up -d --build

# Ver uso de recursos
docker compose exec app htop

# Verificar conexiones BD
docker compose exec pgvector_db psql -U pguser -d vectordb -c "SELECT COUNT(*) FROM vectors;"
```

> 🛠️ **Guía completa de optimización:** [Sección 13: Lo que Realmente Importa](./TEORIA.md#13-sabiduría-destilada-lo-que-realmente-importa)

---

## 🎯 Objetivo

**🐳 Docker-only: 1 minuto → Sistema RAG completo funcionando**

**🔧 Con scripts: 5 minutos → Control total del pipeline**

Todo lo demás son características avanzadas opcionales.

> 📚 **Para entender REALMENTE cómo funciona:** Lee [TEORIA.md](./TEORIA.md) - desde fundamentos hasta casos billonarios reales.
>
> 🔧 **Para embeddings avanzados:** Ver [EMBEDDING_GUIDE.md](./EMBEDDING_GUIDE.md) - local vs OpenAI, offline, configuración.

---
