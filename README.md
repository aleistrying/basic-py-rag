# 🚀 Sistema RAG con Bases de Datos Vectoriales

**Sistema RAG en español que compara Qdrant vs PostgreSQL para búsqueda semántica.**

Perfecto para cursos académicos o para aprender sobre bases de datos vectoriales. Procesa PDFs en español y proporciona búsqueda semántica con respuestas generadas por IA.

> 📚 **¿Nuevo en vectores y RAG?** Lee primero: **[TEORIA.md](./TEORIA.md)** - Guía completa desde fundamentos hasta implementación avanzada.

## Inicio Rápido (5 minutos)

### 1. Iniciar el Sistema

```bash
git clone https://github.com/aleistrying/basic-py-rag.git
cd base-de-datos-avanzadas

# Iniciar bases de datos
docker compose up -d

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Procesar tus Documentos

```bash
# Procesa PDFs y configura ambas bases de datos
python scripts/main_pipeline.py --clear
```

> 🔍 **Detalles del pipeline:** Ver [Sección 9: Pipeline de Procesamiento](./TEORIA.md#9-de-pdf-a-respuesta-inteligente-en-6-pasos) en TEORIA.md

### 4. Probar que Funciona

```bash
# Búsqueda básica
curl "http://localhost:8080/ask?q=vectores&backend=qdrant"

# Respuesta generada por IA (requiere Ollama - ver abajo)
curl "http://localhost:8080/ai?q=¿Qué son las bases de datos vectoriales?"

# Comparar ambas bases de datos
curl "http://localhost:8080/compare?q=bases de datos vectoriales"
```

## 🎓 Demostración para Aulas

```bash
# Demo interactiva paso a paso:
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales
```

> 📖 **Teoría detrás de la demo:** [Sección 2: El Arte de Convertir Palabras en Números](./TEORIA.md#2-el-arte-de-convertir-palabras-en-números) + [Sección 3: Búsqueda en 768 Dimensiones](./TEORIA.md#3-encontrar-agujas-en-pajares-de-768-dimensiones)

## ✨ Características Principales

- 📄 **Procesamiento PDFs** → 🔍 **Búsqueda Semántica** → 🤖 **Respuestas IA**
- ⚖️ **Comparación Qdrant vs PostgreSQL** lado a lado
- 🎓 **Demos interactivas** para enseñanza

> 🏗️ **Arquitectura completa:** [Sección 8: Anatomía del Sistema](./TEORIA.md#8-anatomía-de-nuestro-sistema-rag)

## Características Avanzadas

### 🔧 Filtrado por Metadatos

```bash
# Ejemplos de filtros disponibles:
curl "http://localhost:8080/ask?q=proyecto&document_type=pdf"
curl "http://localhost:8080/ask?q=evaluacion&section=objetivos"
curl "http://localhost:8080/ask?q=vectores&document_type=pdf&section=objetivos"
```

> 📋 **Lista completa de filtros:** `http://localhost:8080/filters/examples`

### 🤖 Búsqueda con IA (Opcional)

```bash
# 1. Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh
# 2. Descargar modelo
ollama pull phi3:mini
# 3. Probar
curl "http://localhost:8080/ai?q=¿Cuáles son los objetivos del curso?"
```

> 🧠 **Cómo funciona RAG:** [Sección 5: RAG - Cuando la IA Busca Antes de Responder](./TEORIA.md#5-rag-cuando-la-ia-busca-antes-de-responder)

## Archivos Principales

```
├── scripts/main_pipeline.py    # Pipeline principal de procesamiento
├── app/main.py                 # Servidor API
├── docker-compose.yml          # Configuración de bases de datos
├── requirements.txt            # Dependencias
└── data/raw/                   # Coloca tus PDFs aquí
```

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

**Problemas comunes:**

- 📄 **Documentos no se procesan:** Verifica PDFs en `data/raw/`, usa `--force`
- 🚀 **API no inicia:** `docker compose ps` y puerto 8080 libre
- 🤖 **Ollama no responde:** Verificar instalación y modelo descargado
- 🔍 **Búsqueda pobre:** Prueba `&backend=pgvector` o filtros

**Logs detallados:**

```bash
docker compose logs -f
python scripts/main_pipeline.py --stats
```

> 🛠️ **Guía completa de optimización:** [Sección 13: Lo que Realmente Importa](./TEORIA.md#13-sabiduría-destilada-lo-que-realmente-importa)

---

## 🎯 Objetivo

**5 minutos → Sistema RAG funcional**. Todo lo demás son mejoras opcionales.

> 📚 **Para entender REALMENTE cómo funciona:** Lee [TEORIA.md](./TEORIA.md) - desde fundamentos hasta casos billonarios reales.

---
