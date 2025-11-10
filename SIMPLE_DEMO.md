````markdown
# 🚀 UNIFIED RAG PIPELINE - Consolidated & Enhanced

## 📋 What You Have

Your **consolidated RAG system** compares **Qdrant** vs **PostgreSQL** for Spanish academic queries with enhanced PDF extraction, advanced metadata filtering, and unified processing pipeline.

### **🔥 Quick Demo Commands (Unified Pipeline):**

```bash
# 1. Start databases
docker compose up -d

# 2. Install dependencies (if needed)
pip3 install --user numpy sentence-transformers fastapi uvicorn psycopg2-binary qdrant-client pdfplumber

# 3. Run unified pipeline (enhanced extraction + memory-safe processing)
python3 scripts/main_pipeline.py --clear

# 4. Start API (shows "Uvicorn running on...")
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# 5. Test enhanced queries
curl "http://localhost:8080/ask?q=objetivos del curso&backend=qdrant"
```

## 🎓 **NEW: Manual Demonstration for Classroom**

### **🌟 Main Demo URL (Perfect for Zoom/Projection):**

```
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales&backend=qdrant
```

### **🔧 Step-by-Step Manual Process:**

```bash
# Show embedding process
http://localhost:8080/manual/embed?q=bases%20de%20datos%20vectoriales

# Show search process
http://localhost:8080/manual/search?q=bases%20de%20datos%20vectoriales&backend=qdrant&k=3

# Show filter examples
http://localhost:8080/filters/examples
```

## 🎯 **NEW: Enhanced Query Examples with Metadata Filtering**

### Basic Search with Filters:

```bash
# Filter by document type
curl "http://localhost:8080/ask?q=vectores&backend=qdrant&document_type=pdf"

# Filter by course section
curl "http://localhost:8080/ask?q=evaluacion&backend=qdrant&section=objetivos"

# Filter by topic
curl "http://localhost:8080/ask?q=bases&backend=qdrant&topic=vectorial"

# Multiple filters combined
curl "http://localhost:8080/ask?q=proyecto&backend=qdrant&section=evaluacion&document_type=pdf&contains=NoSQL"
```

### AI Chat with Filters:

```bash
# AI with metadata filtering
curl "http://localhost:8080/ai?q=¿Qué son las bases vectoriales?&backend=qdrant&k=3&section=objetivos&topic=vectorial"
```

## 📊 **Available Metadata Filters**

| Filter          | Description       | Examples                                | Usage                |
| --------------- | ----------------- | --------------------------------------- | -------------------- |
| `document_type` | File type         | `pdf`, `txt`, `md`                      | `&document_type=pdf` |
| `section`       | Course section    | `objetivos`, `cronograma`, `evaluacion` | `&section=objetivos` |
| `topic`         | Subject topic     | `nosql`, `vectorial`, `sql`, `mongodb`  | `&topic=vectorial`   |
| `page`          | PDF page number   | `1`, `5`, `10`                          | `&page=5`            |
| `contains`      | Must contain text | `NoSQL`, `vector`, etc.                 | `&contains=NoSQL`    |

### **🔍 Filter Examples for Different Use Cases:**

```bash
# Find evaluation info only in PDFs
/ask?q=evaluacion&document_type=pdf&section=evaluacion

# Find vector database info in course objectives
/ask?q=vectoriales&section=objetivos&topic=vectorial

# Find projects mentioning NoSQL
/ask?q=proyecto&contains=NoSQL&section=proyectos

# Find specific page content
/ask?q=cronograma&page=3&document_type=pdf
```

## 🎯 **Demo Scenarios**

### Manual Process Demo (Great for Teaching):

```bash
# Complete step-by-step explanation
curl "http://localhost:8080/manual/demo?q=bases de datos vectoriales"

# Just embedding process
curl "http://localhost:8080/manual/embed?q=bases de datos vectoriales"

# Just search process
curl "http://localhost:8080/manual/search?q=bases de datos vectoriales&backend=qdrant&k=3"
```

### Performance Comparison:

```bash
# Compare backends for Spanish query
curl "http://localhost:8080/compare?q=bases de datos vectoriales&k=3" | python3 -m json.tool
```

### AI Chat Demo (if you have Ollama):

```bash
# Ask AI with RAG context and filters
curl "http://localhost:8080/ai?q=¿Cuándo se entrega el proyecto 1?&backend=qdrant&k=3&section=cronograma" | python3 -m json.tool
```

### Filter Testing:

```bash
# Test different filter combinations
curl "http://localhost:8080/ask?q=NoSQL&backend=qdrant&topic=nosql"
curl "http://localhost:8080/ask?q=evaluación&backend=qdrant&section=evaluacion&document_type=pdf"
curl "http://localhost:8080/ask?q=vectoriales&backend=qdrant&contains=similitud"
```

## 📁 **Essential Files (Consolidated Architecture):**

```
📁 Unified Structure:
├── app/
│   ├── main.py              # 🌐 Enhanced API with metadata filtering
│   ├── rag.py               # 🤖 RAG logic with filter support
│   ├── qdrant_backend.py    # 🔍 Qdrant with advanced filtering
│   └── pgvector_backend.py  # 🐘 PostgreSQL with SQL filtering
├── scripts/
│   ├── main_pipeline.py     # 🎯 MAIN: Complete unified pipeline
│   ├── pdf_processing.py    # 📄 PDF extraction (multi-library + tables)
│   ├── embedding_database.py # 🤖 Embeddings + database operations
│   └── chunker.py           # ✂️  Text chunking
├── archive/                 # 📦 Legacy/test files (moved for cleanup)
├── DEMO_MANUAL_VECTORES.md  # 🎓 Spanish classroom guide
└── docker-compose.yml       # 🐳 Services setup
```

## 🔧 **What the Unified System Does:**

1. **Consolidated Processing** → Single entry point (`main_pipeline.py`) for all operations
2. **Enhanced PDF Extraction** → Multi-library support (pdfplumber, PyMuPDF, unstructured)
3. **Memory-Safe Processing** → Handles massive documents (1322+ pages) efficiently
4. **Advanced Metadata Filtering** → Filter by document type, section, topic, page, or content
5. **Manual Process Demonstration** → Step-by-step explanation for teaching
6. **Enhanced Spanish Support** → Better query expansion and processing
7. **Performance Comparison** → Side-by-side backend comparison

## 🚀 **Pipeline Usage (New Unified Commands):**

```bash
# Enhanced extraction (default - recommended)
python3 scripts/main_pipeline.py --clear

# Memory-safe mode for large documents
python3 scripts/main_pipeline.py --memory-safe --clear

# Force re-processing
python3 scripts/main_pipeline.py --force

# Basic extraction (legacy compatibility)
python3 scripts/main_pipeline.py --basic

# Show configuration
python3 scripts/main_pipeline.py --config

# Show statistics
python3 scripts/main_pipeline.py --stats
```

## 🔧 **What the Enhanced System Does:**

1. **Advanced Metadata Filtering** → Filter by document type, section, topic, page, or content
2. **Manual Process Demonstration** → Step-by-step explanation for teaching
3. **Enhanced Spanish Support** → Better query expansion and processing
4. **Classroom-Ready URLs** → Browser-friendly endpoints for projection
5. **Performance Comparison** → Side-by-side backend comparison
6. **Smart Reranking** → MMR algorithm for better result diversity

## ⚡ **Most Important New Features:**

- ✅ **Metadata Filtering:** Query specific document types, sections, or topics
- ✅ **Manual Demo Mode:** Perfect for classroom explanation of vector search
- ✅ **Enhanced Query Processing:** Better Spanish query expansion
- ✅ **Browser-Friendly:** All endpoints optimized for web demonstration
- ✅ **Filter Combinations:** Mix multiple filters for precise searches

## 🎪 **Live Demo Steps for Class:**

### **1. Manual Process Demo (15 min):**

```bash
# Start with complete demo
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales

# Then show step by step:
http://localhost:8080/manual/embed?q=bases%20de%20datos%20vectoriales
http://localhost:8080/manual/search?q=bases%20de%20datos%20vectoriales&backend=qdrant&k=3
```

### **2. Filter Examples (10 min):**

```bash
# Show available filters
http://localhost:8080/filters/examples

# Test some filters live
http://localhost:8080/ask?q=vectores&section=objetivos&document_type=pdf
http://localhost:8080/ask?q=evaluacion&section=evaluacion&contains=proyecto
```

### **3. Backend Comparison (5 min):**

```bash
# Compare search engines
http://localhost:8080/compare?q=bases%20de%20datos%20vectoriales&k=3
```

### **4. Interactive Testing (10 min):**

- Let students suggest queries
- Test with different filters
- Show semantic vs. keyword search differences

## 🔍 **Qdrant Filter Query Format**

The `where` parameter in Qdrant supports these filter structures:

### **Simple Filters:**

```python
# Document type filter
{"must": [{"key": "document_type", "match": {"value": "pdf"}}]}

# Section filter
{"must": [{"key": "section", "match": {"value": "objetivos"}}]}

# Numeric filter (page)
{"must": [{"key": "page", "range": {"gte": 1, "lte": 5}}]}
```

### **Combined Filters:**

```python
# Multiple conditions
{
    "must": [
        {"key": "document_type", "match": {"value": "pdf"}},
        {"key": "section", "match": {"value": "objetivos"}},
        {"key": "topic", "match": {"value": "vectorial"}}
    ]
}
```

### **Text Search Filters:**

```python
# Content must contain specific text
{"must": [{"key": "content", "match": {"text": "NoSQL"}}]}
```

## 💡 **PostgreSQL Filter Examples**

The enhanced pgvector backend supports SQL-based filtering:

```sql
-- Filter by document type
WHERE source_path LIKE '%.pdf'

-- Filter by section (in metadata or content)
WHERE (content ILIKE '%objetivos%' OR metadata->>'section' ILIKE '%objetivos%')

-- Filter by page number
WHERE page = 5

-- Combined filters
WHERE source_path LIKE '%.pdf'
  AND page BETWEEN 1 AND 10
  AND content ILIKE '%vectorial%'
```

## 🚀 **Quick URLs for Copy-Paste in Class:**

```bash
# Main API info
http://localhost:8080/

# Complete manual demo (MOST IMPORTANT)
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales&backend=qdrant

# Filter examples
http://localhost:8080/filters/examples

# Simple search
http://localhost:8080/ask?q=vectores&backend=qdrant&k=3

# Filtered search
http://localhost:8080/ask?q=vectores&backend=qdrant&k=3&section=objetivos&document_type=pdf

# Backend comparison
http://localhost:8080/compare?q=bases%20de%20datos%20vectoriales&k=3

# AI search with filters
http://localhost:8080/ai?q=¿Qué%20son%20las%20bases%20vectoriales?&backend=qdrant&k=3&section=objetivos&topic=vectorial
```

## � **Troubleshooting:**

**New endpoints not working?**

```bash
# Restart API with new endpoints
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

**Filters not working?**

```bash
# Check if data has metadata
curl "http://localhost:8080/ask?q=test&backend=qdrant&k=1" | python3 -m json.tool
```

**Manual demo not showing?**

```bash
# Verify dependencies
pip3 install --user sentence-transformers
```

---

**🎓 Perfect for classroom demonstration!** The manual demo endpoints explain each step in detail, and the metadata filtering allows precise document retrieval. Use the Spanish guide in `DEMO_MANUAL_VECTORES.md` for complete classroom instructions.
````
