# 🚀 SIMPLE RAG DEMO - Single Pipeline

## 📋 What You Have

Your RAG system compares **Qdrant** vs **PostgreSQL** for Spanish academic queries using a single pipeline script.

### **🔥 Quick Demo Commands (All-in-One):**

```bash
# 1. Start databases
docker compose up -d

# 2. Install dependencies (if needed)
pip3 install --user numpy sentence-transformers fastapi uvicorn psycopg2-binary qdrant-client pdfplumber

# 3. Run complete pipeline (processes PDFs + tables + embedding + databases)
python3 scripts/pipeline.py --clear

# 4. Start API (shows "Uvicorn running on...")
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# 5. Test queries (in another terminal)
curl "http://localhost:8080/ask?q=objetivos del curso&backend=qdrant"

# web test
http://localhost:8080/ai?q=%C2%BFSi%20hoy%20es%20%2025%20de%20octubre,%20cual%20es%20la%20siguiente%20tarea%20para%20entregar%20en%20la%20clase?&backend=qdrant&k=3
```

## 🎯 **Demo Scenarios**

### Basic Search Demo:

```bash
# Compare backends for Spanish query
curl "http://localhost:8080/compare?q=bases de datos vectoriales&k=3" | python3 -m json.tool
```

### AI Chat Demo (if you have Ollama):

```bash
# Ask AI with RAG context
curl "http://localhost:8080/ai?q=¿Cuándo se entrega el proyecto 1?&backend=qdrant&k=3" | python3 -m json.tool
```

### Performance Test:

```bash
# Test different query types
curl "http://localhost:8080/ask?q=NoSQL&backend=qdrant"        # English term
curl "http://localhost:8080/ask?q=bases vectoriales&backend=qdrant"  # Spanish
curl "http://localhost:8080/ask?q=evaluación&backend=qdrant"   # Accents
```

## 📁 **Essential Files (Simplified):**

```
scripts/
└── pipeline.py        # 🎯 EVERYTHING: PDF+tables+chunks+embeddings+DB in ONE file

app/
├── main.py           # 🌐 API server (FastAPI)
└── rag.py            # 🤖 RAG logic

docker-compose.yml     # 🐳 Qdrant + PostgreSQL
```

## 🔧 **What the Single Pipeline Does:**

1. **PDF Processing** → Clean text + table extraction with watermark removal
2. **Text Processing** → Handle .txt and .md files
3. **Smart Chunking** → 350 tokens with overlap (memory-optimized)
4. **E5 Embeddings** → Spanish-friendly multilingual model (768 dims)
5. **Database Management** → Auto-clears and loads both Qdrant + PostgreSQL
6. **Table Extraction** → Extracts schedule tables from PDFs properly

## ⚡ **Most Important Improvements:**

- ✅ **Spanish Support:** E5 multilingual model instead of English-only MiniLM
- ✅ **Proper Similarity:** Cosine similarity (0-1 range) instead of broken >1 scores
- ✅ **Auto Setup:** PostgreSQL schema creates automatically
- ✅ **Smart Chunking:** Token-aware with overlap for better context

## 🎪 **Live Demo Steps:**

1. **Start:** `docker compose up -d` (wait 30 seconds)
2. **Load Data:** `python3 scripts/ingest_all.py` (wait 2-5 minutes)
3. **Start API:** `python3 -m app.main`
4. **Test:** Open http://localhost:8080/docs (FastAPI docs)
5. **Query:** Try the curl commands above

## 💡 **Troubleshooting:**

**🧠 Memory Issues (PC freezing)?**

```bash
# Use step-by-step approach instead of ingest_all.py
python3 scripts/pdf_cleaner.py     # Light
python3 scripts/chunker.py         # Light
python3 scripts/embed_and_upsert.py  # Uses 8-item batches
```

**Dependencies missing?**

```bash
pip3 install --user numpy sentence-transformers fastapi uvicorn
```

**Want to see config?**

```bash
python3 scripts/ingest_all.py --config
```

**Start fresh?**

```bash
python3 scripts/ingest_all.py --clean  # Remove all processed data
```

**API not starting?**

```bash
# Should show "Uvicorn running on http://0.0.0.0:8080"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## 🎯 **Achieved Results:**

- ✅ **Watermark removal working** - UTP-FISC patterns filtered out
- ✅ **Clean text processing** - 6 readable chunks from .txt files
- ✅ **Memory optimized** - Small 8-item batches, no PC crashes
- ✅ **Spanish content ready** - Academic course material properly processed

## 🔍 **Content Quality Comparison:**

**Before (corrupted PDF):**

```
"U C - F 2 S 2 5 P - S T 2 I 0 T 2 U C - F 2 P - S Universidad..."
```

**After (clean text):**

```
"OBJETIVOS DEL CURSO El objetivo principal es proporcionar conocimientos
avanzados sobre: • Bases de datos NoSQL • Bases de datos vectoriales..."
```

---

**That's it!** Everything else is just supporting files. Focus on `ingest_all.py` → `main.py` → curl tests! 🎉
