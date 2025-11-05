# 🚀 RAG Demo - Qdrant vs pgvector

Sistema RAG para comparar **Qdrant** vs **PostgreSQL+pgvector** en búsqueda vectorial.

## 🆕 Latest Improvements (November 2025)

✅ **Multilingual E5 Embeddings** - Better Spanish query support  
✅ **Automatic Schema Setup** - PostgreSQL initializes on startup  
✅ **Cosine Similarity** - Fixed scoring (now 0-1 range, not >1)  
✅ **Query Expansion** - Synonyms for better Spanish retrieval  
✅ **Smart Chunking** - 50-word overlap for context preservation  
✅ **Keyword Boosting** - Lexical signals improve ranking

📚 **See**: [IMPROVEMENTS.md](IMPROVEMENTS.md) | [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

## ⚡ Quick Start

```bash
# 1. Start databases
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest data
python scripts/ingest_qdrant.py
python scripts/ingest_pgvector.py

# 4. Start API
source .venv/bin/activate
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# 5. Test API
curl "http://localhost:8080/ask?q=vectores&backend=qdrant&k=3" | python -m json.tool
curl "http://localhost:8080/compare?q=vectores&k=3" | python -m json.tool
curl "http://localhost:8080/ai?q=¿Si hoy es 17 de noviembre, cual es la siguiente tarea para entregar en la clase?&backend=qdrant&k=3" | python -m json.tool
```

## 📁 Project Structure

```
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── rag.py             # RAG logic
│   └── *_backend.py       # Database backends
├── scripts/               # Data ingestion
├── data/raw/              # Source documents
└── requirements.txt
```

## 🔥 API Endpoints

### Single Backend Search

```bash
curl "http://localhost:8080/ask?q=vectores&backend=qdrant&k=3"
```

### AI-Powered RAG (requires Ollama)

```bash
#Que es el hipercubo? PG VS cuadrant
http://localhost:8080/ai?q=%C2%BFque%20es%20el%20hipercubo?&backend=qdrant&k=3

http://localhost:8080/ai?q=%C2%BFque%20es%20el%20hipercubo?&backend=pgvector&k=3

# Cuales son los objetivos del curso de base de datos avanzados 8b
http://localhost:8080/ai?q=%C2%BFcuales%20son%20los%20objetivos%20del%20curso%20de%20base%20de%20datos%20avanzados&backend=qdrant&k=3&model=llama3.1:8b

curl "http://localhost:8080/ai?q=¿Qué son las bases de datos vectoriales?&backend=qdrant&k=3&model=phi3:mini"
```

http://localhost:8080/ai?q=%C2%BFSi%20hoy%20es%2017%20de%20noviembre,%20cual%20es%20la%20siguiente%20tarea%20para%20entregar%20en%20la%20clase?&backend=qdrant&k=3&model=gemma2:2b

### Side-by-Side Comparison

```bash
curl "http://localhost:8080/compare?q=vectores&k=3"
```

### Example Responses

#### RAG Search Response

```json
{
  "query": "vectores",
  "backend": "QDRANT",
  "total_results": 3,
  "results": [
    {
      "document": "introduccion_vectores.md",
      "similarity": "0.407",
      "preview": "Las bases de datos vectoriales son sistemas..."
    }
  ]
}
```

#### AI-Powered Response

```json
{
  "query": "¿Qué son las bases de datos vectoriales?",
  "backend": "QDRANT",
  "model": "phi3:mini",
  "ai_response": "Las bases de datos vectoriales son sistemas especializados diseñados para almacenar y consultar vectores de alta dimensión de manera eficiente...",
  "total_results": 3,
  "sources": [...],
  "fallback_mode": false
}
```

## ⚖️ Backend Comparison

| Feature                | Qdrant        | PostgreSQL+pgvector |
| ---------------------- | ------------- | ------------------- |
| **Specialization**     | Vector-native | SQL + vectors       |
| **API**                | REST          | SQL queries         |
| **ACID**               | ❌            | ✅                  |
| **Metadata filtering** | Advanced      | JSONB               |
| **Ecosystem**          | ML-focused    | Enterprise SQL      |

## 🤖 Ollama Setup (Optional - for AI endpoints)

### Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download
```

### Pull and Run Models

```bash
# Pull phi3:mini (recommended, ~2GB)
ollama pull phi3:mini
curl http://localhost:11434/api/pull -d '{"name":"llama3.1:8b-instruct"}'

# Alternative models
ollama pull llama2:7b-chat    # ~4GB
ollama pull mistral:7b        # ~4GB

# List available models
ollama list
```

### Start Ollama Service

```bash
# Start Ollama (usually auto-starts)
ollama serve

# Test Ollama is working
curl http://localhost:11434/api/tags
```

### Memory Issues?

If you get "model requires more system memory":

```bash
# Use smaller models
ollama pull phi3:mini         # Smallest option
ollama pull qwen2:0.5b       # Even smaller if available

# Or configure Ollama with less GPU memory
export OLLAMA_NUM_GPU=0      # Use CPU only
ollama serve
```

## 🛠️ Useful Commands

```bash
# Check services
docker compose ps

# View logs
docker compose logs -f

# Clean restart
docker compose down -v && docker compose up -d

# Count documents
curl -s http://localhost:6333/collections/docs_qdrant | jq '.result.points_count'

# Test Ollama
curl http://localhost:11434/api/tags
```

## � Links

- [Qdrant Docs](https://qdrant.tech/documentation/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
