# 🎓 Classroom Demo Guide

**Quick demo URLs for teaching vector databases in Spanish.**

## 🌐 Demo URLs (Copy-Paste Ready)

### Main Demo (Best for Projection)

```
http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales
```

Shows the complete process: text → vectors → search → results

### Step-by-Step Process

```
# 1. Show how text becomes vectors
http://localhost:8080/manual/embed?q=bases%20de%20datos%20vectoriales

# 2. Show how search works
http://localhost:8080/manual/search?q=bases%20de%20datos%20vectoriales&backend=qdrant&k=3

# 3. Compare both databases
http://localhost:8080/compare?q=bases%20de%20datos%20vectoriales
```

### Interactive Testing

```
# Basic search
http://localhost:8080/ask?q=vectores&backend=qdrant

# With filters
http://localhost:8080/ask?q=vectores&section=objetivos&document_type=pdf

# AI answer (requires Ollama)
http://localhost:8080/ai?q=¿Qué%20son%20las%20bases%20vectoriales?
```

## 📋 Demo Flow (15 minutes)

1. **Start with main demo URL** (5 min)
   - Explain text → vector conversion
   - Show similarity calculation
2. **Try different queries** (5 min)
   - Let students suggest Spanish queries
   - Show semantic vs keyword differences
3. **Show filters** (3 min)
   - Document type, section filtering
   - Practical relevance
4. **Compare databases** (2 min)
   - Qdrant vs PostgreSQL results
   - Speed and accuracy differences

## 💡 Teaching Points

- **Semantic Search:** Finds meaning, not just words
- **Spanish Support:** Works naturally with academic Spanish
- **Metadata Filtering:** Real-world relevance
- **Performance:** Millisecond search over thousands of documents

## 🔧 Setup Before Class

```bash
# 1. Make sure everything is running
docker compose up -d
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# 2. Test the main demo URL works
curl "http://localhost:8080/manual/demo?q=test"

# 3. Set browser to full screen, zoom to 150%
```

## 🎯 Key Concepts to Emphasize

1. **Vector = Meaning in Numbers**

   - 768 numbers capture semantic meaning
   - Similar concepts → similar numbers

2. **Cosine Similarity**

   - Measures "angle" between vectors
   - 0.9+ = very similar, <0.3 = unrelated

3. **Why It Works**

   - Trained on millions of Spanish documents
   - Finds synonyms, related concepts automatically

4. **Practical Applications**
   - Academic research
   - Document search
   - AI assistants
   - Recommendation systems

---

**💡 Pro Tip:** Keep the main demo URL bookmarked. It's designed to be clear and visible from the back of the classroom.
