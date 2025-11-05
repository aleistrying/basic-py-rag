# 🚀 Essential Commands

## 🐳 Docker Services

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Restart
docker compose restart

# Stop & clean
docker compose down -v
```

## ✅ Service Verification

### Qdrant

```bash
# Check status
curl -s http://localhost:6333/collections | jq

# Web interface
open http://localhost:6333/dashboard
```

### PostgreSQL

```bash
# Connect to DB
psql postgresql://pguser:pgpass@localhost:5432/vectordb

# Check pgvector extension
docker compose exec postgres psql -U pguser -d vectordb -c "\dx"
```

## 🔧 Useful Operations

```bash
# Clean restart
docker compose down -v && docker compose up -d --build

# Monitor resources
docker stats

# Check port usage
sudo lsof -i :6333  # Qdrant
sudo lsof -i :5432  # PostgreSQL

# Test connectivity
pg_isready -h localhost -p 5432 -U pguser
curl -f http://localhost:6333/health
```

## 🌐 Quick Access URLs

| Service          | URL                               | Purpose  |
| ---------------- | --------------------------------- | -------- |
| Qdrant API       | `http://localhost:6333`           | REST API |
| Qdrant Dashboard | `http://localhost:6333/dashboard` | Web UI   |
| PostgreSQL       | `localhost:5432`                  | Database |
| RAG API          | `http://localhost:8080`           | FastAPI  |
