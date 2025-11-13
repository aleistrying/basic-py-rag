#!/bin/bash
# 🚀 Complete Demo Script - Shows the full RAG pipeline working
# This script demonstrates the entire workflow from documents to AI answers

echo "🚀 Vector Database RAG System Demo"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local service=$2
    echo -e "${YELLOW}⏳ Waiting for $service to start...${NC}"
    
    for i in {1..30}; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $service is ready!${NC}"
            return 0
        fi
        sleep 2
    done
    
    echo -e "${RED}❌ $service failed to start${NC}"
    return 1
}

# Step 1: Check prerequisites
echo -e "${BLUE}📋 Step 1: Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if ! command_exists python; then
    echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found. Are you in the right directory?${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

# Step 2: Start databases
echo -e "${BLUE}📦 Step 2: Starting databases...${NC}"
docker compose up -d --quiet-pull

# Wait for services
wait_for_service "http://localhost:6333/collections" "Qdrant"
wait_for_service "http://localhost:5432" "PostgreSQL" || true  # PostgreSQL check might fail, that's ok

echo ""

# Step 3: Install Python dependencies (if not installed)
echo -e "${BLUE}🐍 Step 3: Installing Python dependencies...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv .venv
fi

source .venv/bin/activate 2>/dev/null || true
pip install -q -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 4: Process documents
echo -e "${BLUE}📄 Step 4: Processing documents and creating embeddings...${NC}"

if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠️  No documents found in data/raw/. Using existing processed data.${NC}"
else
    echo -e "${YELLOW}Processing PDFs and creating vector embeddings...${NC}"
    echo -e "${YELLOW}This may take a few minutes depending on document size...${NC}"
    python scripts/main_pipeline.py --clear
fi

echo -e "${GREEN}✅ Document processing complete${NC}"
echo ""

# Step 5: Start API server
echo -e "${BLUE}🌐 Step 5: Starting API server...${NC}"

# Start API in background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 &
API_PID=$!

# Wait for API to be ready
wait_for_service "http://localhost:8080/" "RAG API"

echo ""

# Step 6: Run demo queries
echo -e "${BLUE}🔍 Step 6: Testing the system with example queries...${NC}"
echo ""

# Test 1: Basic search
echo -e "${YELLOW}Test 1: Basic semantic search${NC}"
echo "Query: 'bases de datos vectoriales'"
echo ""
curl -s "http://localhost:8080/ask?q=bases%20de%20datos%20vectoriales&backend=qdrant&k=2" | \
    python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('Results found:', data.get('total_results', 'unknown'))
    for i, result in enumerate(data.get('results', [])[:2]):
        print(f'  {i+1}. {result.get(\"source\", \"unknown\")} (similarity: {result.get(\"similarity\", \"unknown\")})')
        preview = result.get('content', result.get('preview', ''))[:100]
        print(f'     \"{preview}...\"')
except:
    print('API response received but could not parse JSON')
"
echo ""

# Test 2: Backend comparison  
echo -e "${YELLOW}Test 2: Comparing Qdrant vs PostgreSQL${NC}"
echo "Query: 'vectores'"
echo ""
curl -s "http://localhost:8080/compare?q=vectores&k=2" | \
    python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('Qdrant results:', len(data.get('qdrant_results', [])))
    print('PostgreSQL results:', len(data.get('pgvector_results', [])))
except:
    print('Comparison results received')
"
echo ""

# Test 3: AI-powered answer (if Ollama available)
echo -e "${YELLOW}Test 3: AI-powered answer (if Ollama available)${NC}"
echo "Query: '¿Qué son las bases de datos vectoriales?'"
echo ""

if curl -s "http://localhost:11434/api/tags" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama detected, testing AI endpoint...${NC}"
    curl -s "http://localhost:8080/ai?q=¿Qué%20son%20las%20bases%20de%20datos%20vectoriales?&backend=qdrant&k=2&model=phi3:mini" | \
        python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    ai_response = data.get('ai_response', '')
    if ai_response:
        print('AI Response preview:')
        print(f'  \"{ai_response[:200]}...\"')
        print(f'Sources used: {len(data.get(\"sources\", []))}')
    else:
        print('AI response generated but could not extract preview')
except:
    print('AI endpoint responded')
"
else
    echo -e "${YELLOW}⚠️  Ollama not available. Install Ollama to test AI features.${NC}"
    echo "   Visit: https://ollama.com/"
fi
echo ""

# Step 7: Show access URLs
echo -e "${BLUE}🎓 Step 7: Access points for exploration${NC}"
echo ""
echo -e "${GREEN}✅ Demo complete! Your system is running.${NC}"
echo ""
echo "🌐 API Endpoints:"
echo "   • Main API: http://localhost:8080/"
echo "   • Search: http://localhost:8080/ask?q=vectores&backend=qdrant"
echo "   • Compare: http://localhost:8080/compare?q=vectores" 
echo "   • Classroom Demo: http://localhost:8080/manual/demo?q=bases%20de%20datos%20vectoriales"
echo ""
echo "📊 Database Admin:"
echo "   • Qdrant: http://localhost:6333/dashboard" 
echo "   • PostgreSQL: localhost:5432 (user: postgres, password: password)"
echo ""
echo "🛠️ Management:"
echo "   • Stop databases: docker compose down"
echo "   • View logs: docker compose logs -f"
echo "   • Reprocess documents: python scripts/main_pipeline.py --force"
echo ""

# Wait for user input before cleanup
echo -e "${YELLOW}Press Enter to stop the demo, or Ctrl+C to keep it running...${NC}"
read -r

# Cleanup
echo -e "${BLUE}🧹 Cleaning up...${NC}"
kill $API_PID 2>/dev/null || true
echo -e "${GREEN}✅ Demo stopped. Databases are still running.${NC}"
echo "   Use 'docker compose down' to stop databases."