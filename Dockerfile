FROM python:3.9-slim
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with pip cache mount for faster rebuilds
# Also install Docker CLI for container management
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && \
    apt-get install -y docker.io && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Pre-download embedding model for fully offline operation
# This downloads the model during build so it's available offline
ENV HF_HOME=/app/models/huggingface
ENV TRANSFORMERS_OFFLINE=0
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    import os; \
    os.makedirs('/app/models/embeddings', exist_ok=True); \
    print('Downloading embedding model...'); \
    model = SentenceTransformer('intfloat/multilingual-e5-base', cache_folder='/app/models/embeddings'); \
    print('Model downloaded successfully!');" || echo "Model download failed, will download on first use"

# Copy the rest of the application
# Use build arg to bust cache when needed
ARG CACHE_BUST
RUN echo "Cache bust: $CACHE_BUST"
COPY . .

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]