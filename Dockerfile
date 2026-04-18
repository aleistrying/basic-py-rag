FROM python:3.9-slim
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with pip cache mount for faster rebuilds.
#
# WHY CPU-only torch:
#   All LLM inference is handled by the Ollama container over HTTP — the app
#   container itself only runs sentence-transformers for embeddings, which run
#   perfectly on CPU.  Pulling CUDA-enabled torch would download ~2.5 GB of
#   NVIDIA libraries (cublas, cudnn, cusparse…) that are never used and break
#   on Apple Silicon / ARM hosts.  CPU torch is ~200 MB and identical in
#   behaviour for this workload.
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && \
    apt-get install -y docker.io && \
    # Install everything EXCEPT torch/torchvision/torchaudio (strip the CUDA
    # index-url lines and torch packages so we can re-install them below).
    grep -vE "^\s*(--index-url|torch|torchvision|torchaudio)" requirements.txt \
        | pip install -r /dev/stdin && \
    # Now install CPU-only PyTorch — skips all NVIDIA packages entirely.
    pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.0.0" torchvision torchaudio && \
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