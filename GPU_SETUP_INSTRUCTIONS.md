# GPU Setup for Docker Desktop on Ubuntu 22.04

## Current Situation

You're running **Docker Desktop** (not native Docker), which requires different GPU setup.

## Steps to Enable GPU Support

### 1. Enable GPU in Docker Desktop Settings

**Via GUI:**

1. Open Docker Desktop application
2. Go to Settings ⚙️ → Resources → GPU
3. Enable "Use GPU for Docker"
4. Click "Apply & Restart"

**Via CLI (Alternative):**

```bash
# Edit Docker Desktop settings
mkdir -p ~/.docker
cat > ~/.docker/daemon.json <<'EOF'
{
  "features": {
    "buildkit": true
  },
  "experimental": false
}
EOF
```

### 2. Test GPU Access

After enabling GPU in Docker Desktop:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see your RTX 5060 listed!

### 3. Start Your Services with GPU

Once GPU access works, restart your services:

```bash
docker compose down
docker compose up -d
```

### 4. Verify Ollama is Using GPU

```bash
# Check if Ollama sees the GPU
docker exec ollama nvidia-smi

# Test with a real model (phi3:mini)
curl http://localhost:8080/ai?q=hello&model=phi3:mini
```

## Alternative: Switch to Native Docker

If Docker Desktop GPU support doesn't work well, you can switch to native Docker:

```bash
# Use the host dockerd (native Docker)
docker context use host

# Reconfigure GPU support for native Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus nvidia nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Current Configuration Status

✅ NVIDIA Driver installed: 570.195.03  
✅ NVIDIA Container Toolkit installed: 1.18.0  
✅ GPU detected: RTX 5060 (8GB VRAM)  
✅ docker-compose.yml configured for GPU  
⚠️ Using Docker Desktop (requires GUI enable or context switch)

## Next Step

**Open Docker Desktop → Settings → Resources → GPU → Enable GPU support**

Then run:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```
