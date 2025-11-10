#!/bin/bash

echo "🚀 Setting up NVIDIA GPU support for Docker on Ubuntu 22.04"
echo "============================================================"

# Check if nvidia-smi works
echo "📊 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

nvidia-smi
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo ""

# Install NVIDIA Container Toolkit using the correct method for Ubuntu 22.04
echo "📦 Installing NVIDIA Container Toolkit..."
echo ""

# Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "📥 Updating package list..."
sudo apt-get update

echo "📥 Installing nvidia-container-toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
echo "⚙️  Configuring Docker to use NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
echo "🔄 Restarting Docker daemon..."
sudo systemctl restart docker

# Test GPU access in Docker
echo ""
echo "🧪 Testing GPU access in Docker..."
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi; then
    echo ""
    echo "✅ SUCCESS! GPU is accessible from Docker containers"
    echo ""
    echo "📝 Next steps:"
    echo "1. The GPU configuration is already enabled in docker-compose.yml"
    echo "2. Run: docker compose down && docker compose up -d"
    echo "3. Your Ollama container will now use the GPU!"
else
    echo ""
    echo "❌ GPU test failed. Please check the errors above."
    echo "💡 You may need to:"
    echo "   - Reboot your system"
    echo "   - Check NVIDIA driver installation"
    echo "   - Verify Docker permissions (add user to docker group)"
fi

echo ""
echo "🔍 To verify GPU is being used by Ollama:"
echo "   docker exec ollama nvidia-smi"
echo ""
