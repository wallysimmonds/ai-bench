#!/bin/bash
# setup-gb10.sh — Ollama install + model pull for lenovo-gb10 (Grace Blackwell)
# Usage: ./scripts/setup/setup-gb10.sh [host] [user]
# Note: DGX OS is Ubuntu-based, ARM architecture

HOST=${1:-"192.168.1.XX"}
USER=${2:-"ubuntu"}

echo "=== Setting up lenovo-gb10 (Grace Blackwell GB10): $USER@$HOST ==="

ssh $USER@$HOST << 'REMOTE'
set -e

echo "--- System info ---"
uname -m
nvidia-smi 2>/dev/null || echo "nvidia-smi not in PATH — checking..."
which nvidia-smi || find /usr -name nvidia-smi 2>/dev/null | head -3

echo "--- Installing Ollama ---"
if ! command -v ollama &> /dev/null; then
    # ARM build — Ollama supports arm64 natively
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable --now ollama
    sleep 3
fi

echo "Ollama version: $(ollama --version)"

# GB10 unified memory config
sudo mkdir -p /etc/systemd/system/ollama.service.d
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_MODELS=/home/ubuntu/ollama-models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3

mkdir -p ~/ollama-models

echo "--- Pulling models for lenovo-gb10 ---"
# Start conservative — confirm unified memory allocation works
# before pulling the large models

ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen3.5:9b

echo "--- Smoke test before pulling large models ---"
ollama run qwen2.5-coder:7b "respond with only: ok" --nowordwrap

echo ""
echo "--- Small models working. Pulling mid-tier ---"
ollama pull qwen2.5-coder:32b
ollama pull qwen3.5:27b
ollama pull qwen3.5:35b-a3b

echo ""
echo "--- Pulling large models (watch memory with nvidia-smi) ---"
ollama pull qwen3.5:72b
ollama pull qwen3-coder-next:80b-a3b-q4_K_M

echo "--- Model list ---"
ollama list

echo "=== lenovo-gb10 setup complete ==="
echo "NOTE: Verify CUDA/unified memory allocation is correct with:"
echo "  nvidia-smi"
echo "  ollama ps  (while a model is loaded)"
REMOTE
