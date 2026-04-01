#!/bin/bash
# setup-nvidia.sh — Ollama install + model pull for nvidia-ai node
# Usage: ./scripts/setup/setup-nvidia.sh [host] [user]

HOST=${1:-"192.168.1.XX"}
USER=${2:-"johanus"}

echo "=== Setting up nvidia-ai node: $USER@$HOST ==="

ssh $USER@$HOST << 'REMOTE'
set -e

echo "--- Installing Ollama ---"
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable --now ollama
    sleep 3
fi

echo "Ollama version: $(ollama --version)"

# Set model storage path (adjust if you have a dedicated drive)
mkdir -p ~/ollama-models
sudo mkdir -p /etc/systemd/system/ollama.service.d
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_MODELS=/home/johanus/ollama-models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3

echo "--- Pulling models for nvidia-ai (40GB VRAM) ---"

# Common models
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen3.5:9b

# Mid-tier — fits in 40GB with dual GPU
ollama pull qwen2.5-coder:32b
ollama pull qwen3.5:35b-a3b   # MoE, very efficient

echo "--- Model list ---"
ollama list

echo "--- Quick smoke test ---"
ollama run qwen2.5-coder:7b "respond with only: ok" --nowordwrap

echo "=== nvidia-ai setup complete ==="
REMOTE
